import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
import argparse
import logging

def setup_logging(output_dir):
    logging.basicConfig(filename=os.path.join(output_dir, 'interpolation.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def slerp(val, low, high):
    """Spherical linear interpolation for 4D tensors."""
    low_2d = low.view(low.shape[0], -1)
    high_2d = high.view(high.shape[0], -1)
    low_2d_norm = low_2d / torch.norm(low_2d, dim=1, keepdim=True)
    high_2d_norm = high_2d / torch.norm(high_2d, dim=1, keepdim=True)
    omega = torch.acos((low_2d_norm * high_2d_norm).sum(1).clamp(-1, 1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low_2d + \
          (torch.sin(val * omega) / so).unsqueeze(1) * high_2d
    return res.view(low.shape)

def load_and_preprocess_image(image_path, device, max_image_dimension):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    aspect_ratio = original_size[0] / original_size[1]
    
    if aspect_ratio > 1:
        new_width = min(original_size[0], max_image_dimension)
        new_width = new_width - (new_width % 8)
        new_height = int(new_width / aspect_ratio)
        new_height = new_height - (new_height % 8)
    else:
        new_height = min(original_size[1], max_image_dimension)
        new_height = new_height - (new_height % 8)
        new_width = int(new_height * aspect_ratio)
        new_width = new_width - (new_width % 8)
    
    new_size = (new_width, new_height)
    
    transform = Compose([
        Resize(new_size),
        ToTensor(),
        Normalize([0.5], [0.5])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device).to(torch.float16)
    
    return image_tensor, new_size, original_size

def load_model(model_id, device):
    logging.info("Loading Stable Diffusion model...")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    if device.type == "cuda":
        pipe.enable_attention_slicing()
    logging.info("Model loaded successfully.")
    return pipe

def preprocess_images(image_paths, device, max_image_dimension):
    logging.info("Preprocessing images...")
    images = []
    sizes = []
    original_sizes = []
    for image_path in tqdm(image_paths, desc="Loading and preprocessing images"):
        image, size, original_size = load_and_preprocess_image(image_path, device, max_image_dimension)
        images.append(image)
        sizes.append(size)
        original_sizes.append(original_size)
    logging.info(f"Processed sizes: {sizes}")
    logging.info(f"Original sizes: {original_sizes}")
    return images, sizes, original_sizes

def encode_latents(pipe, images):
    logging.info("Encoding images to latent representations...")
    latents = []
    with torch.no_grad():
        for image in tqdm(images, desc="Encoding images to latent representations"):
            latent = pipe.vae.encode(image).latent_dist.sample() * 0.18215
            latents.append(latent)
    logging.info(f"Shapes of latents: {[latent.shape for latent in latents]}")
    return latents

def interpolate_latents(latents, num_steps, pipe, output_dir, output_image_size):
    logging.info("Interpolating between latents...")
    alphas = np.linspace(0, 1, num_steps)
    interpolated_images = []
    
    for alpha in tqdm(alphas, desc="Interpolating and Decoding"):
        interpolated_latent = slerp(alpha, latents[0], latents[-1])
        with torch.no_grad():
            interpolated_latent = 1 / 0.18215 * interpolated_latent
            decoded_image = pipe.vae.decode(interpolated_latent).sample
            decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
            decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).numpy()[0]
        
        decoded_image_pil = Image.fromarray((decoded_image * 255).astype(np.uint8))
        decoded_image_pil = decoded_image_pil.resize(output_image_size, Image.LANCZOS)
        
        interpolated_images.append(decoded_image_pil)
        
        output_path = os.path.join(output_dir, f"interpolated_{len(interpolated_images)}.png")
        decoded_image_pil.save(output_path, quality=95)
    
    logging.info(f"Interpolation complete. {num_steps} images generated and saved in {output_dir}")
    return interpolated_images

def plot_results(interpolated_images, output_dir):
    logging.info("Plotting results...")
    fig, axes = plt.subplots(1, len(interpolated_images), figsize=(20, 4))
    for ax, img in zip(axes, interpolated_images):
        ax.imshow(img)
        ax.axis('off')
    
    plot_path = os.path.join(output_dir, "interpolation_steps.png")
    fig.savefig(plot_path, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_path}")

def run_interpolation(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    os.makedirs(config['output_dir'], exist_ok=True)
    setup_logging(config['output_dir'])

    try:
        pipe = load_model(config['model_id'], device)
        images, sizes, original_sizes = preprocess_images(config['image_paths'], device, config['max_image_dimension'])
        latents = encode_latents(pipe, images)
        interpolated_images = interpolate_latents(latents, config['num_steps'], pipe, config['output_dir'], config['output_image_size'])
        plot_results(interpolated_images, config['output_dir'])
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    
    run_interpolation(config)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])