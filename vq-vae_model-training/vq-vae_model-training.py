import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import json
import csv
import argparse

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_embeddings, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(embedding_dim, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        flattened = x.view(-1, self.embedding_dim)
        distances = torch.cdist(flattened, self.embedding.weight)
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(x.size())

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        return quantized, loss, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss, _ = self.vq_layer(encoded)
        decoded = self.decoder(quantized)
        return decoded, vq_loss

class SampledImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Returning 0 as a placeholder label

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def setup_directories(config):
    output_dir = os.path.join(config['output_base_dir'], config['identifier'])
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_dataloader(config):
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(config['normalize_mean'], config['normalize_std'])
    ])

    all_image_paths = []
    for dataset_dir in config['dataset_dirs']:
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(root, file))

    sampled_image_paths = random.sample(all_image_paths, config['total_samples'])
    sampled_dataset = SampledImageDataset(sampled_image_paths, transform=transform)
    dataloader = DataLoader(sampled_dataset, batch_size=config['batch_size'], shuffle=True)
    return dataloader

def setup_model_and_optimizer(config, device):
    model = VQVAE(in_channels=3, hidden_channels=config['hidden_channels'],
                  num_embeddings=config['num_embeddings'],
                  embedding_dim=config['embedding_dim'],
                  commitment_cost=config['commitment_cost']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    return model, optimizer

def save_training_params(config, output_dir):
    params_path = os.path.join(output_dir, 'training_params.json')
    with open(params_path, 'w') as f:
        json.dump(config, f)
    print(f"Training parameters saved to {params_path}")

def setup_logging(output_dir):
    log_path = os.path.join(output_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    return log_path

def train_model(model, optimizer, dataloader, config, device, output_dir, log_path):
    criterion = nn.MSELoss()
    total_iterations = config['num_epochs'] * len(dataloader)
    progress_bar = tqdm(total=total_iterations, desc="Training Progress")

    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0

        for images, _ in dataloader:
            images = images.to(device)

            reconstructed, vq_loss = model(images)
            recon_loss = criterion(reconstructed, images)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_description(f"Epoch [{epoch + 1}/{config['num_epochs']}]")

        avg_loss = running_loss / len(dataloader)
        progress_bar.set_postfix(Loss=avg_loss)

        with open(log_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'loss'])
            writer.writerow({'epoch': epoch + 1, 'loss': avg_loss})

        if (epoch + 1) % config['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    progress_bar.close()

def save_model(model, output_dir, config):
    model_output_dir = os.path.join(output_dir, 'models')
    os.makedirs(model_output_dir, exist_ok=True)
    model_save_path = os.path.join(model_output_dir, f"vq-vae_model_{config['identifier']}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def generate_images(model, dataloader, device, output_dir, config):
    img_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        sample_images, _ = next(iter(dataloader))
        sample_images = sample_images.to(device)
        reconstructed, _ = model(sample_images)

    progress_bar = tqdm(total=config['batch_size'], desc="Generating Images")

    for i in range(config['batch_size']):
        img = reconstructed[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize the image to [0, 1]

        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis('off')

        img_output_path = os.path.join(img_output_dir, f'output_image_{i}.png')
        plt.savefig(img_output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        progress_bar.update(1)

    progress_bar.close()
    print(f"Generated and saved {config['batch_size']} images to {img_output_dir}")

def main(config_path):
    config = load_config(config_path)
    output_dir = setup_directories(config)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    dataloader = setup_dataloader(config)
    model, optimizer = setup_model_and_optimizer(config, device)
    
    save_training_params(config, output_dir)
    log_path = setup_logging(output_dir)
    
    train_model(model, optimizer, dataloader, config, device, output_dir, log_path)
    save_model(model, output_dir, config)
    generate_images(model, dataloader, device, output_dir, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-VAE Model Training")
    parser.add_argument("config_path", type=str, help="Path to the configuration JSON file")
    args = parser.parse_args()
    
    main(args.config_path)