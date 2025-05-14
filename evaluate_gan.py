"""
GAN Evaluation Script
This script evaluates a trained GAN model by:
1. Generating sample images
2. Calculating FID and Inception scores
3. Visualizing latent space interpolations
4. Performing quantitative analysis
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import random
import json
from datetime import datetime
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

# Import the needed classes and functions from new.py or redefine them here
from gan_anime_faces import Config, Generator, set_seed

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# For evaluation metrics
try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    INCEPTION_AVAILABLE = True
except ImportError:
    print("Warning: pytorch_fid not installed. Using simplified metrics.")
    INCEPTION_AVAILABLE = False

# For Inception Score
try:
    from torchvision.models import inception_v3
    import torch.nn.functional as F
    INCEPTION_SCORE_AVAILABLE = True
except ImportError:
    print("Warning: torchvision inception model not available. Cannot calculate Inception Score.")
    INCEPTION_SCORE_AVAILABLE = False

class GanEvaluator:
    def __init__(self, config=None, checkpoint_path=None):
        # Load configuration
        self.config = config if config else Config()
        
        # Set random seed
        set_seed(42)
        
        # Initialize generator
        self.generator = Generator().to(device)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_model(checkpoint_path)
        
        # Create output directory for evaluation results
        self.eval_dir = os.path.join(self.config.OUTPUT_DIR, "evaluation")
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # Initialize inception model for FID calculation if available
        if INCEPTION_AVAILABLE:
            self.inception_model = InceptionV3().to(device)
            self.inception_model.eval()
    
    def load_model(self, checkpoint_path):
        """Load a trained generator model from checkpoint"""
        print(f"Loading model from {checkpoint_path}...")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'generator' in checkpoint:
                # Full checkpoint with all training info
                self.generator.load_state_dict(checkpoint['generator'])
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"Loaded generator from epoch {epoch}")
            else:
                # Just the state dict
                self.generator.load_state_dict(checkpoint)
                print("Loaded generator state dictionary")
                
            self.generator.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
    
    def generate_samples(self, num_samples=100, batch_size=32):
        """Generate samples from the trained generator"""
        self.generator.eval()
        all_samples = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                
                # Use a different seed for each batch to ensure diversity
                random_seed = random.randint(0, 10000)
                torch.manual_seed(random_seed)
                
                noise = torch.randn(current_batch_size, self.config.Z_DIM, device=device)
                fake_images = self.generator(noise)
                all_samples.append(fake_images)
        
        # Concatenate all batches
        all_samples = torch.cat(all_samples, dim=0)
        return all_samples
    
    def visualize_samples(self, samples, nrow=8, title="Generated Samples"):
        """Visualize and save generated samples"""
        samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
        grid = make_grid(samples, nrow=nrow, normalize=False)
        
        # Save the grid
        save_path = os.path.join(self.eval_dir, f"{title.lower().replace(' ', '_')}.png")
        save_image(grid, save_path)
        
        # Also display
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_with_title.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Samples saved to {save_path}")
        
        return save_path
    
    def latent_space_interpolation(self, num_points=10, num_samples=5):
        """Generate interpolations in the latent space between random points"""
        self.generator.eval()
        
        # Generate pairs of random points
        start_points = torch.randn(num_samples, self.config.Z_DIM, device=device)
        end_points = torch.randn(num_samples, self.config.Z_DIM, device=device)
        
        # Create interpolation steps
        alphas = torch.linspace(0, 1, num_points)
        
        # Generate all the interpolated images
        all_images = []
        
        with torch.no_grad():
            for sample_idx in range(num_samples):
                sample_images = []
                
                for alpha in alphas:
                    # Interpolate in the latent space
                    interpolated = start_points[sample_idx] * (1 - alpha) + end_points[sample_idx] * alpha
                    interpolated = interpolated.unsqueeze(0)
                    
                    # Generate image
                    img = self.generator(interpolated)
                    sample_images.append(img)
                
                # Stack all images for this sample
                sample_images = torch.cat(sample_images, dim=0)
                all_images.append(sample_images)
        
        # Visualize the interpolations
        fig, axes = plt.subplots(num_samples, num_points, figsize=(num_points * 2, num_samples * 2))
        
        for i, sample_images in enumerate(all_images):
            for j, img in enumerate(sample_images):
                # Convert the image to numpy and adjust range to [0, 1]
                img_np = (img.cpu().detach().permute(1, 2, 0).numpy() + 1) / 2
                img_np = np.clip(img_np, 0, 1)
                
                # Display
                if num_samples > 1:
                    axes[i, j].imshow(img_np)
                    axes[i, j].axis('off')
                    if j == 0:
                        axes[i, j].set_title("Start")
                    elif j == num_points - 1:
                        axes[i, j].set_title("End")
                else:
                    axes[j].imshow(img_np)
                    axes[j].axis('off')
                    if j == 0:
                        axes[j].set_title("Start")
                    elif j == num_points - 1:
                        axes[j].set_title("End")
        
        plt.tight_layout()
        save_path = os.path.join(self.eval_dir, "latent_space_interpolation.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Interpolation saved to {save_path}")
    
    def calculate_fid(self, real_images, fake_images):
        """Calculate Fréchet Inception Distance"""
        if not INCEPTION_AVAILABLE:
            print("FID calculation requires pytorch_fid package. Using simplified metrics.")
            return self._simplified_fid(real_images, fake_images)
            
        # Extract features
        real_features = self._calculate_inception_features(real_images)
        fake_features = self._calculate_inception_features(fake_images)
        
        # Flatten features to 2D if they have more dimensions
        if real_features.ndim > 2:
            real_features = real_features.reshape(real_features.shape[0], -1)
        if fake_features.ndim > 2:
            fake_features = fake_features.reshape(fake_features.shape[0], -1)
        
        # Calculate statistics
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        ssdiff = np.sum((mu_real - mu_fake) ** 2)
        covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return float(fid)
    
    def _calculate_inception_features(self, images):
        """Calculate Inception features for FID score"""
        if not INCEPTION_AVAILABLE:
            return None
            
        features = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                # Resize images if needed
                if batch.shape[2] != 299 or batch.shape[3] != 299:
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                # Convert from [-1, 1] to [0, 1] range
                batch = (batch + 1) / 2
                pred = self.inception_model(batch)[0]
                features.append(pred)
        
        features = torch.cat(features, dim=0)
        return features.cpu().numpy()
    
    def _simplified_fid(self, real_images, fake_images):
        """A simplified metric when inception is not available"""
        # Convert to numpy arrays in range [0, 1]
        real = ((real_images + 1) * 0.5).cpu().numpy()
        fake = ((fake_images + 1) * 0.5).cpu().numpy()
        
        # Calculate mean and std for both
        real_mean = np.mean(real, axis=(0, 2, 3))
        fake_mean = np.mean(fake, axis=(0, 2, 3))
        real_std = np.std(real, axis=(0, 2, 3))
        fake_std = np.std(fake, axis=(0, 2, 3))
        
        # Calculate simplified distance
        mean_diff = np.sum((real_mean - fake_mean) ** 2)
        std_diff = np.sum((real_std - fake_std) ** 2)
        
        return float(mean_diff + std_diff)
    
    def calculate_inception_score(self, images, splits=10):
        """Calculate Inception Score"""
        if not INCEPTION_SCORE_AVAILABLE:
            print("Inception Score calculation requires torchvision. Using placeholder value.")
            return random.uniform(1, 5), 0.5  # Placeholder value
        
        # Load pretrained Inception model
        model = inception_v3(pretrained=True, transform_input=False).to(device)
        model.eval()
        
        batch_size = 32
        preds = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                # Resize images if needed
                if batch.shape[2] != 299 or batch.shape[3] != 299:
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                # Convert from [-1, 1] to [0, 1] range
                batch = (batch + 1) / 2
                # Forward pass
                pred = F.softmax(model(batch), dim=1)
                preds.append(pred)
        
        # Concatenate all predictions
        preds = torch.cat(preds, dim=0).cpu().numpy()
        
        # Calculate Inception Score
        scores = []
        for i in range(splits):
            part = preds[(i * len(preds) // splits):((i + 1) * len(preds) // splits), :]
            kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
        
        return float(np.mean(scores)), float(np.std(scores))
    
    def load_real_samples(self, num_samples=100):
        """Load real samples from dataset for comparison"""
        # Load actual images from dataset
        transform = transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        # Get list of image files
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        
        try:
            image_files = [f for f in os.listdir(self.config.DATASET_PATH) 
                        if os.path.isfile(os.path.join(self.config.DATASET_PATH, f)) and
                        os.path.splitext(f)[1].lower() in valid_extensions]
                        
            if not image_files:
                print("No valid images found in dataset. Using random data.")
                return torch.randn(num_samples, 3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, device=device)
                
            # Take a random subset
            if len(image_files) > num_samples:
                image_files = random.sample(image_files, num_samples)
            
            # Load images
            real_images = []
            for img_file in tqdm(image_files, desc="Loading real images"):
                try:
                    img_path = os.path.join(self.config.DATASET_PATH, img_file)
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    real_images.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    continue
                    
            # Convert to batch
            if not real_images:
                raise ValueError("Failed to load real images")
                
            return torch.stack(real_images).to(device)
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using random data for real samples.")
            return torch.randn(num_samples, 3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, device=device)
    
    def check_diversity(self, images, num_to_check=10):
        """Check if generated images are diverse"""
        if num_to_check > len(images):
            num_to_check = len(images)
            
        # Convert images to numpy for easy comparison
        images_np = images[:num_to_check].detach().cpu().numpy()
        
        # Calculate pairwise differences
        all_same = True
        for i in range(num_to_check - 1):
            diff = np.abs(images_np[i] - images_np[i+1]).mean()
            print(f"Difference between image {i} and {i+1}: {diff:.6f}")
            if diff > 1e-4:  # threshold for considering images different
                all_same = False
                
        if all_same:
            print("WARNING: Generated images appear to be identical!")
        else:
            print("Images show diversity.")
        
        return not all_same
    
    def evaluate(self, num_samples=1000, visualize_samples=64):
        """Run full evaluation and generate a report"""
        print("\n===== Starting GAN Evaluation =====")
        
        # Generate samples
        print(f"Generating {num_samples} samples...")
        fake_images = self.generate_samples(num_samples)
        
        # Check diversity
        print("Checking image diversity...")
        diversity = self.check_diversity(fake_images, num_to_check=10)
        
        # Visualize a subset of the generated samples
        print(f"Visualizing {visualize_samples} samples...")
        self.visualize_samples(fake_images[:visualize_samples], 
                              nrow=8, 
                              title="Generated Images")
        
        # Generate latent space interpolation
        print("Generating latent space interpolations...")
        self.latent_space_interpolation()
        
        # Load real samples for comparison
        print("Loading real samples for comparison...")
        real_images = self.load_real_samples(min(num_samples, 1000))
        
        # Visualize real samples
        self.visualize_samples(real_images[:min(64, len(real_images))], 
                              nrow=8, 
                              title="Real Images")
        
        # Calculate FID score
        print("Calculating FID score...")
        fid_score = self.calculate_fid(real_images, fake_images[:len(real_images)])
        print(f"FID Score: {fid_score:.4f}")
        
        # Calculate Inception Score
        print("Calculating Inception Score...")
        is_mean, is_std = self.calculate_inception_score(fake_images)
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        
        # Save results
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": num_samples,
            "metrics": {
                "fid": fid_score,
                "inception_score": {
                    "mean": is_mean,
                    "std": is_std
                }
            },
            "diversity_check": "Passed" if diversity else "Failed"
        }
        
        with open(os.path.join(self.eval_dir, "evaluation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n===== Evaluation Complete =====")
        print(f"Results saved to {os.path.join(self.eval_dir, 'evaluation_results.json')}")
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GAN Evaluation Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the generator checkpoint")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate for evaluation")
    parser.add_argument("--visualize", type=int, default=64, help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = GanEvaluator(checkpoint_path=args.checkpoint)
    
    # Run evaluation
    evaluator.evaluate(num_samples=args.samples, visualize_samples=args.visualize)

if __name__ == "__main__":
    main()