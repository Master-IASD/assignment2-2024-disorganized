import os
import torch
import torchvision
import argparse
from torchvision import datasets, transforms
from precision_recall import IPR
from pytorch_fid.fid_score import calculate_fid_given_paths
from model import Generator
from utils import load_model_G

# Define paths for real and fake images
real_images_path = "samples/real_images"
fake_images_path = "samples/fake_images"

def clear_directory(path):
    """Delete all files in the given directory."""
    if os.path.exists(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(path)

def generate_real_images(data_loader, output_path, device):
    os.makedirs(output_path, exist_ok=True)
    for batch_idx, (images, _) in enumerate(data_loader):
        images = images.to(device)
        for i in range(images.size(0)):
            image_path = os.path.join(output_path, f"real_image_{batch_idx * data_loader.batch_size + i}.png")
            torchvision.utils.save_image(images[i], image_path)

def generate_fake_images(generator, output_path, device, total_samples=10000, batch_size=2048):
    os.makedirs(output_path, exist_ok=True)
    n_samples = 0
    with torch.no_grad():
        while n_samples < total_samples:
            z = torch.randn(batch_size, 100).to(device)
            fake_images = generator(z).view(batch_size, 1, 28, 28)
            for i in range(fake_images.size(0)):
                if n_samples < total_samples:
                    image_path = os.path.join(output_path, f"fake_image_{n_samples}.png")
                    torchvision.utils.save_image(fake_images[i], image_path)
                    n_samples += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Precision, Recall, and FID Scores.')
    parser.add_argument("--batch_size", type=int, default=2048, help="The batch size for image generation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Clearing existing files in real and fake image directories...")
    clear_directory(real_images_path)
    clear_directory(fake_images_path)

    # Load generator model
    print("Loading generator model...")
    mnist_dim = 784
    generator = Generator(g_output_dim=mnist_dim).to(device)
    generator = load_model_G(generator, 'checkpoints')
    generator = torch.nn.DataParallel(generator).to(device)
    generator.eval()
    print("Model loaded.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    print("Loading MNIST test dataset for real images...")
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Generating real images for evaluation...")
    generate_real_images(test_loader, real_images_path, device)
    print("Generating fake images for evaluation...")
    generate_fake_images(generator, fake_images_path, device, total_samples=10000, batch_size=args.batch_size)

    print("Calculating Precision and Recall...")
    ipr = IPR(device=device, batch_size=128, k=1)
    ipr.compute_manifold_ref(real_images_path)
    precision, recall = ipr.precision_and_recall(fake_images_path)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("Calculating FID...")
    fid_value = calculate_fid_given_paths(
        [real_images_path, fake_images_path],
        batch_size=64,
        device=device,
        dims=2048
    )
    print(f"FID Score: {fid_value:.4f}")

    print("Cleaning up real images to conserve space...")
    clear_directory(real_images_path)
    print("Evaluation complete.")
