import os
import torch
import torchvision
import argparse
from torchvision import datasets, transforms
from precision_recall import IPR
from pytorch_fid.fid_score import calculate_fid_given_paths
from model import Generator, Latent_reweighting
from utils import load_model_G, load_model_w

real_images_path = "samples_latentRSGA/real_images"
fake_images_path = "samples_latentRSGA/fake_images"

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

def generate_fake_images(generator, latent_model, output_path, device, total_samples=10000, batch_size=2048):
    os.makedirs(output_path, exist_ok=True)
    n_samples = 0
    n_step = 5
    step_size = 5e-2
    m = 3

    with torch.no_grad():
        while n_samples < total_samples:
            # Generate initial random samples and apply rejection sampling
            z = torch.randn(batch_size, 100).to(device)
            a = torch.rand(batch_size).to(device)
            z = z[(latent_model(z) / m >= a.unsqueeze(1)).squeeze()].detach().requires_grad_()

            # Latent gradient ascent
            for _ in range(n_step):
                pseudo_loss = torch.sum(latent_model(z))
                grad_z = torch.autograd.grad(pseudo_loss, z)[0]
                grad_z = grad_z - torch.sum(z * grad_z, dim=1, keepdim=True) / 10 * z
                z = z + step_size * grad_z

            # Generate fake images
            fake_images = generator(z).view(z.size(0), 1, 28, 28)
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

    print("Loading models...")
    mnist_dim = 784
    generator = Generator(g_output_dim=mnist_dim).to(device)
    generator = load_model_G(generator, 'checkpoints')
    generator = torch.nn.DataParallel(generator).to(device)
    generator.eval()

    latent_model = Latent_reweighting().to(device)
    latent_model = load_model_w(latent_model, 'checkpoints')
    latent_model = torch.nn.DataParallel(latent_model).to(device)
    latent_model.eval()
    print("Models loaded.")

    # Data transformations for real images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    print("Loading MNIST test dataset for real images...")
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Generating real images for evaluation...")
    generate_real_images(test_loader, real_images_path, device)
    print("Generating fake images for evaluation using RSGA...")
    generate_fake_images(generator, latent_model, fake_images_path, device, total_samples=10000, batch_size=args.batch_size)

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
