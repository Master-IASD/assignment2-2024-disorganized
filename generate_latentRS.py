import torch 
import torchvision
import os
import argparse


from model import Generator, Latent_reweighting
from utils import load_model_G, load_model_w

## ---- Generate using latent RS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')



    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    G = Generator(g_output_dim = mnist_dim).to(device)
    G = load_model_G(G, 'checkpoints')
    G = torch.nn.DataParallel(G).to(device)
    G.eval()
    
    w = Latent_reweighting().to(device)
    w = load_model_w(w, 'checkpoints')
    w = torch.nn.DataParallel(w).to(device)
    w.eval()
    
    print('models loaded.')

    print('Start Generating')
    os.makedirs('samples_latentRS', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).to(device)
            a = torch.rand(args.batch_size).to(device)
            m = 3
            x = G(z[(w(z)/m >= a.unsqueeze(1)).squeeze()])
            x = x.reshape(x.shape[0], 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1
