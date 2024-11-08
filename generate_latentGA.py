import torch 
import torchvision
import os
import argparse

## ----- Generate from latent GA


from model import Generator, Latent_reweighting
from utils import load_model_G, load_model_w


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
    os.makedirs('samples_latentGA', exist_ok=True)
    
    # hyperparameter for latent gradient ascent :
    n_step = 10
    step_size = 5e-2

    n_samples = 0

    while n_samples<10000:
        z = torch.randn(args.batch_size, 100, requires_grad=True).to(device)

        for i_step in range(n_step):
            pseudo_loss = torch.sum(w(z))
            grad_z = torch.autograd.grad(pseudo_loss,z)[0]

            # Projection step for high-dimensional gaussians
            grad_z = grad_z - torch.sum(torch.mul(z,grad_z),dim=1).unsqueeze(1)/10*z # 10 = sqrt(d) dimension of latent space
            z = z + step_size*grad_z
            
        with torch.no_grad():
            x = G(z)
            x = x.reshape(x.shape[0], 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples_latentGA', f'{n_samples}.png'))         
                    n_samples += 1
