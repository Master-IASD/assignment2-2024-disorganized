import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model import Generator, Discriminator, Latent_reweighting
from utils import Dw_train, w_train, save_models_Dw_w, load_model_G, load_model_D




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print('Device :', device)
    
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    
    G = Generator(g_output_dim = mnist_dim).to(device)
    G = load_model_G(G, 'checkpoints')
    G = torch.nn.DataParallel(G).to(device)
    
    Dw = Discriminator(mnist_dim).to(device)
    Dw = load_model_D(Dw, 'checkpoints')
    Dw = torch.nn.DataParallel(Dw).to(device)
    
    w = torch.nn.DataParallel(Latent_reweighting()).to(device)

    print('Model loaded.')


    # define optimizers
    Dw_optimizer = optim.Adam(Dw.parameters(), lr = 4e-4, betas=(0.5,0.5))
    w_optimizer = optim.Adam(w.parameters(), lr = 1e-4, betas=(0.5,0.5))

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            Dw_train(x, w, G, Dw, Dw_optimizer, device)
            w_train(x, w, G, Dw, w_optimizer, device)

        if epoch % 10 == 0:
            save_models_Dw_w(Dw, w, 'checkpoints')
    
    save_models_Dw_w(Dw, w, 'checkpoints')
    print()      
    print('Training done')

        
