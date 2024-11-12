import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


from model import Generator, Discriminator
from utils import DWass_train, GWass_train, save_models_G_D, load_model_G, load_model_D




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="Number of critic iterations per generator iteration")
    parser.add_argument("--no_train_from_checkpoint", action='store_false', default=True, dest='train_from_checkpoint')

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print('Training from checkpoint :', args.train_from_checkpoint)

    print('Device :', device)
    
    os.makedirs('checkpoints', exist_ok=True)
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
    
    if args.train_from_checkpoint :
        print('Loading from checkpoint...')
        G = Generator(g_output_dim = mnist_dim).to(device)
        G = load_model_G(G, 'checkpoints_wass')
        G = torch.nn.DataParallel(G).to(device)
    
        D = Discriminator(mnist_dim).to(device)
        D = load_model_D(D, 'checkpoints_wass')
        D = torch.nn.DataParallel(D).to(device)
    else : 
        G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).to(device)
        D = torch.nn.DataParallel(Discriminator(mnist_dim)).to(device)


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 

    # define loss
    criterion = nn.BCELoss() 

    # define optimizers (original WGAN paper uses RMSprop)
    G_optimizer = optim.RMSprop(G.parameters(), lr = args.lr)
    D_optimizer = optim.RMSprop(D.parameters(), lr = args.lr)

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            for _ in range(args.n_critic):
                DWass_train(x, G, D, D_optimizer, device)
            GWass_train(x, G, D, G_optimizer, device)

        if epoch % 10 == 0:
            save_models_G_D(G, D, 'checkpoints_wass')
    
    save_models_G_D(G, D, 'checkpoints_wass')        
    print()    
    print('Training done')

        
