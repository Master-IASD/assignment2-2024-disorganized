import torch
import os



def D_train(x, G, D, D_optimizer, criterion, device):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on fake
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion, device):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()

def Dw_train(x, w, G, D, Dw_optimizer, device):
    #=====================Train the discriminator=====================#
    D.zero_grad()

    # sample real data :
    x_real = x.to(device)
    
    # sample latent vector and generate images :
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake = G(z)
    
    EMD = torch.mean(D(x_real)-torch.mul(w(z),D(G(z))))
    GP = gradient_penalty(x_real, x_fake, D, device)
    
    loss = -EMD+GP
    loss.backward()
    Dw_optimizer.step()
    
    return loss.data.item()
    
    

def w_train(x, w, G, D, w_optimizer, device, m=3, lambda1=10, lambda2=3):
    #==============Train the latent reweighting function==============#
    w.zero_grad()
    
    # sample latent vector and generate images :
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake = G(z)
    
    Delta = torch.min(D(x_fake))
    EMD = torch.mean(torch.mul(w(z),D(G(z))-Delta))
    Rnorm = torch.square(torch.mean(w(z))-1)
    Rclip = torch.mean(torch.maximum(torch.zeros(w(z).shape),w(z)-m))
    
    loss = EMD + lambda1*Rnorm + lambda2*Rclip
    loss.backward()
    w_optimizer.step()
    
    return loss.data.item()
    
    


def save_models_G_D(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))

def save_models_Dw_w(Dw, w, folder):
    torch.save(Dw.state_dict(), os.path.join(folder,'Dw.pth'))
    torch.save(w.state_dict(), os.path.join(folder,'w.pth'))


def load_model_G(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def load_model_D(D, folder):
    ckpt = torch.load(os.path.join(folder,'D.pth'))
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return D

def load_model_w(w, folder):
    ckpt = torch.load(os.path.join(folder,'w.pth'))
    w.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return w

def gradient_penalty(x_real, x_fake, D, device):
        batch_size = x_real.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1).to(device)# attention ici modif un peu à l'aveugle
        eps = eps.expand_as(x_real)
        
        # Interpolation between real data and fake data.
        interpolation = eps * x_real + (1 - eps) * x_fake
        
        # get logits for interpolated images
        interp_logits = D(interpolation).to(device)
        grad_outputs = torch.ones_like(interp_logits).to(device)
        
        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)
