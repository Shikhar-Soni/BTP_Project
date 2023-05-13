import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

def calculate_fid(real_imgs, fake_imgs, net, device, eps=1e-6):
    """
    Calculates the Fréchet Inception Distance (FID) between real and generated images.
    """
    net.eval()
    with torch.no_grad():
        # Get Inception activations for real images
        real_activations = net(real_imgs.to(device))[0].view(real_imgs.shape[0], -1)

        # Get Inception activations for generated images
        fake_activations = net(fake_imgs.to(device))[0].view(fake_imgs.shape[0], -1)

        # Calculate mean and covariance for real and generated activations
        mu1, sigma1 = torch.mean(real_activations, dim=0), torch_cov(real_activations, rowvar=False)
        mu2, sigma2 = torch.mean(fake_activations, dim=0), torch_cov(fake_activations, rowvar=False)

        # Calculate FID score
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1.cpu().numpy() @ sigma2.cpu().numpy(), disp=False)
        covmean = torch.tensor(covmean, device=device)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid_score = diff.dot(diff) + torch.trace(sigma1 + sigma2 - 2*covmean)
    
    return fid_score


def torch_cov(m, rowvar=False, eps=1e-6):
    """
    Estimate a covariance matrix given data.
    """
    if rowvar:
        m = m.t()
    m = m.type(torch.float32)
    fact = 1.0 / (m.shape[1] - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).unsqueeze(0)

class MSEFIDLoss(torch.nn.Module):
    def __init__(self, device, lambda_fid=-0.25):
        super().__init__()
        self.lambda_fid = lambda_fid
        self.device = device
        self.net = inception_v3(pretrained=True, transform_input=False).to(device)
        self.net.eval()

    def forward(self, real, generated):
        # Calculate primary loss (MSE)
        mse_loss = F.mse_loss(real, generated)

        # Calculate FID regularization term
        fid_loss = calculate_fid(real, generated, self.net, self.device)

        # Combine losses
        loss = mse_loss + self.lambda_fid * fid_loss

        return loss
