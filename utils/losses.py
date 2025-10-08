"""
Loss functions for deraining models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant)."""
    
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)


class EdgeLoss(nn.Module):
    """Edge-aware loss using Sobel filters."""
    
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def get_edges(self, x):
        """Extract edges using Sobel filters."""
        # Convert to grayscale if RGB
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        
        return edge
    
    def forward(self, pred, target):
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        
        loss = F.l1_loss(pred_edges, target_edges)
        return loss


class SSIMLoss(nn.Module):
    """SSIM Loss."""
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def ssim(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, 
                            groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2,
                            groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2,
                          groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, pred, target):
        return 1 - self.ssim(pred, target)


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, layers=[3, 8, 15, 22]):
        super(PerceptualLoss, self).__init__()
        
        try:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True).features
            
            self.layers = layers
            self.model = nn.ModuleList()
            
            prev_layer = 0
            for layer in layers:
                self.model.append(vgg[prev_layer:layer+1])
                prev_layer = layer + 1
            
            # Freeze parameters
            for param in self.parameters():
                param.requires_grad = False
            
            self.model.eval()
            
        except Exception as e:
            print(f"Warning: Could not load VGG for perceptual loss: {e}")
            self.model = None
    
    def forward(self, pred, target):
        if self.model is None:
            return torch.tensor(0.0, device=pred.device)
        
        loss = 0
        x = pred
        y = target
        
        for layer in self.model:
            x = layer(x)
            y = layer(y)
            loss += F.l1_loss(x, y)
        
        return loss / len(self.model)


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components.
    """
    
    def __init__(self, l1_weight=1.0, charbonnier_weight=0.0, 
                 edge_weight=0.05, ssim_weight=0.1, perceptual_weight=0.0):
        super(CombinedLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.charbonnier_weight = charbonnier_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1_loss = nn.L1Loss()
        self.charbonnier_loss = CharbonnierLoss()
        self.edge_loss = EdgeLoss()
        self.ssim_loss = SSIMLoss()
        
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
    
    def forward(self, pred, target):
        losses = {}
        total_loss = 0
        
        if self.l1_weight > 0:
            l1 = self.l1_loss(pred, target)
            losses['l1'] = l1.item()
            total_loss += self.l1_weight * l1
        
        if self.charbonnier_weight > 0:
            charbonnier = self.charbonnier_loss(pred, target)
            losses['charbonnier'] = charbonnier.item()
            total_loss += self.charbonnier_weight * charbonnier
        
        if self.edge_weight > 0:
            edge = self.edge_loss(pred, target)
            losses['edge'] = edge.item()
            total_loss += self.edge_weight * edge
        
        if self.ssim_weight > 0:
            ssim = self.ssim_loss(pred, target)
            losses['ssim'] = ssim.item()
            total_loss += self.ssim_weight * ssim
        
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(pred, target)
            losses['perceptual'] = perceptual.item()
            total_loss += self.perceptual_weight * perceptual
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


if __name__ == "__main__":
    # Test losses
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pred = torch.rand(2, 3, 256, 256).to(device)
    target = torch.rand(2, 3, 256, 256).to(device)
    
    print("Testing loss functions:")
    
    # Test individual losses
    l1 = nn.L1Loss()(pred, target)
    print(f"L1 Loss: {l1.item():.4f}")
    
    charbonnier = CharbonnierLoss()(pred, target)
    print(f"Charbonnier Loss: {charbonnier.item():.4f}")
    
    edge = EdgeLoss()(pred, target)
    print(f"Edge Loss: {edge.item():.4f}")
    
    ssim = SSIMLoss()(pred, target)
    print(f"SSIM Loss: {ssim.item():.4f}")
    
    # Test combined loss
    combined = CombinedLoss(l1_weight=1.0, edge_weight=0.05, ssim_weight=0.1)
    total, losses = combined(pred, target)
    print(f"\nCombined Loss: {total.item():.4f}")
    print(f"Components: {losses}")
