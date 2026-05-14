## ULS_MSA Enhanced: Ultra-Lightweight Subspace Multi-Scale Attention Network
## Enhanced with LMNet components to solve RC=1.0000 problem

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, laplace

def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    """Create Gaussian kernel for edge enhancement (from LMNet)"""
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue 
    g = gaussian_filter(n, sigma)
    gaussiankernel = torch.from_numpy(g)
    return gaussiankernel.float()

def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    """Create Laplacian kernel for edge enhancement (from LMNet)"""
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:,:,int((kernelsize-1)/2),int((kernelsize-1)/2)] = kernelvalue
    l = laplace(n)
    laplacekernel = torch.from_numpy(l)
    return laplacekernel.float()

class SEM(nn.Module):
    """Squeeze-and-Excitation Module from LMNet - Better attention than SimpleAttention"""
    def __init__(self, ch_out, reduction=16):
        super(SEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out//reduction, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(ch_out//reduction, ch_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y.expand_as(x)

class EEM(nn.Module):
    """Edge Enhancement Module from LMNet - Helps with boundary detection"""
    def __init__(self, ch_in, ch_out, kernel=3, groups=1, reduction=8):
        super(EEM, self).__init__()
        
        self.groups = groups if groups <= ch_in else ch_in
        self.gk = gaussiankernel(ch_in, max(1, ch_in//self.groups), kernel, kernel-2, 0.9)
        self.lk = laplaceiankernel(ch_in, max(1, ch_in//self.groups), kernel, 0.9)
        self.gk = nn.Parameter(self.gk, requires_grad=False)
        self.lk = nn.Parameter(self.lk, requires_grad=False)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//2, kernel_size=1, padding=0, bias=False),
            nn.PReLU(num_parameters=ch_out//2, init=0.05),
            nn.GroupNorm(max(1, (ch_out//2)//4), ch_out//2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//2, kernel_size=1, padding=0, bias=False),
            nn.PReLU(num_parameters=ch_out//2, init=0.05),
            nn.GroupNorm(max(1, (ch_out//2)//4), ch_out//2)
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch_out//2, ch_out, kernel_size=1, padding=0, bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.01),
            nn.GroupNorm(max(1, ch_out//4), ch_out)
        )
        
        self.sem1 = SEM(ch_out, reduction=reduction)
        self.sem2 = SEM(ch_in, reduction=reduction)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.03)
      
    def forward(self, x):
        device = x.device
        groups = min(self.groups, x.size(1))
        
        DoG = F.conv2d(x, self.gk.to(device), padding='same', groups=groups)
        LoG = F.conv2d(DoG, self.lk.to(device), padding='same', groups=groups)
        DoG = self.conv1(DoG - x)
        LoG = self.conv2(LoG)
        tot = self.conv3(DoG * LoG)
        
        tot1 = self.sem1(tot)
        x1 = self.sem2(x)
        
        return self.prelu(x + x1 + tot + tot1)

class EnhancedDepthwiseConv(nn.Module):
    """Enhanced Depthwise Separable Convolution with LMNet improvements"""
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1):
        super(EnhancedDepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=ch_in, bias=False)
        self.pointwise = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
        # Use GroupNorm instead of BatchNorm for better stability
        self.gn = nn.GroupNorm(max(1, ch_out//4), ch_out)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.05)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.gn(x)
        x = self.prelu(x)
        return x

class EnhancedBlock(nn.Module):
    """Enhanced building block with LMNet components"""
    def __init__(self, ch_in, ch_out):
        super(EnhancedBlock, self).__init__()
        self.conv1 = EnhancedDepthwiseConv(ch_in, ch_out)
        self.conv2 = EnhancedDepthwiseConv(ch_out, ch_out)
        self.eem = EEM(ch_out, ch_out, kernel=3, groups=ch_out, reduction=8)
        self.sem = SEM(ch_out, reduction=8)
        
        # Residual connection if dimensions match
        self.residual = ch_in == ch_out
        if not self.residual:
            self.proj = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
            
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.eem(out)
        out = self.sem(out)
        
        if self.residual:
            out = out + identity
        else:
            out = out + self.proj(identity)
            
        return out

class ULS_MSA_v2(nn.Module):
    """Enhanced ULS_MSA with LMNet components to solve RC=1.0000"""
    def __init__(self, img_ch=3, output_ch=1):
        super(ULS_MSA_v2, self).__init__()

        # Encoder pathway with enhanced blocks
        self.input_conv = nn.Conv2d(img_ch, 16, kernel_size=3, padding=1, bias=False)
        self.input_norm = nn.GroupNorm(4, 16)
        self.input_act = nn.PReLU(num_parameters=16, init=0.05)
        
        # Enhanced encoder blocks
        self.enc1 = EnhancedBlock(16, 32)
        self.enc2 = EnhancedBlock(32, 64)
        self.enc3 = EnhancedBlock(64, 128)
        
        # Bottleneck with edge enhancement
        self.bottleneck = nn.Sequential(
            EnhancedDepthwiseConv(128, 128),
            EEM(128, 128, kernel=3, groups=128, reduction=16),
            SEM(128, reduction=16)
        )
        
        # Decoder pathway
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = EnhancedBlock(192, 64)  # 64 + 128 from skip connection = 192
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = EnhancedBlock(96, 32)   # 32 + 64 from skip connection = 96
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = EnhancedBlock(48, 16)   # 16 + 32 from skip connection = 48
        
        # Final output layers with edge-aware processing
        self.final_eem = EEM(16, 16, kernel=3, groups=16, reduction=4)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, 8),
            nn.PReLU(num_parameters=8, init=0.01),
            nn.Conv2d(8, output_ch, kernel_size=1, bias=False)
        )
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # Add dropout for regularization (helps prevent overprediction)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        # Input processing
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = self.input_act(x)
        
        # Encoder pathway with skip connections
        e1 = self.enc1(x)
        p1 = self.maxpool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.maxpool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.maxpool(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        b = self.dropout(b)  # Add dropout to prevent overprediction
        
        # Decoder pathway
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Final processing with edge enhancement
        d1 = self.final_eem(d1)
        output = self.final_conv(d1)
        
        return output

# For backward compatibility
ULS_MSA = ULS_MSA_v2
