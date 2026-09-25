import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.wavelet_transform import SeparableWaveletTransform3D

class OrthogonalityLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, tokens: dict):
        ortho_loss = 0.0
        feats = []
        
        for name, token in tokens.items():
            assert token.dim() == 3, f"expect [B, num_tokens, embed_dim],get {token.shape}"
            if token.shape[1]==0:
                continue
            feat = token.mean(dim=1) 
            feats.append(feat)
        
        if len(feats) < 2:
            return torch.tensor(0.0, device=feats[0].device if feats else torch.device('cpu'))
        
        pair_count = 0
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                f_i, f_j = feats[i], feats[j]
                assert f_i.shape[1] == f_j.shape[1]
                
                f_i_norm = F.normalize(f_i, p=2, dim=1)
                f_j_norm = F.normalize(f_j, p=2, dim=1)
                
                cos_sim = (f_i_norm * f_j_norm).sum(dim=1)
                ortho_loss += cos_sim.abs().mean() 
                pair_count += 1
                
        return ortho_loss / pair_count


class WaveletLoss(nn.Module):
    def __init__(self, in_channels, lambda_low=1.0, lambda_high=0.1, device=None):
        """
        lambda_low: weight for LLL 
        lambda_high: weight for others 7 
        """
        super().__init__()
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        
        self.wavelet_transform = SeparableWaveletTransform3D(in_channels, self.device)
        self.criterion = nn.L1Loss() 

    def forward(self, output, target):

        if output.dtype != torch.float32:
            output = output.float()
            target = target.float()
            
        # return: (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
        output_comps = self.wavelet_transform(output)
        target_comps = self.wavelet_transform(target)
        
        total_loss = 0.0
        
        for i, (r_comp, t_comp) in enumerate(zip(output_comps, target_comps)):
            comp_loss = self.criterion(r_comp, t_comp)
            
            if i == 0:  # LLL (Low-Low-Low)
                total_loss += comp_loss * self.lambda_low
            else:       # High Frequency Components
                total_loss += comp_loss * self.lambda_high
                
        return total_loss

class ModalityLoss(nn.Module):
    def __init__(self, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
        ones_3 = torch.ones(3, dtype=torch.float32, device=device)
        
        k_x = torch.einsum('i,jk->ijk', ones_3, k).unsqueeze(0).unsqueeze(0)
        k_y = torch.einsum('i,jk->ijk', ones_3, k.t()).unsqueeze(0).unsqueeze(0)
        z_grad = torch.tensor([-1, 0, 1], dtype=torch.float32, device=device)
        z_weight = torch.tensor([1, 2, 1], dtype=torch.float32, device=device)
        k_z = torch.einsum('i,j,k->ijk', z_grad, z_weight, z_weight).unsqueeze(0).unsqueeze(0)

        self.register_buffer('k_x', k_x)
        self.register_buffer('k_y', k_y)
        self.register_buffer('k_z', k_z)
        
    def _gradient_loss(self, output, target):
        B, C, D, H, W = target.shape
        assert C == 1
        
        gx_x = F.conv3d(output, self.k_x, padding=1)
        gy_x = F.conv3d(output, self.k_y, padding=1)
        gz_x = F.conv3d(output, self.k_z, padding=1)
        gx_y = F.conv3d(target, self.k_x, padding=1)
        gy_y = F.conv3d(target, self.k_y, padding=1)
        gz_y = F.conv3d(target, self.k_z, padding=1)
        
        mag_x = torch.sqrt(gx_x**2 + gy_x**2 + gz_x**2 + 1e-8)
        mag_y = torch.sqrt(gx_y**2 + gy_y**2 + gz_y**2 + 1e-8)
        return F.l1_loss(mag_x, mag_y)
    
    def _pixel_loss(self, output, tagret):
        return F.mse_loss(output, tagret)
    
    def forward(self, output, target):
        return self._gradient_loss(output, target) + self._pixel_loss(output, target)

# ==========================================
# 4. Main Loss Wrapper
# ==========================================
class ModalityDifferenceLoss(nn.Module):
    def __init__(self, in_channels, alpha=1.0, beta=1.0, gamma=0.1, 
                lambda_low=1.0, lambda_high=1.0, device=None, *args, **kwargs):
        """
        in_channels: input images channels
        alpha:  WaveletLoss weight
        beta:   ResidualDecouplingLoss weight
        gamma:  OrthogonalityLoss weight
        """
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.wd = WaveletLoss(in_channels, device=self.device, lambda_high=lambda_high, lambda_low=lambda_low)
        self.md = ModalityLoss(device=self.device)
        self.od = OrthogonalityLoss()
        

    def forward(self, batch, output, output_tasks, *args, **kwargs):
        num_tasks = len(output_tasks)

        wd_loss, md_loss, od_loss = 0.0, 0.0, 0.0
        for task in output_tasks:
            wd_loss += self.wd(output["reconstructed_patches"][task], batch[task])
            md_loss += self.md(output["reconstructed_patches"][task], batch[task])
        od_loss += self.od(output["tokens"])
        wd_loss /= num_tasks
        md_loss/= num_tasks

        total_loss = self.alpha * wd_loss + self.beta * md_loss + self.gamma * od_loss

        loss_value = {
            "wd_loss": wd_loss,
            "md_loss": md_loss,
            "od_loss": od_loss
        }
        return total_loss, loss_value