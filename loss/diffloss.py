import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import SSIMLoss
from loss.wavelet_transform import SeparableWaveletTransform3D

# ==========================================
# 1. Orthogonality Loss 
# ==========================================
class OrthogonalityLoss(nn.Module):
    def __init__(self, flatten=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fllaten = flatten

    def forward(self, tokens):
        ortho_loss = 0.0
        feats = []
        for _, token in tokens.items():
            if self.fllaten:
                token =  token.flatten(1)
            elif token.dim() > 2:
                if token.shape[1] == 0:
                    continue
                if token.dim() == 5: 
                    token = F.adaptive_avg_pool3d(token, 1).squeeze(-1).squeeze(-1).squeeze(-1)
                elif token.dim() == 4: 
                    token = F.adaptive_avg_pool2d(token, 1).squeeze(-1).squeeze(-1)
                elif token.dim() == 3: 
                    token = token.mean(dim=1)
            feats.append(token)
        
        if len(feats) < 2:
            return torch.tensor(0.0, device=feats[0].device if feats else torch.device('cpu'))
        
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                f_i, f_j = feats[i], feats[j]
                assert f_i.shape[1] == f_j.shape[1], f"{f_i.shape}, {f_j.shape}"
                f_i_norm = F.normalize(f_i, p=2, dim=1)
                f_j_norm = F.normalize(f_j, p=2, dim=1)
                ortho_loss += (f_i_norm * f_j_norm).sum(dim=1).abs().mean()
        return ortho_loss


# ==========================================
# 2. Wavelet Loss 
# ==========================================
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

    def forward(self, rec, target):

        if rec.dtype != torch.float32:
            rec = rec.float()
            target = target.float()
            
        # return: (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
        rec_comps = self.wavelet_transform(rec)
        target_comps = self.wavelet_transform(target)
        
        total_loss = 0.0
        
        for i, (r_comp, t_comp) in enumerate(zip(rec_comps, target_comps)):
            comp_loss = self.criterion(r_comp, t_comp)
            
            if i == 0:  # LLL (Low-Low-Low)
                total_loss += comp_loss * self.lambda_low
            else:       # High Frequency Components
                total_loss += comp_loss * self.lambda_high
                
        return total_loss


# ==========================================
# 3. Residual Decoupling Loss 
# ==========================================
class ResidualDecouplingLoss(nn.Module):
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
        
    def _gradient_loss(self, x, y):
        B, C, D, H, W = x.shape
        x_ = x.view(-1, 1, D, H, W)
        y_ = y.view(-1, 1, D, H, W)
        
        gx_x = F.conv3d(x_, self.k_x, padding=1)
        gy_x = F.conv3d(x_, self.k_y, padding=1)
        gz_x = F.conv3d(x_, self.k_z, padding=1)
        gx_y = F.conv3d(y_, self.k_x, padding=1)
        gy_y = F.conv3d(y_, self.k_y, padding=1)
        gz_y = F.conv3d(y_, self.k_z, padding=1)
        
        mag_x = torch.sqrt(gx_x**2 + gy_x**2 + gz_x**2 + 1e-8)
        mag_y = torch.sqrt(gx_y**2 + gy_y**2 + gz_y**2 + 1e-8)
        return F.l1_loss(mag_x, mag_y)

    def forward(self, targets, recons):
        res_loss = 0.0 
        pairs = [('t1', 't1ce'), ('t1n', 't1c')]
        for k1, k2 in pairs:
            if k1 in recons and k2 in recons and k1 in targets and k2 in targets:
                real_diff = targets[k2] - targets[k1]
                pred_diff = recons[k2] - recons[k1]
                res_loss += F.mse_loss(pred_diff, real_diff)
                res_loss += 0.2 * self._gradient_loss(pred_diff, real_diff)
                break 
        return res_loss

# ==========================================
# 4. Main Loss Wrapper
# ==========================================
class ModalityDifferenceLoss(nn.Module):
    def __init__(self, in_channels, alpha=1.0, beta = 0.0, gamma=0.0, 
                 lambda_low=1.0, lambda_high=1.0, device=None, flatten=False, *args, **kwargs):
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

        self.orthLoss = OrthogonalityLoss(flatten=flatten)
        self.resLoss = ResidualDecouplingLoss(device)
        
        self.waveLoss = WaveletLoss(
            in_channels=in_channels,
            lambda_low=lambda_low,
            lambda_high=lambda_high,
            device=device
        )
            
        self.mseLoss = nn.MSELoss(reduction="mean")
        self.to(device)

    def forward(self, batch, output, output_tasks, *args, **kwargs):
        num_tasks = len(output_tasks)

        mse_loss = 0.0
        wavelet_loss = 0.0
        res_loss = 0.0
        orth_loss = 0.0

        for task in output_tasks:
            mse_loss += self.mseLoss(batch[task], output["reconstructed_patches"][task])
            wavelet_loss += self.waveLoss(batch[task], output["reconstructed_patches"][task])
        if self.beta != 0:
            res_loss += self.resLoss(batch, output["reconstructed_patches"])
        else:
            res_loss = 0.0
        

        if self.gamma != 0:
            orth_loss += self.orthLoss(output['tokens'])
        else:
            orth_loss = 0.0
            
        mse_loss /= num_tasks
        wavelet_loss /= num_tasks
       
        total_loss = mse_loss + self.alpha * wavelet_loss + self.beta * res_loss + self.gamma * orth_loss
        loss_value = {
            "mse_loss": mse_loss,
            "wavelet _loss": wavelet_loss,
            "res_loss": res_loss,
            "orth_loss": orth_loss
        }
        return total_loss, loss_value

