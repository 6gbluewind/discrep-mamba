# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.vision_transformer import  Block

class VitEncoder(nn.Module):
    def __init__(self, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks =  nn.ModuleList([
            Block(embed_dim, num_heads,  mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

def vit_encoder_base(embed_dim=768, depth=12, num_heads=12):
    model = VitEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4,norm_layer=nn.LayerNorm)
    print('use vit_base encoder')
    return model

def vit_encoder_large(embed_dim=1024, depth=24, num_heads=16):
    model = VitEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4,norm_layer=nn.LayerNorm)
    print('use vit_large encoder')
    return model


if __name__=='__main__':
    data = torch.randn(2, 990, 768)
    model = vit_encoder_base().to('cuda:0')
    opt = model(data)
    print(opt.shape)