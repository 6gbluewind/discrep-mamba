import math
import logging


import torch
import torch.nn as nn

from einops import rearrange
from functools import partial

from timm.layers import trunc_normal_
from timm.layers.helpers import to_3tuple
from timm.models.vision_transformer import Block
from models.vim_encoder import vim_encoder_small
from typing import List, Tuple, Dict, Optional, Literal
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetrPrUpBlock

from models.multimae3d_utils import (
    calc_patchified_dim,
    patchify,
    unpatchify,
    shuffle_patches,
    unshuffle_patches,
    get_batch_pos_embed,
    build_position_embedding,
    CrossAttention,
)


logger = logging.getLogger(__name__)

# =============================================================================
# Input and output adapters
class PatchedInputAdapter(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim=None, depth=0,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.patch_size = to_3tuple(patch_size)
        self.embed_dim = embed_dim
        self.depth = depth
        if depth > 0:
            print(f"inputater depth {depth}")
            self.vim_encoder = vim_encoder_small(depth=depth)
        else:
            self.vim_encoder = None

        if self.embed_dim is not None:
            self.init(embed_dim=embed_dim)

    def init(self, embed_dim, *args, **kwarks):

        self.embed_dim = embed_dim
        self.proj = torch.nn.Conv3d(
            self.in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, "b t c x y z -> (b t) c x y z")
        x = self.proj(x)
        x = rearrange(x, "(b t) c x y z -> b t c x y z", b=b)
        x = x.squeeze()
        
        if  self.depth > 0 and (x.dim() != 2) and (x.dim() == 3 and x.shape[1]!=0):  
            x = self.vim_encoder(x)
        
        if isinstance(x, List):
            return x[-1]

        return x


class SpatialOutputAdapter(nn.Module):
    def __init__(self, out_channels=1, img_size=(160,176,144), patch_size=(16, 16, 16), enc_embed_dim=None,
        embed_dim=768, num_heads=12, depth=4,
        mlp_ratio=4.0, qkv_bias=True, drop_path_rate= 0.0, attn_drop_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        cls_token=False, has_learnable_embed=False, pos_embed_type="sincos",
        reconstruct_rate=1.0, reconstruct_all_testing=True,
        *args, **kwargs,
    ):
        super().__init__()

        # =====================================================================
        # store additional parameters here:
        self.out_channels = out_channels
        self.img_size = to_3tuple(img_size)
        self.patch_size = to_3tuple(patch_size)
        self.enc_embed_dim = enc_embed_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias 
        self.drop_path_rate = drop_path_rate  
        self.attn_drop_rate = attn_drop_rate  
        self.norm_layer = norm_layer 
        self.pos_embed_type = pos_embed_type
        self.reconstruct_rate = reconstruct_rate
        self.reconstruct_all_testing = reconstruct_all_testing
        
        self.cls_token = cls_token
        self.has_learnable_embed = has_learnable_embed
        self.patchified_dim = calc_patchified_dim(self.img_size, self.patch_size)
        
        self.num_patches = (self.patchified_dim[0] * self.patchified_dim[1] * self.patchified_dim[2])
        self.visualizable = True

        if self.cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        else:
            self.cls_token = None

        if self.has_learnable_embed:
            self.learnable_embed = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Init mask tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02) 
        
        self.init_transformer()

       
        self.dim_patch = (self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * self.out_channels)
        self.out_proj = nn.Linear(embed_dim, self.dim_patch)

        if self.enc_embed_dim is not None:
            self.init(enc_embed_dim=enc_embed_dim)

    def init_pos_embed(self):
        self.pos_embed, self.global_grid_size = build_position_embedding(
            num_patches=self.num_patches,
            embed_dim=self.embed_dim,
            pos_embed_type=self.pos_embed_type,
            patchified_dim=self.patchified_dim,
            patch_size=self.patch_size,
            global_img_size=self.img_size
        )

    def init_transformer(self):
        self.xattn = CrossAttention(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop_rate,
        )
        self.context_norm = self.norm_layer(self.embed_dim)
        self.query_norm = self.norm_layer(self.embed_dim)
        self.out_norm = self.norm_layer(self.embed_dim)

        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, self.embed_dim),
        )

        if self.depth > 0:
            dpr = [ x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
            self.blocks = nn.Sequential(
                *[
                    Block(
                        dim=self.embed_dim,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=self.qkv_bias,
                        attn_drop=self.attn_drop_rate,
                        drop_path=dpr[i],
                        act_layer=nn.GELU,
                        norm_layer=self.norm_layer, 
                    )
                    for i in range(self.depth)
                ]
            )
        else:
            self.blocks = nn.Identity()

    def init(self, enc_embed_dim, *args, **kwarks):
        self.init_pos_embed()
        self.enc_embed_dim = enc_embed_dim
        self.proj_context = nn.Linear(enc_embed_dim, self.embed_dim)

    @staticmethod
    def apply_pos_embed_to_query(
        query, batch_pos_embed, perm_idx: Optional[torch.Tensor]
    ):
        batch_size = query.shape[0]
        if perm_idx is None:
            query += batch_pos_embed
        else:
            query += batch_pos_embed[torch.arange(batch_size)[:, None], perm_idx, ...]
        return query

    @staticmethod
    def apply_pos_embed_to_context(context, batch_pos_embed, perm_indices, task_ranges):
        batch_size = context.shape[0]
        new_context = []
        for task, (start, end) in task_ranges.items():
            task_context_num_tokens = context[:, start:end].shape[1]
            if task in perm_indices:
                task_perm_idx = perm_indices[task]
                task_batch_pos_embed = batch_pos_embed[
                    torch.arange(batch_size)[:, None], task_perm_idx, ...
                ]
            else:
                task_batch_pos_embed = batch_pos_embed
            selected_task_batch_pos_embed = task_batch_pos_embed[
                :, :task_context_num_tokens
            ]
            new_context.append(context[:, start:end] + selected_task_batch_pos_embed)
        new_context = torch.concat(new_context, dim=1)
        
        cls_dims = context.shape[1] - new_context.shape[1]
        new_context = torch.concat([context[:, :cls_dims], new_context], dim=1)
        return new_context

    def _process_tokens(self, encoder_tokens, task_range, perm_idx,
        patchified_dim=None, return_all_layers=False, batch=None, adapter_task=None,
    ):
        if isinstance(encoder_tokens, list):
            encoder_tokens = encoder_tokens[-1]

        # handle task_range
        if isinstance(task_range, tuple):
            query_task_range = task_range
            context_task_range = None
        elif isinstance(task_range, dict):
            query_task_range = task_range.get(adapter_task, None)
            context_task_range = task_range
        
        # handle perm_idx
        if isinstance(perm_idx, torch.Tensor):
            query_perm_idx = perm_idx
            context_perm_idx = None
        elif isinstance(task_range, dict):
            query_perm_idx = perm_idx.get(adapter_task, None)
            context_perm_idx = perm_idx

        num_selected_tokens = (query_task_range[1] - query_task_range[0] if query_task_range is not None else 0)
        num_patches = (self.num_patches if patchified_dim is None else patchified_dim[0] * patchified_dim[1] * patchified_dim[2])

        # Compute number of masked_tokens
        num_masked_tokens = num_patches - num_selected_tokens 
        if not self.training and self.reconstruct_all_testing:
            reconstruct_rate = 1.0
        else:
            reconstruct_rate = self.reconstruct_rate
        num_masked_tokens = int(num_masked_tokens * reconstruct_rate)

        batch_size = encoder_tokens.shape[0]

        # project encoder tokens to embedding dim of this task
        context_tokens = self.proj_context(encoder_tokens)

        masked_tokens = self.mask_token.repeat(batch_size, num_masked_tokens, 1)

        # generate the fitting positional embedding for the input size and repeat it for the entire batch. 
        batch_pos_embed, patchified_dim = get_batch_pos_embed(
            batch=batch,
            pos_embed=self.pos_embed,
            pos_embed_type=self.pos_embed_type,
            input_patchified_dim=( self.patchified_dim if patchified_dim is None else patchified_dim),
            global_grid_size=self.global_grid_size,
            patch_size=self.patch_size,
            batch_size=batch_size,
        )

        if query_task_range is not None:
            queries = context_tokens[:, query_task_range[0] : query_task_range[1]]
            queries = torch.cat([queries, masked_tokens], dim=1)
        else:
            queries = masked_tokens

        # add positional embedding to queries
        queries = self.apply_pos_embed_to_query(queries, batch_pos_embed, query_perm_idx)
        
        # if task_range is not just a tuple but a dictionary of tuples, then we apply 
        # positional embeddings to all tasks within context, otherwise we skip this step.
        if context_task_range is not None:
            context_tokens = self.apply_pos_embed_to_context(context_tokens, batch_pos_embed, context_perm_idx, context_task_range)

        # add learnable tokens for this task
        if self.cls_token:
            learnable_tokens = self.cls_token.repeat(batch_size, 1, 1)
            queries = torch.cat([learnable_tokens, queries], dim=1)

        # add learnable embedding to queries
        if self.has_learnable_embed:
            queries = queries + self.learnable_embed

        all_tokens = []
        if self.depth > 0:
            x = self.xattn(self.query_norm(queries), self.context_norm(context_tokens))
            x = x + self.mlp(self.out_norm(x))

            for block in self.blocks:
                x = block(x)
                if return_all_layers:
                    all_tokens.append(x)
            return all_tokens if return_all_layers else x
        else:
            return [queries] if return_all_layers else queries

    def forward(self, encoder_tokens, task_range, perm_idx,
        patchified_dim=None, batch=None, adapter_task=None, *args, **kwargs,
    ):
        x = self._process_tokens(
            encoder_tokens,
            task_range,
            perm_idx,
            patchified_dim,
            batch=batch,
            adapter_task=adapter_task,
        )

        # remove learnable tokens
        if self.cls_token is not None:
            learnable_tokens = x[:, :1, ...]
            x = x[:, 1:, ...]

        # project token to number of values in one patch
        x = self.out_proj(x)

        # rearrange projected tokens to patch shape
        x = rearrange(
            x,
            "b t (c x y z) -> b t c x y z",
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

        return x


# Copied from original MultiMAE repository, slightly modified 
# https://github.com/EPFL-VILAB/MultiMAE
class LinearOutputAdapter(nn.Module):
    def __init__(self, num_classes=2, enc_embed_dim=768, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1.0, num_global_tokens=1, token_aggregation: Literal["cls", "cls-mean", "mean", "middle"] = "cls", *args, **kwargs,
    ):
        super().__init__()

        # binary classification is done with 1 class, not 2 classes
        self.num_classes = 1 if num_classes == 2 else num_classes
        # self.num_classes = num_classes
        self.enc_embed_dim = enc_embed_dim
        self.norm_layer = norm_layer
        self.init_scale = init_scale  # copied from MultiMAE code
        self.num_global_tokens = num_global_tokens
        self.token_aggregation = token_aggregation
        self.visualizable = False

        if enc_embed_dim is not None:
            self.init(enc_embed_dim=enc_embed_dim)

    def init(self, enc_embed_dim: int, *args, **kwargs):
        self.enc_embed_dim = enc_embed_dim

        self.norm = self.norm_layer(enc_embed_dim)
        self.head = (
            nn.Linear(enc_embed_dim, self.num_classes)
            if self.num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)
        self.head.weight.data.mul_(self.init_scale)
        self.head.bias.data.mul_(self.init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, encoder_tokens: torch.Tensor, *args, **kwargs):
        if isinstance(encoder_tokens, list):
            encoder_tokens = encoder_tokens[-1]

            # CLS tokens at the beginning
            if self.token_aggregation == "cls":
                x = encoder_tokens[:, 0] 
            if self.token_aggregation == "cls-mean":
                x = encoder_tokens[:, : self.num_global_tokens].mean(dim=1)
            if self.token_aggregation == "mean":
                x = encoder_tokens.mean(dim=1)
            if self.token_aggregation == "middle":
                # 取序列中间位置的 token（0-indexed）
                mid_idx = (encoder_tokens.shape[1] - 1) // 2
                x = encoder_tokens[:, mid_idx]
        # x = encoder_tokens[:, 0]
        x = self.head(self.norm(x))
        return x


class UNETROutputAdapter(nn.Module):
    def __init__(self, out_channels, img_size, patch_size, embed_dim,
        unetr_in_channels=4, enc_embed_dim=None, enc_depth=None,
        unetr_use_input=True, spatial_dims = 3,
        norm_name="instance", res_block=True, conv_block=True,input_tasks=None,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.img_size = to_3tuple(img_size)
        self.patch_size = to_3tuple(patch_size)
        self.enc_embed_dim = enc_embed_dim
        self.enc_depth = enc_depth
        self.embed_dim = embed_dim
        self.unetr_use_input = unetr_use_input
        self.visualizable = True
        
        self.depth = 4
        print(f"setting depth to {self.depth}")
        print(f"unetr img_size: {self.img_size}")
        print(f"unetr patch_size: {self.patch_size}")
        print(f"unetr out_channels: {self.out_channels}")
        
        
        if (enc_embed_dim is not None) and (enc_depth is not None):
            self.init(enc_embed_dim, enc_depth, input_tasks=input_tasks)

        feature_size = self.patch_size[0] # 16 
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=unetr_in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.embed_dim,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.embed_dim,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.embed_dim,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.embed_dim,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def init(
        self,
        enc_embed_dim,
        enc_depth,
        input_tasks,
        in_channels=None,
        global_img_size=None,
        *args,
        **kwarks,
    ):
        assert (enc_depth % self.depth == 0), "Encoder Depth must be a multiple of decoder depth" 
        self.enc_embed_dim = enc_embed_dim
        self.enc_depth = enc_depth
        self.input_tasks = input_tasks

        blocks_per_decoder_adapter = int(enc_depth / self.depth)
        indices = list(range(blocks_per_decoder_adapter - 1, enc_depth, blocks_per_decoder_adapter))
        print(f"initializing UNETR - {indices=}")

        self.proj_context = nn.Linear(enc_embed_dim, self.embed_dim)
        self.in_channels = (
            len([task for task in self.input_tasks if task != "seg"])
            if in_channels is None
            else in_channels
        )

    def forward(
        self,
        encoder_tokens,
        task_range: Optional[Tuple[int, int]],
        perm_idx: Optional[torch.Tensor],
        patchified_dim: Tuple[int, int, int] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        adapter_task: str = None,
        *args,
        **kwargs,      
    ):
        
        blocks_per_decoder_adapter = int(self.enc_depth / self.depth)
        indices = list(
            range(
                blocks_per_decoder_adapter - 1,
                self.enc_depth,
                blocks_per_decoder_adapter,
            )
        )
        z = []
        for i in range(self.depth):
            if (not isinstance(perm_idx, torch.Tensor)) and (perm_idx is not None):
                task_embeddings = []
                for task, (start, end) in task_range.items():
                    task_tokens = encoder_tokens[indices[i]][:, start:end, ...]
                    if task_tokens.shape[1] == 0:
                        continue
                    p = perm_idx.get(task, None)
                    if p is not None:
                        task_tokens = unshuffle_patches(task_tokens, p)
                    task_embeddings.append(task_tokens[..., None])
                x = torch.concat(task_embeddings, dim=-1).mean(-1)
            else:
                x = encoder_tokens[indices[i]][:, task_range[0] : task_range[1], ...]
                if isinstance(perm_idx, torch.Tensor):
                    x = unshuffle_patches(x, perm_idx, patch_size=1)
            x = self.proj_context(x)
            x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = unpatchify(x, grid_size=patchified_dim, patch_size=1)
            z.append(x)

        if self.unetr_use_input:
            if "images" in batch:
                x_in = batch["images"]
            else:
                x_in = torch.cat(
                    [batch[task] for task in self.input_tasks if task != "seg"],
                    dim=1,
                )
        
        x1, x2, x3, x4 = z[0], z[1], z[2], z[3]
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(x2)
        enc3 = self.encoder3(x3)
        enc4 = self.encoder4(x4)
        dec4 = x1

        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)

        out = self.out(out)
        out = patchify(out, patch_size=self.patch_size, grid_size=patchified_dim)
        if isinstance(perm_idx, torch.Tensor):
            out, _ = shuffle_patches(out, permutations=perm_idx)
        
        return out