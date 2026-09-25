import math
import torch
import logging


import torch.nn as nn

from typing import Union, List, Tuple, Dict, Optional
from collections import OrderedDict
from models.multimae3d_utils import (
    init_image_parameters,
    build_position_embeding,
    unpatchify,
    unshuffle_patches,
    mask_data,
)

from models.multimae3d_adapters import (
    PatchedInputAdapter,
    SpatialOutputAdapter,
    UNETROutputAdapter,
    LinearOutputAdapter,
)

from models.vim_encoder import vim_encoder_base

logger = logging.getLogger(__name__)

class MultiMAE3D(nn.Module):
    def __init__(self, 
                input_adapters: Dict, output_adapters: Dict, encoder: nn.Module,
                mask_ratio, img_size=(160,176,144), patch_size=(16,16,16), cls_token=True, embed_dim=768, 
                mask_mode='dirichlet', dirichlet_alpha=1.0, embed_type="sincos",
                middle_token=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ==========================================================================
        # Init parameters
        self.img_size, self.patch_size, self.grid_size = init_image_parameters(img_size, patch_size)
        self.num_patches = (self.grid_size[0] * self.grid_size[1] * self.grid_size[2])

        self.cls_token = cls_token
        self.middle_token = middle_token
        self.embed_dim = embed_dim
        self.embed_type = embed_type

        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.dirichlet_alpha = dirichlet_alpha

        # ==========================================================================
        # Init input_adapters and output_adapters 
        for _, adapter in input_adapters.items():
            adapter.init(embed_dim=self.embed_dim)
        self.input_adapters = nn.ModuleDict(input_adapters)

        for _, adapter in output_adapters.items():
            adapter.init(
                enc_embed_dim=embed_dim,
                enc_depth=encoder.depth,
                input_tasks=self.input_adapters.keys()
            )
        self.output_adapters = nn.ModuleDict(output_adapters)

        self.encoder = encoder

        if self.cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, self.embed_dim))
        else:
            self.cls_token = None

        self.embed = build_position_embeding(self.num_patches, self.embed_dim, self.grid_size, self.embed_type) # (1, num_patches, embed_dim)

        self.initialize_weights()
        
    def initialize_weights(self):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if "encoder" in name:
                continue
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
            if isinstance(m, nn.Conv3d):
                if ".proj" in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, batch, mask_ratio=None, return_image=True, mask_mode=None, dirichlet_alpha=None, same_permutations=False, complementarity=False):
        tokens = {}
        selected_pos_embed = {}
        reconstructed_patches = {}

        mask_mode = self.mask_mode if mask_mode is None else mask_mode
        dirichlet_alpha = self.dirichlet_alpha if dirichlet_alpha is None else dirichlet_alpha
        if isinstance(self.mask_ratio, dict):
            mask_ratio = self.mask_ratio.copy() if mask_ratio is None else mask_ratio
        else:
            mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio

        input_tasks = list(self.input_adapters)
        if same_permutations:
            rand = torch.rand(batch.shape[0], self.num_patches)
            perm_idx = torch.argsort(rand, dim=1)
            permutations = {task: perm_idx for task in input_tasks}
        else:
            permutations = None
        
        all_patches, selected_patches, masked_patches, perm_indices, mask_ratio = mask_data(
            batch=batch,
            tasks=input_tasks,
            mask_ratio=mask_ratio, 
            grid_size=self.grid_size,
            mask_mode=mask_mode,
            patch_size=self.patch_size,
            dirichlet_alpha=dirichlet_alpha,
            permutations=permutations,
        )

        for task in self.input_adapters:
            selected_shuffled_patches = selected_patches[task]
            num_selected_patches = selected_shuffled_patches.shape[1]
            perm_idx = perm_indices[task]
            batch_size = selected_shuffled_patches.shape[0]

            if task == "seg":
                selected_shuffled_patches = selected_shuffled_patches.float()

            tokens[task] = self.input_adapters[task](selected_shuffled_patches)
            batch_pos_embed = self.embed.repeat(batch_size, 1, 1)
            batch_pos_embed = batch_pos_embed[torch.arange(batch_size)[:, None], perm_idx, ...]
            selected_pos_embed[task] = batch_pos_embed[:, :num_selected_patches]

            tokens[task] = tokens[task] + selected_pos_embed[task]

        # =====================================================================
        # Concatenate tokens for each task
        task_order = list(tokens.keys())
        task_ranges = OrderedDict()

        # Calculate the range of tokens for each task
        # lower_bound = 1 if self.cls_token is not None else 0
        lower_bound = 0
        for task in task_order:
            upper_bound = lower_bound + tokens[task].shape[1]
            task_ranges[task] = (lower_bound, upper_bound)
            lower_bound = upper_bound

        input_tokens = torch.cat([tokens[task] for task in task_order], dim=1)

        # Add cls_token
        if self.cls_token is not None:
            batch_global_tokens = self.cls_token.repeat(input_tokens.shape[0], 1, 1)
            if self.middle_token:
                middle_index = input_tokens.shape[1] // 2
                input_tokens = torch.cat([input_tokens[:, :middle_index, ...], batch_global_tokens, input_tokens[:, middle_index:, ...]], dim=1)
            else:
                input_tokens = torch.cat([batch_global_tokens, input_tokens], dim=1)
        # =====================================================================
        # Pass Encoder
        encoder_tokens = []
        x_hat = input_tokens
        if self.encoder is None:
            encoder_tokens = [input_tokens]
        else:
            encoder_tokens = [*self.encoder(x_hat)]

        spatial_encoder_tokens = []
        for layer_tokens in encoder_tokens:
            if self.cls_token is not None:
                if self.middle_token:
                    mid = (layer_tokens.shape[1] - 1) // 2
                    spatial_only = torch.cat([layer_tokens[:, :mid, ...], layer_tokens[:, mid+1:, ...]], dim=1)
                else:
                    spatial_only = layer_tokens[:, 1:, ...]
                spatial_encoder_tokens.append(spatial_only)
            else:
                spatial_encoder_tokens.append(layer_tokens)

        # =====================================================================
        # Pass through output adapters
        for task, output_adapter in self.output_adapters.items():
            if hasattr(output_adapter, 'visualizable') and not output_adapter.visualizable:
                pass_tokens = encoder_tokens
            else:
                pass_tokens = spatial_encoder_tokens 

            task_range = task_ranges[task] if task in task_ranges else None
            perm_index = perm_indices[task] if task in perm_indices else None
            if hasattr(output_adapter, 'unetr_use_input'):  
                perm_index = perm_indices if perm_index is None else perm_index
                task_range = task_ranges if task_range is None else task_range
                
            output_patches = output_adapter(
                pass_tokens,
                task_range,
                perm_index,
                self.grid_size,
                batch=batch,
                adapter_task=task,
            )

            num_selected_patches = (selected_patches[task].shape[1] if task in selected_patches else 0)
            num_masked_patches = output_patches.shape[1] - num_selected_patches

            if task in masked_patches:
                masked_patches[task] = masked_patches[task][ :, :num_masked_patches]  # remove the patches ignored by the decoder

            task_range = task_ranges[task] if task in task_ranges else None
            perm_index = perm_indices[task] if task in perm_indices else None

            if (return_image and output_adapter.visualizable):  # only for testing, needs full size
                if perm_index is not None and isinstance(perm_index, torch.Tensor):
                    output_patches = unshuffle_patches(output_patches, perm_index)
                output_patches = unpatchify(output_patches, self.patch_size, self.grid_size)
            reconstructed_patches[task] = output_patches

        # if return_as_image is true, unshuffle and unpatchify the selected input patches with zeros for masked patches
        if return_image:
            for task in selected_patches:
                # masked_patches can also be empty but has an entry in the dict
                if task in masked_patches:
                    patches = torch.cat(
                        [
                            selected_patches[task],
                            torch.zeros_like(masked_patches[task]),
                        ],
                        dim=1,
                    )
                    patches = unshuffle_patches(patches, perm_indices[task])
                    patches = unpatchify(patches, self.patch_size, self.grid_size)
                    selected_patches[task] = (patches) # overwrite selected patches with full size image of selected and masked (zeros)

        
        return {
            "selected_patches": selected_patches,
            "masked_patches": masked_patches,
            "perm_indices": perm_indices,
            "tokens": tokens,
            "encoder_tokens": encoder_tokens,
            "task_ranges": task_ranges,
            "reconstructed_patches": reconstructed_patches,
        }
       
 

def instantiate_input_adapters(
    patch_size=(16,16,16), in_channels=1, input_tasks=['t1n','t1c','t2w','t2f'], 
    encoder_embed_dim=768, adapter_depth=12, seg_classes=None
):
    input_adapters = []
    for task in input_tasks:
        task_in_channels = (seg_classes if task == "seg" else in_channels)
        adapter = PatchedInputAdapter(
            task_in_channels,
            patch_size,
            embed_dim=encoder_embed_dim,
            depth=adapter_depth
        )
        input_adapters.append((task, adapter))
    return OrderedDict(input_adapters)

def instantiate_SpatialOutputAdapter(
    out_channels=1, img_size=(160,176,144), patch_size=(16,16,16), output_tasks=['t1n','t1c','t2w','t2f'],
    encoder_embed_dim=768, embed_dim=768, num_heads=12, depth=4, pos_embed_type="sincos"
):
    output_adapters = []
    for task in output_tasks:
        adapter = SpatialOutputAdapter(
            out_channels=out_channels, img_size=img_size, patch_size=patch_size,
            enc_embed_dim=encoder_embed_dim, embed_dim=embed_dim, num_heads=num_heads, depth=depth, pos_embed_type=pos_embed_type
        )
        output_adapters.append((task, adapter))
    return OrderedDict(output_adapters)


def instantiate_UNETROutputAdapter(
    seg_classes=3, img_size=(160,176,144), patch_size=(16,16,16), embed_dim=768, output_tasks=["seg"],
    unetr_in_channels=4, encoder_embed_dim=768, enc_depth=24,input_tasks=['t1n','t1c','t2w','t2f']
):
    output_adapters = []
    for task in output_tasks:
        adapter = UNETROutputAdapter(
            out_channels=seg_classes, img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            unetr_in_channels=unetr_in_channels, enc_embed_dim=encoder_embed_dim, enc_depth=enc_depth,input_tasks=input_tasks
        )
        output_adapters.append((task, adapter))
    return OrderedDict(output_adapters)


def instantiate_LinearOutputAdapter(
    num_classes=2, enc_embed_dim=768, output_tasks=["label"],middle_token=True,
):
    output_adapters = []
    for task in output_tasks:
        agg_mode = "middle" if middle_token else "cls"
        adapter = LinearOutputAdapter(num_classes=num_classes, enc_embed_dim=enc_embed_dim, token_aggregation=agg_mode)
        output_adapters.append((task, adapter))
    return OrderedDict(output_adapters)


def instantiate_MultiMAE(
    task, img_size=(160,176,144) ,patch_size=(16,16,16), in_channels=1, input_tasks=['t1n','t1c','t2w','t2f'],
    encoder_embed_dim=768, adapter_depth=12, encoder_depth=24, seg_classes=3, output_tasks=['t1n','t1c','t2w','t2f'], num_classes=2,
    mask_ratio=0.75, mask_mode="dirichlet", middle_token=True,
    *args, **kwargs
):
    input_adapters = instantiate_input_adapters(patch_size, in_channels, input_tasks, encoder_embed_dim, adapter_depth, seg_classes)

    if task == "rec":
        output_apapters = instantiate_SpatialOutputAdapter(
            out_channels=in_channels, img_size=img_size, patch_size=patch_size,output_tasks=output_tasks,
            encoder_embed_dim=encoder_embed_dim
        )
    elif task == "seg":
        output_apapters = instantiate_UNETROutputAdapter(
            seg_classes=seg_classes, img_size=img_size, patch_size=patch_size, output_tasks=output_tasks, 
            enc_depth=encoder_depth, encoder_embed_dim=encoder_embed_dim,input_tasks=input_tasks
        )
    elif task == "cla":
        output_apapters = instantiate_LinearOutputAdapter(num_classes=num_classes, enc_embed_dim=encoder_embed_dim, 
                                                          output_tasks=output_tasks, middle_token=middle_token)
    
    encoder = vim_encoder_base(embed_dim=encoder_embed_dim, depth=encoder_depth)
    
    mae = MultiMAE3D(
        input_adapters, output_apapters, encoder,
        mask_ratio=mask_ratio, mask_mode=mask_mode, img_size=img_size, patch_size=patch_size, cls_token=True, middle_token=middle_token
    )

    return mae