import torch

from torch import nn
from einops import rearrange
from torch.distributions import Dirichlet
from timm.layers.helpers import to_3tuple
from omegaconf.dictconfig import DictConfig
from timm.models.vision_transformer import Mlp, LayerScale, DropPath, Block
from typing import Union, List, Tuple, Dict, Optional, Literal

def calc_patchified_dim(
    img_size: Union[int, Tuple[int, int, int]],
    patch_size: Union[int, Tuple[int, int, int]],
) -> Tuple[int, int, int]:
    img_size = to_3tuple(img_size)
    patch_size = to_3tuple(patch_size)
    return tuple([img_size[i] // patch_size[i] for i in range(3)])



def init_image_parameters(img_size, patch_size):
    img_size = to_3tuple(img_size)
    patch_size = to_3tuple(patch_size)
    grid_size = (
        img_size[0] // patch_size[0], 
        img_size[1] // patch_size[1], 
        img_size[2] // patch_size[2], 
    )

    return img_size, patch_size, grid_size


def build_perceptron_position_embedding(num_patches, embed_dim):
    # TODO: Add source: MAEv3 code, slightly altered
    pos_embed = torch.rand([1, num_patches, embed_dim])
    # nn.init.normal_(pos_emb, std=0.5)
    nn.init.normal_(pos_embed, std=0.02)
    return nn.Parameter(pos_embed)

def build_3d_sincos_position_embedding(grid_size, embed_dim, temperature=10000.0):
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d, indexing='ij')
    assert (
        embed_dim % 6 == 0
    ), "Embed dimension must be divisible by 6 for 3D sin-cos position embedding"
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_d = torch.einsum("m,d->md", [grid_d.flatten(), omega])
    pos_emb = torch.cat(
        [
            torch.sin(out_h),
            torch.cos(out_h),
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_d),
            torch.cos(out_d),
        ],
        dim=1,
    )[None, :, :]

    pos_emb = nn.Parameter(pos_emb)
    pos_emb.requires_grad = False
    return pos_emb

def build_position_embeding(num_patches, embed_dim, grid_size, embed_type="sincos"):
    if embed_type == "sincos":
        embed = build_3d_sincos_position_embedding(grid_size=grid_size, embed_dim=embed_dim)
    elif embed_type == "perceptron":
        embed = build_perceptron_position_embedding(num_patches=num_patches, embed_dim=embed_dim)
    else:
        raise NotImplementedError(f"pos_embed_type {embed_type} not implemented")

    return embed



def generate_dirichlet_masked(num_input_tasks: int, alpha, mask_ratio):
    assert mask_ratio >= (
        1 - 1 / num_input_tasks
    ), f"mask_ratio ({mask_ratio}) < selected ratio * num_input_tasks ({1 - 1/num_input_tasks}) -> Could oversample single distribution"
    dirichlet = Dirichlet(torch.tensor([float(alpha)] * num_input_tasks))
    selected_ratio = 1 - mask_ratio
    total_selected = selected_ratio * num_input_tasks
    selected = total_selected * dirichlet.sample()
    return 1 - selected


def get_mask_ratio(tasks, mask_ratio, mask_mode, dirichlet_alpha=1.0):
    """maks mode: ["uniform", "dirichlet"]
    """
    if mask_ratio == 0.0:
        return {task: 0.0 for task in tasks}

    if isinstance(mask_ratio, dict) or isinstance(mask_ratio, DictConfig):
       return mask_ratio

    n_tasks = len(tasks)
    if mask_mode == "uniform":
        return {task: mask_ratio for task in tasks}
    elif mask_mode == "dirichlet":
        dirichlet_mask_ratio = generate_dirichlet_masked(
            num_input_tasks=n_tasks,
            alpha=dirichlet_alpha,
            mask_ratio=mask_ratio,
        )
        mask_ratio = {k: dirichlet_mask_ratio[i].item() for i, k in enumerate(tasks)}
        return mask_ratio
    else:
        assert False

def patchify(image, patch_size, grid_size):
    patch_size = to_3tuple(patch_size)
    patches = rearrange(
        image,
        "b c (nx x) (ny y) (nz z) -> b (nx ny nz) c x y z",
        x=patch_size[0],
        y=patch_size[1],
        z=patch_size[2],
        nx=grid_size[0],
        ny=grid_size[1],
        nz=grid_size[2],
    )

    return patches


def unpatchify(patches, patch_size, grid_size):
    patch_size = to_3tuple(patch_size)
    image = rearrange(
        patches,
        "b (nx ny nz) c x y z -> b c (nx x) (ny y) (nz z)",
        x=patch_size[0],
        y=patch_size[1],
        z=patch_size[2],
        nx=grid_size[0],
        ny=grid_size[1],
        nz=grid_size[2],
    )

    return image


def shuffle_patches(patches, permutations=None):
    batch_size = patches.shape[0]
    num_patches = patches.shape[1]
    if permutations is not None:
        perm_idx = permutations
    else:
        rand = torch.rand(batch_size, num_patches)
        perm_idx = torch.argsort(rand, dim=1)
    patches = patches[torch.arange(batch_size)[:, None], perm_idx, ...]

    return patches, perm_idx


def unshuffle_patches(patches, perm_idx):
    if isinstance(patches, torch.Tensor):
        batch_size = patches.shape[0]
        unshuffled_idx = torch.argsort(perm_idx, dim=1)
        patches = patches[torch.arange(batch_size)[:, None], unshuffled_idx, ...]
    else:
        for task, task_patches in patches.items():
            batch_size = task_patches.shape[0]
            unshuffled_idx = torch.argsort(perm_idx[task], dim=1)
            patches[task] = task_patches[torch.arange(batch_size)[:, None], unshuffled_idx, ...]

    return patches


def mask_data(batch, tasks, mask_ratio, patch_size, grid_size, mask_mode="dirichlet", dirichlet_alpha=1.0, permutations=None):
    mask_ratio = get_mask_ratio(tasks, mask_ratio, mask_mode=mask_mode, dirichlet_alpha=dirichlet_alpha)
    all_patches = {}
    perm_indices = {}
    selected_patches = {}
    masked_patches = {}

    for task in tasks:
        patches = patchify(batch[task], patch_size, grid_size)
        all_patches[task] = patches
        num_patches = patches.shape[1]

        if permutations is not None:
            patches_shuffled, perm_idx = shuffle_patches(patches, permutations[task])
        else:
            patches_shuffled, perm_idx = shuffle_patches(patches)
        perm_indices[task] = perm_idx
        
        num_selected_patches = int((1 - mask_ratio[task]) * num_patches)

        selected_shuffled_patches = patches_shuffled[:, :num_selected_patches]
        selected_patches[task] = selected_shuffled_patches

        masked_shuffled_patches = patches_shuffled[:, num_selected_patches:]
        masked_patches[task] = masked_shuffled_patches

    return (
        all_patches,
        selected_patches,
        masked_patches,
        perm_indices,
        mask_ratio
    )

def build_3d_sincos_position_embedding_as_grid(
    grid_size, embed_dim, temperature=10000.0
):
    grid_size = to_3tuple(grid_size)
    assert embed_dim % 6 == 0, "Embedding dimension must be divisible by 6"

    H, W, D = grid_size
    pos_dim = embed_dim // 6  # Each axis gets (embed_dim / 6) channels

    # Create normalized grid indices
    grid_h = torch.arange(H, dtype=torch.float32).view(H, 1, 1, 1)
    grid_w = torch.arange(W, dtype=torch.float32).view(1, W, 1, 1)
    grid_d = torch.arange(D, dtype=torch.float32).view(1, 1, D, 1)

    # Compute frequency scaling factors
    omega = 1.0 / (
        temperature ** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim)
    )  # (pos_dim,)

    # Compute sinusoidal encodings for each axis
    sin_h, cos_h = torch.sin(grid_h * omega), torch.cos(
        grid_h * omega
    )  # (H, 1, 1, pos_dim)
    sin_w, cos_w = torch.sin(grid_w * omega), torch.cos(
        grid_w * omega
    )  # (1, W, 1, pos_dim)
    sin_d, cos_d = torch.sin(grid_d * omega), torch.cos(
        grid_d * omega
    )  # (1, 1, D, pos_dim)

    # Expand dimensions to match (H, W, D, pos_dim)
    sin_h = sin_h.expand(H, W, D, pos_dim)
    cos_h = cos_h.expand(H, W, D, pos_dim)
    sin_w = sin_w.expand(H, W, D, pos_dim)
    cos_w = cos_w.expand(H, W, D, pos_dim)
    sin_d = sin_d.expand(H, W, D, pos_dim)
    cos_d = cos_d.expand(H, W, D, pos_dim)

    # Concatenate along the last dimension (embedding channels)
    pos_emb = torch.cat(
        [sin_h, cos_h, sin_w, cos_w, sin_d, cos_d], dim=-1
    )  # (H, W, D, embed_dim)

    # Reorder to (1, embed_dim, H, W, D) for Transformer compatibility
    pos_emb = pos_emb.permute(3, 0, 1, 2)  # (1, embed_dim, H, W, D)
    pos_emb = nn.Parameter(pos_emb)
    pos_emb.requires_grad = False

    return pos_emb



def build_position_embedding(
    num_patches, embed_dim, pos_embed_type, patchified_dim, patch_size, global_img_size
):
    global_grid_size = patchified_dim
    global_img_size = to_3tuple(global_img_size)
    if pos_embed_type == "perceptron":
        pos_embed = build_perceptron_position_embedding(
            num_patches=num_patches, embed_dim=embed_dim
        )
    elif pos_embed_type == "sincos":
        pos_embed = build_3d_sincos_position_embedding(
            grid_size=patchified_dim, embed_dim=embed_dim
        )
    elif pos_embed_type == "global_sincos":
        ## compute pos_embed based on a larger "global" grid
        ## any image smaller will be indexed according to position in the global grid
        ## should have same voxel spacing!! - otherwise interpolate
        global_grid_size = calc_patchified_dim(global_img_size, patch_size)
        pos_embed = build_3d_sincos_position_embedding_as_grid(
            grid_size=global_grid_size, embed_dim=embed_dim
        )
    else:
        raise NotImplementedError(f"pos_embed_type {pos_embed_type} not implemented")
    return pos_embed, global_grid_size


def pop_range(
    x: torch.Tensor, r: Tuple[int, int], dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    return x[:, r[0] : r[1]], torch.cat([x[:, : r[0]], x[:, r[1] :]], dim=dim)

def get_cropped_pos_embed(pos_embed, crop_start_coords, crop_end_coords):
    B = crop_start_coords.shape[0]
    C, H, W, D = pos_embed.shape  # (C, H, W, D)

    cropped_embeds = []
    for i in range(B):
        start_h, start_w, start_d = crop_start_coords[i]
        end_h, end_w, end_d = crop_end_coords[i]
        # Crop the positional embedding
        pos_embed_crop = pos_embed[
            :, start_h:end_h, start_w:end_w, start_d:end_d
        ]  # (C, H_crop, W_crop, D_crop)
        # Flatten spatial dimensions while preserving order
        pos_embed_flat = pos_embed_crop.flatten(1).T  # (H_crop*W_crop*D_crop, C)

        cropped_embeds.append(pos_embed_flat)
    return torch.stack(cropped_embeds, dim=0)  # (B, H_crop*W_crop*D_crop, C)

def interpolate_pos_embed(
    pos_embed: torch.Tensor,
    old_grid_size: Union[int, Tuple[int, int, int]],
    new_grid_size: Union[int, Tuple[int, int, int]],
):
    if old_grid_size == new_grid_size:
        return pos_embed

    old_grid_size = to_3tuple(old_grid_size)
    new_grid_size = to_3tuple(new_grid_size)
    pos_embed = rearrange(
        pos_embed,
        "c (nx ny nz) z -> c nx ny nz z",
        nx=old_grid_size[0],
        ny=old_grid_size[1],
        nz=old_grid_size[2],
    )
    pos_embed = pos_embed.permute(0, 4, 1, 2, 3)
    pos_embed = torch.nn.functional.interpolate(
        pos_embed,
        size=new_grid_size,
        mode="trilinear",
        align_corners=False,
    )
    pos_embed = pos_embed.permute(0, 2, 3, 4, 1)
    pos_embed = rearrange(pos_embed, "c nx ny nz z -> c (nx ny nz) z")
    return pos_embed



def get_batch_pos_embed(
    batch,
    pos_embed,
    pos_embed_type,
    input_patchified_dim,
    global_grid_size,
    patch_size,
    batch_size,
):
    if input_patchified_dim != global_grid_size:
        if (
            pos_embed_type == "global_sincos"
        ):  # global_grid is larger than selfpatchified_dim
            crop_start_patchified = torch.stack(
                [batch["crop_start"][:, i] // patch_size[i] for i in range(3)], dim=-1
            ).long()
            crop_end_patchified = torch.stack(
                [batch["crop_end"][:, i] // patch_size[i] for i in range(3)], dim=-1
            ).long()
            batch_pos_embed = get_cropped_pos_embed(
                pos_embed, crop_start_patchified, crop_end_patchified
            )
            # patchified_dim = None # still will be equal to self.patchified_dim (or has to be)
            patchified_dim = input_patchified_dim  # still will be equal to self.patchified_dim (or has to be)
        else:
            pos_embed = interpolate_pos_embed(
                pos_embed, global_grid_size, input_patchified_dim
            )
            batch_pos_embed = pos_embed.repeat(batch_size, 1, 1)
            patchified_dim = input_patchified_dim
    else:
        pos_embed = pos_embed
        batch_pos_embed = pos_embed.repeat(batch_size, 1, 1)
        patchified_dim = input_patchified_dim
    return batch_pos_embed, patchified_dim

class SpecificLayerNorm(torch.nn.LayerNorm):
    def __init__(
        self,
        dim,
        normalized_shape,
        eps=0.00001,
        elementwise_affine=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        permutation = list(range(input.dim()))
        permutation.append(permutation.pop(self.dim))
        input = input.permute(permutation)
        input = super().forward(input)
        permutation = list(range(input.dim()))
        permutation.insert(self.dim, permutation.pop(-1))
        input = input.permute(permutation)
        return input




class CrossAttention(nn.Module):
    # Copied from MultiMAE code
    # TODO: Add source of original code
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(context)
            .reshape(B, M, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FullAttention(nn.Module):
    # Copied from original MultiMAE repository
    # https://github.com/EPFL-VILAB/MultiMAE
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        full_context = torch.cat([context, x], dim=1)

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(full_context)
            .reshape(B, M + N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Copied Block from timm and lightly adjusted
class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        # qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        attn_type: str = None,
    ) -> None:
        super().__init__()
        print(f"Init AttentionBlock - {attn_type}")
        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim)

        # default to None for backwards compatibility with models that don't specify attn_type
        attn_type = "cross" if attn_type is None else attn_type
        ATTN_TYPES = {
            "cross": CrossAttention,
            "full": FullAttention,
        }
        self.xattn = ATTN_TYPES[attn_type](
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls_xattn = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_xattn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp_norm = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls_mlp = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_mlp = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = query + self.drop_xattn(
            self.ls_xattn(
                self.xattn(self.query_norm(query), self.context_norm(context))
            )
        )
        x = x + self.drop_mlp(self.ls_mlp(self.mlp(self.mlp_norm(x))))
        return x


