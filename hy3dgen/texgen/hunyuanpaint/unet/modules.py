# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the respective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_2d import BasicTransformerBlock
from einops import rearrange


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]}"
            f"has to be divisible by chunk size: {chunk_size}."
            f" Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


class Basic2p5DTransformerBlock(torch.nn.Module):
    def __init__(self, transformer: BasicTransformerBlock,layer_name,use_ma=True,use_ra=True,is_turbo=False) -> None:
        super().__init__()
        self.transformer = transformer
        self.layer_name = layer_name
        self.use_ma = use_ma
        self.use_ra = use_ra
        self.is_turbo = is_turbo

        # multiview attn
        if self.use_ma:
            self.attn_multiview = Attention(
                query_dim=self.dim,
                heads=self.num_attention_heads,
                dim_head=self.attention_head_dim,
                dropout=self.dropout,
                bias=self.attention_bias,
                cross_attention_dim=None,
                upcast_attention=self.attn1.upcast_attention,
                out_bias=True,
            )

        # ref attn
        if self.use_ra:
            self.attn_refview = Attention(
                query_dim=self.dim,
                heads=self.num_attention_heads,
                dim_head=self.attention_head_dim,
                dropout=self.dropout,
                bias=self.attention_bias,
                cross_attention_dim=None,
                upcast_attention=self.attn1.upcast_attention,
                out_bias=True,
            )
        if self.is_turbo:
            self._initialize_attn_weights()

    def _initialize_attn_weights(self):

        if self.use_ma:
            self.attn_multiview.load_state_dict(self.attn1.state_dict()) 
            with torch.no_grad():
                for layer in self.attn_multiview.to_out:
                    for param in layer.parameters():
                        param.zero_()
        if self.use_ra:
            self.attn_refview.load_state_dict(self.attn1.state_dict()) 
            with torch.no_grad():
                for layer in self.attn_refview.to_out:
                    for param in layer.parameters():
                        param.zero_()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        num_in_batch = cross_attention_kwargs.pop('num_in_batch', 1)
        mode = cross_attention_kwargs.pop('mode', None)
        if not self.is_turbo:
            mva_scale = cross_attention_kwargs.pop('mva_scale', 1.0)
            ref_scale = cross_attention_kwargs.pop('ref_scale', 1.0)
        else:
            position_attn_mask = cross_attention_kwargs.pop("position_attn_mask", None)
            position_voxel_indices = cross_attention_kwargs.pop("position_voxel_indices", None)
            mva_scale = 1.0
            ref_scale = 1.0
            
        condition_embed_dict = cross_attention_kwargs.pop("condition_embed_dict", None)

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 Reference Attention
        if 'w' in mode:
            condition_embed_dict[self.layer_name] = rearrange(
                norm_hidden_states, '(b n) l c -> b (n l) c',
                n=num_in_batch
            )  # B, (N L), C

        if 'r' in mode and self.use_ra:
            condition_embed = condition_embed_dict[self.layer_name].unsqueeze(1).repeat(1, num_in_batch, 1,
                                                                                        1)  # B N L C
            condition_embed = rearrange(condition_embed, 'b n l c -> (b n) l c')

            attn_output = self.attn_refview(
                norm_hidden_states,
                encoder_hidden_states=condition_embed,
                attention_mask=None,
                **cross_attention_kwargs
            )
            if not self.is_turbo:
                ref_scale_timing = ref_scale
                if isinstance(ref_scale, torch.Tensor):
                    ref_scale_timing = ref_scale.unsqueeze(1).repeat(1, num_in_batch).view(-1)
                    for _ in range(attn_output.ndim - 1):
                        ref_scale_timing = ref_scale_timing.unsqueeze(-1)
                        
            hidden_states = ref_scale_timing * attn_output + hidden_states

            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        # 1.3 Multiview Attention
        if num_in_batch > 1 and self.use_ma:
            multivew_hidden_states = rearrange(norm_hidden_states, '(b n) l c -> b (n l) c', n=num_in_batch)

            if self.is_turbo:
                position_mask = None
                if position_attn_mask is not None:
                    if multivew_hidden_states.shape[1] in position_attn_mask:
                        position_mask = position_attn_mask[multivew_hidden_states.shape[1]]
                position_indices = None
                if position_voxel_indices is not None:
                    if multivew_hidden_states.shape[1] in position_voxel_indices:
                        position_indices = position_voxel_indices[multivew_hidden_states.shape[1]]
                attn_output = self.attn_multiview(
                    multivew_hidden_states,
                    encoder_hidden_states=multivew_hidden_states,
                    attention_mask=position_mask,
                    position_indices=position_indices,
                    **cross_attention_kwargs
                )
            else:
                attn_output = self.attn_multiview(
                    multivew_hidden_states,
                    encoder_hidden_states=multivew_hidden_states,
                    **cross_attention_kwargs
                )

            attn_output = rearrange(attn_output, 'b (n l) c -> (b n) l c', n=num_in_batch)
            
            hidden_states = mva_scale * attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )

            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

@torch.no_grad()
def compute_voxel_grid_mask(position, grid_resolution=8):

    position = position.half()    
    B,N,_,H,W = position.shape
    assert H%grid_resolution==0 and W%grid_resolution==0

    valid_mask = (position != 1).all(dim=2, keepdim=True)
    valid_mask = valid_mask.expand_as(position)
    position[valid_mask==False] = 0

    
    position = rearrange(
        position,
        'b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w', 
        num_h=grid_resolution, num_w=grid_resolution
    )
    valid_mask = rearrange(
        valid_mask, 
        'b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w', 
        num_h=grid_resolution, num_w=grid_resolution
    )

    grid_position = position.sum(dim=(-2, -1))
    count_masked = valid_mask.sum(dim=(-2, -1))

    grid_position = grid_position / count_masked.clamp(min=1)
    grid_position[count_masked<5] = 0

    grid_position = grid_position.permute(0,1,4,2,3)
    grid_position = rearrange(grid_position, 'b n c h w -> b n (h w) c')

    grid_position_expanded_1 = grid_position.unsqueeze(2).unsqueeze(4)  # 形状变为 B, N, 1, L, 1, 3
    grid_position_expanded_2 = grid_position.unsqueeze(1).unsqueeze(3)  # 形状变为 B, 1, N, 1, L, 3

    # 计算欧氏距离
    distances = torch.norm(grid_position_expanded_1 - grid_position_expanded_2, dim=-1)  # 形状为 B, N, N, L, L

    weights = distances
    grid_distance = 1.73/grid_resolution
    
    #weights = weights*-32
    #weights = weights.clamp(min=-10000.0)
    
    weights = weights< grid_distance

    return weights
    
def compute_multi_resolution_mask(position_maps, grid_resolutions=[32, 16, 8]):
    position_attn_mask = {}
    with torch.no_grad():
        for grid_resolution in grid_resolutions:
            position_mask = compute_voxel_grid_mask(position_maps, grid_resolution)
            position_mask = rearrange(position_mask, 'b ni nj li lj -> b (ni li) (nj lj)')
            position_attn_mask[position_mask.shape[1]] = position_mask
    return position_attn_mask

@torch.no_grad()
def compute_discrete_voxel_indice(position, grid_resolution=8, voxel_resolution=128):

    position = position.half()    
    B,N,_,H,W = position.shape
    assert H%grid_resolution==0 and W%grid_resolution==0

    valid_mask = (position != 1).all(dim=2, keepdim=True)
    valid_mask = valid_mask.expand_as(position)
    position[valid_mask==False] = 0
    
    position = rearrange(
        position, 
        'b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w', 
        num_h=grid_resolution, num_w=grid_resolution
    )
    valid_mask = rearrange(
        valid_mask, 
        'b n c (num_h grid_h) (num_w grid_w) -> b n num_h num_w c grid_h grid_w', 
        num_h=grid_resolution, num_w=grid_resolution
    )

    grid_position = position.sum(dim=(-2, -1))
    count_masked = valid_mask.sum(dim=(-2, -1))

    grid_position = grid_position / count_masked.clamp(min=1)
    grid_position[count_masked<5] = 0

    grid_position = grid_position.permute(0,1,4,2,3).clamp(0, 1) # B N C H W
    voxel_indices = grid_position * (voxel_resolution - 1)
    voxel_indices = torch.round(voxel_indices).long()
    return voxel_indices
    
def compute_multi_resolution_discrete_voxel_indice(
    position_maps, 
    grid_resolutions=[64, 32, 16, 8], 
    voxel_resolutions=[512, 256, 128, 64]
):
    voxel_indices = {}
    with torch.no_grad():
        for grid_resolution, voxel_resolution in zip(grid_resolutions, voxel_resolutions):
            voxel_indice = compute_discrete_voxel_indice(position_maps, grid_resolution, voxel_resolution)
            voxel_indice = rearrange(voxel_indice, 'b n c h w -> b (n h w) c')
            voxel_indices[voxel_indice.shape[1]] = {'voxel_indices':voxel_indice, 'voxel_resolution':voxel_resolution}
    return voxel_indices

class UNet2p5DConditionModel(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel) -> None:
        super().__init__()
        self.unet = unet

        self.use_ma = True
        self.use_ra = True
        self.use_camera_embedding = True
        self.use_dual_stream = True
        self.is_turbo = False

        if self.use_dual_stream:
            self.unet_dual = copy.deepcopy(unet)
            self.init_attention(self.unet_dual)
        self.init_attention(self.unet, use_ma=self.use_ma, use_ra=self.use_ra, is_turbo=self.is_turbo)
        self.init_condition()
        self.init_camera_embedding()

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        torch_dtype = kwargs.pop('torch_dtype', torch.float32)
        config_path = os.path.join(pretrained_model_name_or_path, 'config.json')
        unet_ckpt_path = os.path.join(pretrained_model_name_or_path, 'diffusion_pytorch_model.bin')
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        unet = UNet2DConditionModel(**config)
        unet = UNet2p5DConditionModel(unet)
        unet_ckpt = torch.load(unet_ckpt_path, map_location='cpu', weights_only=True)
        unet.load_state_dict(unet_ckpt, strict=True)
        unet = unet.to(torch_dtype)
        return unet

    def init_condition(self):
        self.unet.conv_in = torch.nn.Conv2d(
            12,
            self.unet.conv_in.out_channels,
            kernel_size=self.unet.conv_in.kernel_size,
            stride=self.unet.conv_in.stride,
            padding=self.unet.conv_in.padding,
            dilation=self.unet.conv_in.dilation,
            groups=self.unet.conv_in.groups,
            bias=self.unet.conv_in.bias is not None)

        self.unet.learned_text_clip_gen = nn.Parameter(torch.randn(1, 77, 1024))
        self.unet.learned_text_clip_ref = nn.Parameter(torch.randn(1, 77, 1024))

    def init_camera_embedding(self):

        if self.use_camera_embedding:
            time_embed_dim = 1280
            self.max_num_ref_image = 5
            self.max_num_gen_image = 12 * 3 + 4 * 2
            self.unet.class_embedding = nn.Embedding(self.max_num_ref_image + self.max_num_gen_image, time_embed_dim)

    def init_attention(self, unet, use_ma=False, use_ra=False, is_turbo=False):

        for down_block_i, down_block in enumerate(unet.down_blocks):
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                for attn_i, attn in enumerate(down_block.attentions):
                    for transformer_i, transformer in enumerate(attn.transformer_blocks):
                        if isinstance(transformer, BasicTransformerBlock):
                            attn.transformer_blocks[transformer_i] = Basic2p5DTransformerBlock(
                                transformer,
                                f'down_{down_block_i}_{attn_i}_{transformer_i}',
                                use_ma, use_ra, is_turbo
                            )

        if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
            for attn_i, attn in enumerate(unet.mid_block.attentions):
                for transformer_i, transformer in enumerate(attn.transformer_blocks):
                    if isinstance(transformer, BasicTransformerBlock):
                        attn.transformer_blocks[transformer_i] = Basic2p5DTransformerBlock(
                            transformer,
                            f'mid_{attn_i}_{transformer_i}',
                            use_ma, use_ra, is_turbo
                        )

        for up_block_i, up_block in enumerate(unet.up_blocks):
            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                for attn_i, attn in enumerate(up_block.attentions):
                    for transformer_i, transformer in enumerate(attn.transformer_blocks):
                        if isinstance(transformer, BasicTransformerBlock):
                            attn.transformer_blocks[transformer_i] = Basic2p5DTransformerBlock(
                                transformer,
                                f'up_{up_block_i}_{attn_i}_{transformer_i}',
                                use_ma, use_ra, is_turbo
                            )

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(
        self, sample, timestep, encoder_hidden_states,
        *args, down_intrablock_additional_residuals=None,
        down_block_res_samples=None, mid_block_res_sample=None,
        **cached_condition,
    ):
        B, N_gen, _, H, W = sample.shape
        assert H == W

        if self.use_camera_embedding:
            camera_info_gen = cached_condition['camera_info_gen'] + self.max_num_ref_image
            camera_info_gen = rearrange(camera_info_gen, 'b n -> (b n)')
        else:
            camera_info_gen = None

        sample = [sample]
        if 'normal_imgs' in cached_condition:
            sample.append(cached_condition["normal_imgs"])
        if 'position_imgs' in cached_condition:
            sample.append(cached_condition["position_imgs"])
        sample = torch.cat(sample, dim=2)

        sample = rearrange(sample, 'b n c h w -> (b n) c h w')

        encoder_hidden_states_gen = encoder_hidden_states.unsqueeze(1).repeat(1, N_gen, 1, 1)
        encoder_hidden_states_gen = rearrange(encoder_hidden_states_gen, 'b n l c -> (b n) l c')

        if self.use_ra:
            if 'condition_embed_dict' in cached_condition:
                condition_embed_dict = cached_condition['condition_embed_dict']
            else:
                condition_embed_dict = {}
                ref_latents = cached_condition['ref_latents']
                N_ref = ref_latents.shape[1]
                if self.use_camera_embedding:
                    camera_info_ref = cached_condition['camera_info_ref']
                    camera_info_ref = rearrange(camera_info_ref, 'b n -> (b n)')
                else:
                    camera_info_ref = None

                ref_latents = rearrange(ref_latents, 'b n c h w -> (b n) c h w')

                encoder_hidden_states_ref = self.unet.learned_text_clip_ref.unsqueeze(1).repeat(B, N_ref, 1, 1)
                encoder_hidden_states_ref = rearrange(encoder_hidden_states_ref, 'b n l c -> (b n) l c')

                noisy_ref_latents = ref_latents
                timestep_ref = 0

                if self.use_dual_stream:
                    unet_ref = self.unet_dual
                else:
                    unet_ref = self.unet
                unet_ref(
                    noisy_ref_latents, timestep_ref,
                    encoder_hidden_states=encoder_hidden_states_ref,
                    class_labels=camera_info_ref,
                    # **kwargs
                    return_dict=False,
                    cross_attention_kwargs={
                        'mode': 'w', 'num_in_batch': N_ref,
                        'condition_embed_dict': condition_embed_dict},
                )
                cached_condition['condition_embed_dict'] = condition_embed_dict
        else:
            condition_embed_dict = None

        mva_scale = cached_condition.get('mva_scale', 1.0)
        ref_scale = cached_condition.get('ref_scale', 1.0)

        if self.is_turbo:
            cross_attention_kwargs_ = {
                'mode': 'r', 'num_in_batch': N_gen,
                'condition_embed_dict': condition_embed_dict,
                'position_attn_mask':position_attn_mask, 
                'position_voxel_indices':position_voxel_indices,
                'mva_scale': mva_scale,
                'ref_scale': ref_scale,
            }
        else:
            cross_attention_kwargs_ = {
                'mode': 'r', 'num_in_batch': N_gen,
                'condition_embed_dict': condition_embed_dict,
                'mva_scale': mva_scale,
                'ref_scale': ref_scale,
            }
        return self.unet(
            sample, timestep,
            encoder_hidden_states_gen, *args,
            class_labels=camera_info_gen,
            down_intrablock_additional_residuals=[
                sample.to(dtype=self.unet.dtype) for sample in down_intrablock_additional_residuals
            ] if down_intrablock_additional_residuals is not None else None,
            down_block_additional_residuals=[
                sample.to(dtype=self.unet.dtype) for sample in down_block_res_samples
            ] if down_block_res_samples is not None else None,
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=self.unet.dtype)
                if mid_block_res_sample is not None else None
            ),
            return_dict=False,
            cross_attention_kwargs=cross_attention_kwargs_,
        )
