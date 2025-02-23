
from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr
#from pointnet2_ops import pointnet2_utils
import einops
from timm.models.layers import DropPath
#from lib.pointops.functions import pointops
import pointops
from model.pointtransformer.pcrutils import offset2batch, batch2offset
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
import torch.nn.functional as F
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .block import Block




def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mlp_cls = lambda dim: nn.Linear(dim, dim, **factory_kwargs)
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
       # drop_path=drop_path,
        mlp_cls=mlp_cls
    )
    block.layer_idx = layer_idx
    return block

class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:                   # hidden_states: all 32, 192, 384
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states





class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return self.norm(input.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


class GroupedVectorAttention(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 attn_drop_rate=0.,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True
                 ):
        super(GroupedVectorAttention, self).__init__()
        self.embed_channels = embed_channels
        self.groups = groups
        assert embed_channels % groups == 0
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pe_multiplier = pe_multiplier
        self.pe_bias = pe_bias

        self.linear_q = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(embed_channels, embed_channels, bias=qkv_bias),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )

        self.linear_v = nn.Linear(embed_channels, embed_channels, bias=qkv_bias)

        if self.pe_multiplier:
            self.linear_p_multiplier = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        if self.pe_bias:
            self.linear_p_bias = nn.Sequential(
                nn.Linear(3, embed_channels),
                PointBatchNorm(embed_channels),
                nn.ReLU(inplace=True),
                nn.Linear(embed_channels, embed_channels),
            )
        self.weight_encoding = nn.Sequential(
            nn.Linear(embed_channels, groups),
            PointBatchNorm(groups),
            nn.ReLU(inplace=True),
            nn.Linear(groups, groups)
        )
        self.softmax = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

    def forward(self, feat, coord, reference_index):
        query, key, value = self.linear_q(feat), self.linear_k(feat), self.linear_v(feat)
        key = pointops.grouping(reference_index, key, coord, with_xyz=True)
        value = pointops.grouping(reference_index, value, coord, with_xyz=False)
        pos, key = key[:, :, 0:3], key[:, :, 3:]
        relation_qk = key - query.unsqueeze(1)
        if self.pe_multiplier:
            pem = self.linear_p_multiplier(pos)
            relation_qk = relation_qk * pem
        if self.pe_bias:
            peb = self.linear_p_bias(pos)
            relation_qk = relation_qk + peb
            value = value + peb

        weight = self.weight_encoding(relation_qk)
        weight = self.attn_drop(self.softmax(weight))

        mask = torch.sign(reference_index + 1)
        weight = torch.einsum("n s g, n s -> n s g", weight, mask)
        value = einops.rearrange(value, "n ns (g i) -> n ns g i", g=self.groups)
        feat = torch.einsum("n s g i, n s g -> n g i", value, weight)
        feat = einops.rearrange(feat, "n g i -> n (g i)")
        return feat


class Blocks(nn.Module):
    def __init__(self,
                 embed_channels,
                 groups,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(Blocks, self).__init__()
        self.attn = GroupedVectorAttention(
            embed_channels=embed_channels,
            groups=groups,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias
        )
        self.fc1 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc3 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm1 = PointBatchNorm(embed_channels)
        self.norm2 = PointBatchNorm(embed_channels)
        self.norm3 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, points, reference_index):
        coord, feat, offset = points
        identity = feat
        feat = self.act(self.norm1(self.fc1(feat)))
        feat = self.attn(feat, coord, reference_index) \
            if not self.enable_checkpoint else checkpoint(self.attn, feat, coord, reference_index)
        feat = self.act(self.norm2(feat))
        feat = self.norm3(self.fc3(feat))
        feat = identity + self.drop_path(feat)
        feat = self.act(feat)
        return [coord, feat, offset]

class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / norm


class CalImportance(nn.Module):
    def __init__(self, embed_channels, detach_score_prediction):
        super().__init__()

        self.detach_score_prediction = detach_score_prediction

        # 特征提取器
        self.feature_extractor1 = nn.Sequential(
            nn.Conv1d(embed_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.feature_extractor2 = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, embed_channels, 1)
        )

        # 投影层
        self.projection = nn.Sequential(
            nn.Conv1d(embed_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            Normalize(dim=1)
        )

        # 重要性得分计算器
        self.cal_imp_score = nn.Sequential(
            nn.Conv1d(embed_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, 1)
        )

    def forward(self, patch):
        '''
        输入:
            patch : B G D  (批量大小 B，分组数量 G，特征维度 D)
        输出:
            投影后的特征 : B G D1
            重要性得分 : B G 1
        '''
        patch = patch.transpose(2, 1)  # 转换维度以适应 Conv1d 操作，变为 B D G

        # 通过特征提取器
        patch_feature = self.feature_extractor2(self.feature_extractor1(patch))  # 输出维度依旧是 B D G

        # 投影特征
        map_patch_feature = self.projection(patch_feature)  # 输出 B D1 G

        # 计算重要性得分
        if self.detach_score_prediction:
            pred_importance = self.cal_imp_score(patch_feature.detach())  # 在预测得分时分离梯度
        else:
            pred_importance = self.cal_imp_score(patch_feature)

        # 转换回原始维度顺序，返回投影特征和重要性得分
        return map_patch_feature.transpose(2, 1), pred_importance.transpose(2, 1)  # 返回 B G D1 和 B G 1

def z_curve_serialization(pos):
    B, N, _ = pos.shape

    min_pos = pos.min(dim=1, keepdim=True)[0]
    max_pos = pos.max(dim=1, keepdim=True)[0]
    norm_pos = (pos - min_pos) / (max_pos - min_pos)

    morton_codes = (norm_pos * (2 ** 10 - 1)).long()
    z_order = (morton_codes[:, :, 0] << 2) | (morton_codes[:, :, 1] << 1) | morton_codes[:, :, 2]

    sorted_indices = torch.argsort(z_order, dim=1)

    return sorted_indices

def hilbert_curve_serialization(pos, order=10):
    B, N, _ = pos.shape

    min_pos = pos.min(dim=1, keepdim=True)[0]
    max_pos = pos.max(dim=1, keepdim=True)[0]
    norm_pos = (pos - min_pos) / (max_pos - min_pos)

    hilbert_codes = (norm_pos * (2 ** order - 1)).long()
    hilbert_order = hilbert_index(hilbert_codes, order)

    sorted_indices = torch.argsort(hilbert_order, dim=1)

    return sorted_indices


def hilbert_index(codes, order):
    hilbert_order = (codes[:, :, 0] << 2 * order) | (codes[:, :, 1] << order) | codes[:, :, 2]
    return hilbert_order


class MultiSequenceAggregation(nn.Module):
    def __init__(self, embed_channels, sequences=['hilbert', 'z'], detach_score_prediction=False):
        super(MultiSequenceAggregation, self).__init__()
        self.sequences = sequences
        self.cal_importance = CalImportance(embed_channels, detach_score_prediction=detach_score_prediction)

    def forward(self, pos):
        importance_scores = []

        for seq in self.sequences:
            if seq == 'hilbert':
                serialized_pos = self.hilbert_curve(pos)
            elif seq == 'z':
                serialized_pos = self.z_curve(pos)
            else:
                raise ValueError(f"Unsupported sequence type: {seq}")

            _, importance_score = self.cal_importance(serialized_pos)
            importance_scores.append(importance_score)

        importance_scores = torch.stack(importance_scores, dim=-1)
        importance_scores = importance_scores.mean(dim=-1)

        return importance_scores

    def hilbert_curve(self, pos):
        sorted_indices = hilbert_curve_serialization(pos)
        return torch.gather(pos, 1, sorted_indices.unsqueeze(-1).expand_as(pos))

    def z_curve(self, pos):
        sorted_indices = z_curve_serialization(pos)
        return torch.gather(pos, 1, sorted_indices.unsqueeze(-1).expand_as(pos))



class PositionalEmbedding(nn.Module):
    def __init__(self, embed_channels):
        super(PositionalEmbedding, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(6, 48),  # 将维度从6扩展到48
            nn.GELU(),
            nn.Linear(48, embed_channels)  # 保持与最终的encoder输出通道一致
        )

    def forward(self, coord):
        x = coord[:, 0]
        y = coord[:, 1]
        z = coord[:, 2]

        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = torch.atan2(y, x)
        phi = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))

        pos_embed = torch.stack((x, y, z, r, theta, phi), dim=-1)
        pos_embed = self.linear(pos_embed)

        return pos_embed


class BlockSequence(nn.Module):
    def __init__(self,
                 depth,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False):
        super(BlockSequence, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        elif isinstance(drop_path_rate, float):
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        else:
            drop_path_rates = [0. for _ in range(depth)]

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()

        self.multi_sequence_aggregation = MultiSequenceAggregation(embed_channels)

        for i in range(depth):
            block = Blocks(
                embed_channels=embed_channels,
                groups=groups,
                qkv_bias=qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                enable_checkpoint=enable_checkpoint
            )
            self.blocks.append(block)

        self.pos_embed2 = PositionalEmbedding(embed_channels)

        self.pos_embed1 = nn.Sequential(
            nn.Linear(3, 48),
            nn.GELU(),
            nn.Linear(48, embed_channels)
        )

    def forward(self, points):
        coord, feat, offset = points
        reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)

        for block in self.blocks:
            points = block(points, reference_index)
            coord, feat, offset = points

            n = feat.size(0)
            batch_size = offset.size(0)
            seqlen = n // batch_size

            padding = 0
            if n % batch_size != 0:
                padding = batch_size * (seqlen + 1) - n
                feat = torch.nn.functional.pad(feat, (0, 0, 0, padding))
                coord = torch.nn.functional.pad(coord, (0, 0, 0, padding))
                seqlen += 1

            feat = einops.rearrange(feat, '(b n) c -> b n c', b=batch_size, n=seqlen)

            pos = self.pos_embed2(coord)
            pos = einops.rearrange(pos, '(b n) c -> b n c', b=batch_size, n=seqlen)

            importance_scores = self.multi_sequence_aggregation(pos)

            pos = self.multi_scale_sort(pos, importance_scores)

            mixer = MixerModel(
                d_model=embed_channels,  #384
                n_layer=2,
                rms_norm=False,
                drop_out_in_block=0.1,
                drop_path=0.3
            ).cuda()
            feat = mixer(feat, pos)

            feat = einops.rearrange(feat, 'b n c -> (b n) c')

            if padding != 0:
                feat = feat[:-padding]
                coord = coord[:-padding]

            points = [coord, feat, offset]

        return points

    def multi_scale_sort(self, feat, importance_scores):
        # 获取排序索引
        _, indices = torch.sort(importance_scores, dim=1, descending=True)

        # 调整 indices 维度以匹配 feat 的维度
        indices = indices.expand(-1, -1, feat.size(-1))  # 调整为 [B, N, C] 与 feat 一致

        # 对 feat 进行排序
        sorted_feat = torch.gather(feat, 1, indices)

        return sorted_feat


class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 grid_size,
                 bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, offset = points
        batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = segment_csr(coord, torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                            reduce="min") if start is None else start
        cluster = voxel_grid(pos=coord - start[batch], size=self.grid_size, batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        offset = batch2offset(batch)
        return [coord, feat, offset], cluster


class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 bias=True,
                 skip=True,
                 backend="map"
                 ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels, bias=bias),
                                  PointBatchNorm(out_channels),
                                  nn.ReLU(inplace=True))
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels, bias=bias),
                                       PointBatchNorm(out_channels),
                                       nn.ReLU(inplace=True))

    def forward(self, points, skip_points, cluster=None):
        coord, feat, offset = points
        skip_coord, skip_feat, skip_offset = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            feat = pointops.interpolation(coord, skip_coord, self.proj(feat), offset, skip_offset)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_offset]


class Encoder(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 grid_size=None,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 ):
        super(Encoder, self).__init__()

        self.down = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        points, cluster = self.down(points)
        return self.blocks(points), cluster

class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 skip_channels,
                 embed_channels,
                 groups,
                 depth,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=None,
                 drop_path_rate=None,
                 enable_checkpoint=False,
                 unpool_backend="map"
                 ):
        super(Decoder, self).__init__()

        self.up = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.,
            drop_path_rate=drop_path_rate if drop_path_rate is not None else 0.,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points, skip_points, cluster):
        points = self.up(points, skip_points, cluster)
        return self.blocks(points)


class GVAPatchEmbed(nn.Module):
    def __init__(self,
                 depth,
                 in_channels,
                 embed_channels,
                 groups,
                 neighbours=16,
                 qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 enable_checkpoint=False
                 ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            groups=groups,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            enable_checkpoint=enable_checkpoint
        )

    def forward(self, points):
        coord, feat, offset = points
        feat = self.proj(feat)
        return self.blocks([coord, feat, offset])

class TMPNet(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 patch_embed_depth=2,
                 patch_embed_channels=48,
                 patch_embed_groups=6,
                 patch_embed_neighbours=16,
                 enc_depths=(2, 2, 4, 2),
                 enc_channels=(96, 192, 384, 512),
                 enc_groups=(12, 24, 48, 64),
                 enc_neighbours=(16, 16, 16, 16),
                 dec_depths=(1, 1, 1, 1),
                 dec_channels=(48, 96, 192, 384),
                 dec_groups=(6, 12, 24, 48),
                 dec_neighbours=(16, 16, 16, 16),
                 grid_sizes=(0.1, 0.2, 0.4, 0.8),
                 attn_qkv_bias=True,
                 pe_multiplier=False,
                 pe_bias=True,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 enable_checkpoint=False,
                 unpool_backend="interp",
                 mixer_config=None  # Add config parameter for MixerModel
                 ):
        super(TMPNet, self).__init__()

        self.num_stages = len(enc_depths)
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)

        self.patch_embed = GVAPatchEmbed(
            in_channels=6,
            embed_channels=patch_embed_channels,
            groups=patch_embed_groups,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            pe_multiplier=pe_multiplier,
            pe_bias=pe_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint
        )

        enc_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(enc_depths))]
        dec_dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(dec_depths))]
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                groups=enc_groups[i],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=enc_dp_rates[sum(enc_depths[:i]):sum(enc_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint
            )
            self.enc_stages.append(enc)

        #self.mixer = MixerModel(
         #   d_model=enc_channels[-1],  # Use the last encoder's output channels
         #   n_layer=12,  # 12 layers for the final integration
         #   rms_norm=False,
          #  drop_out_in_block=0.1,
           # drop_path=0.3
        #)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),  # Example encoding dimensions
            nn.GELU(),
            nn.Linear(128, enc_channels[-1])  # Match the last encoder's output channels
        )
        for i in range(self.num_stages):
            dec = Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                groups=dec_groups[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                pe_multiplier=pe_multiplier,
                pe_bias=pe_bias,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dec_dp_rates[sum(dec_depths[:i]):sum(dec_depths[:i + 1])],
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend
            )
            self.dec_stages.append(dec)

        self.seg_head = nn.Sequential(
            nn.Linear(dec_channels[0], dec_channels[0]),
            PointBatchNorm(dec_channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(dec_channels[0], num_classes)
        ) if num_classes > 0 else nn.Identity()

    def forward(self, pxo):
        coord, feat, offset = pxo  # (n, 3), (n, c), (b)
        #device = coord.device
        feat = torch.cat((coord, feat), 1)
        offset = offset

       # print('coord',coord.shape)
        #print('feat', feat.shape)
        #print('offset', offset.shape)
        points = [coord, feat, offset]
        points = self.patch_embed(points)
        skips = [[points]]
        for i in range(self.num_stages):
            points, cluster = self.enc_stages[i](points)
            skips[-1].append(cluster)  # record grid cluster of pooling
            skips.append([points])  # record points info of current stage

        points = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points, cluster = skips.pop(-1)
            points = self.dec_stages[i](points, skip_points, cluster)
        coord, feat, offset = points
        seg_logits = self.seg_head(feat)
        return seg_logits

