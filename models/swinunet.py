import math
import torch
import numpy as np
from torch import nn
from torch.nn.functional import dropout
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
#from einops.layers.torch import Rearrange
from einops import rearrange
from models.model_blocks import ResnetBlock, SinusoidalPosEmb, Upsample, Downsample, LinearAttention, Attention
from models.model_helpers import exists, default
import utils.helpers as h
from models.pos_embeddings import get_positional_embed

"""
Original Swin Transformer v2 at
https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
"""

def create_padding_mask(b, remainder, padding_size, device):
    mask = torch.zeros(
        b, remainder + padding_size, dtype=torch.bool, device=device
    )
    # set the padding part to True
    mask[:, remainder:] = True
    return mask

def relative_shift_swin(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    h, _, window, t1, t2 = x.shape
    x = x.reshape(h, -1, window, t2, t1)
    x = x[:, :, :, 1:, :]
    x = x.reshape(h, -1, window, t1, t2 - 1)
    # up to this point: (i,j) in x represents the emb of dot product (i,i-j)
    return x[..., : ((t2 + 1) // 2)]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class PatchEmbed(nn.Module):
    def __init__(self, signal_length, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.signal_length = signal_length
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # embedding into non-overlapping windows: stride = patch_size
        # embedding into overlapping windows: stride = patch_size//2
        #self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=3, padding=1, padding_mode='circular')
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        b, c, s_w = x.shape
        assert s_w == self.signal_length, \
            f"Input signal size ({s_w}) doesn't match model ({self.signal_length})."
        x = self.proj(x).transpose(1, 2)  # b s_w c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (int): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim_in, input_resolution):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_in = dim_in
        self.reduction = nn.Linear(2 * dim_in, 2*dim_in, bias=False)    # 2 2
        self.norm = nn.LayerNorm(2*dim_in)

    def forward(self, x):
        """
        x: B, s_w/2, C
        """
        signal_length = self.input_resolution
        B, L, C = x.shape
        #assert L == signal_length, "input feature has wrong size"
        #assert signal_length % 2 == 0, f"x size (signal_length) is not even."

        x0 = x[:, 0::2, :]  # B s_w/2 C
        x1 = x[:, 1::2, :]  # B s_w/2 C
        x = torch.cat([x0, x1], -1)  # B s_w/2 2*C
        x = x.view(B, -1, 2 * C)  # B s_w/2 2*C
        x = self.norm(x)
        x = self.reduction(x)
        #print(x.shape)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim_in}"

    def flops(self):
        s_w = self.input_resolution
        flops = (s_w // 2) * 2 * self.dim_in * 1 * self.dim_in
        flops += s_w * self.dim_in // 2
        return flops


class PatchExpand(nn.Module):
    def __init__(self, dim, input_resolution, dim_scale=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = nn.LayerNorm(dim // dim_scale)

    def forward(self, x):
        """
        x: B, s_w, C
        """
        s_w = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        #assert L == s_w, f"input feature has wrong size ({L} vs. {s_w})"
        x = x.view(B, -1, C//2)
        x = self.norm(x)
        return x




class WindowAttention(nn.Module):
    def __init__(self, seq_len, dim, dim_key, window_size, num_heads, window_bias=False, dropout=0.0,
                 num_rel_pos_features=None, padding_mode=True):
        super().__init__()
        self.dim = dim
        self.dim_key = dim_key
        self.num_heads = num_heads
        self.window_size = window_size

        self.to_qkv = nn.Linear(dim, dim_key * 3 * num_heads, bias=False)
        self.to_out = nn.Linear(dim_key * num_heads, dim)

        self.num_rel_pos_features = default(
            num_rel_pos_features, dim_key * num_heads
        )
        self.rel_pos_embedding = nn.Linear(
            self.num_rel_pos_features, dim_key * num_heads, bias=False
        )
        self.rel_content_bias = nn.Parameter(
            torch.randn(1, num_heads, 1, 1, dim_key)
        )
        rel_pos_bias_shape = (
            seq_len * 2 // window_size - 1 if window_bias else 1
        )
        self.rel_pos_bias = nn.Parameter(
            torch.randn(1, num_heads, rel_pos_bias_shape, 1, dim_key)
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.pos_dropout = nn.Dropout(dropout)

        self.padding_mode = padding_mode
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Sequential):
            for submodule in m.children():
                self._init_weights(submodule)
        elif isinstance(m, nn.ModuleList):
            for submodule in m:
                self._init_weights(submodule)

    def forward(self, x):
        b, n, c = x.shape
        original_n = n
        device = x.device
        remainder = n % self.window_size
        needs_padding = remainder > 0
        assert (
            n >= self.window_size
        ), f"the sequence {n} is too short for the window {self.window_size}"
        if self.padding_mode is False:
            assert needs_padding, (
                f"Sequence length ({n}) should be"
                f"divisibleby the window size ({self.window_size})."
            )
        else:
            if needs_padding:
                padding_size = self.window_size - remainder
                x = F.pad(x, (0, 0, 0, padding_size, 0, 0), value=0)
                mask = create_padding_mask(b, remainder, padding_size, device)
                n += padding_size

        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (h d k) -> k b h n d", h=self.num_heads, k=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Create sliding window indices
        window_indices = torch.arange(
            0, n - self.window_size + 1, self.window_size, device=device
        )
        q_windows = q[
            ...,
            window_indices.unsqueeze(-1)
            + torch.arange(self.window_size, device=device).unsqueeze(0),
            :,
        ]

        k_windows = k[
            ...,
            window_indices.unsqueeze(-1)
            + torch.arange(self.window_size, device=device).unsqueeze(0),
            :,
        ]

        v_windows = v[
            ...,
            window_indices.unsqueeze(-1)
            + torch.arange(self.window_size, device=device).unsqueeze(0),
            :,
        ]

        # position
        positions = get_positional_embed(
            self.window_size, self.num_rel_pos_features, device
        )
        positions = self.pos_dropout(positions)
        rel_k = self.rel_pos_embedding(positions)
        rel_k = rearrange(
            rel_k, "n (h d) -> h n d", h=self.num_heads, d=self.dim_key
        )
        # original rel_k is (h,windowSize, dimKey)
        # duplicate the rel_K for each window it should have shape
        # (h,numWindows,windowSize, dimKey)
        rel_k = rel_k.unsqueeze(1).repeat(1, q_windows.shape[2], 1, 1)

        k_windows = k_windows.transpose(-2, -1)
        content_attn = torch.matmul(
            q_windows + self.rel_content_bias,
            k_windows
        ) * (self.dim_key**-0.5)

        # calculate position attention
        rel_k = rel_k.transpose(-2, -1)

        rel_logits = torch.matmul(
            q_windows + self.rel_pos_bias,
            rel_k
            # q_windows,
            # rel_k,
        )
        # reshape position_attn to (b, h, n, w, w)
        position_attn = relative_shift_swin(rel_logits)

        attn = content_attn + position_attn
        if needs_padding:
            mask_value = -torch.finfo(attn.dtype).max
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn[:, :, -1, :, :] = attn[:, :, -1, :, :].masked_fill(
                mask, mask_value
            )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v_windows)
        out = rearrange(out, "b h w n d -> b w n (h d)")
        out = self.to_out(out)
        out = self.proj_dropout(out)

        out = rearrange(out, "b w n d -> b (w n) d")
        return out[:, :original_n]


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class SwinTransformerBlock(nn.Module):
    def __init__(self, input_resolution, dim, dim_key, window_size, shift_size, num_heads, mlp_ratio, dropout=0.2, act_layer=nn.GELU):
        super().__init__()
        self.shift_size = shift_size
        swin = WindowAttention(input_resolution, dim, dim_key=dim_key, window_size=window_size, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.attn = Residual(nn.Sequential(swin, nn.Dropout(dropout), nn.LayerNorm(dim)))
        self.fc = Residual(nn.Sequential(mlp, nn.Dropout(dropout), nn.LayerNorm(dim)))

    def forward(self, x):
        x = self.fc(self.attn(x))
        return torch.roll(x, self.shift_size, dims=1)


class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage, including upsampüling.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, dim_key, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop_rate=0., upsample=None, time_emb_dim=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.time_emb_dim = time_emb_dim

        shift_size = window_size // 2
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 dim_key=dim_key,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=-shift_size if (i % 2 == 0) else shift_size,
                                 mlp_ratio=mlp_ratio,
                                 dropout=drop_rate,
                                 )
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(dim, input_resolution)
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, 2 * self.input_resolution)
            ) if exists(time_emb_dim) else None
        else:
            self.upsample = None

    def forward(self, x, time_emb):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            scale_shift = None
            if exists(self.mlp) and exists(time_emb):
                time_emb = self.mlp(time_emb)
                #time_emb = time_emb.view(-1, 1, self.dim//2)
                time_emb = time_emb.reshape([-1, 2 * self.input_resolution, 1])
                scale_shift = time_emb.chunk(2, dim=1)
            if exists(scale_shift):
                scale, shift = scale_shift
                x = x * (scale + 1) + shift
            x = self.upsample(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage including downsampling.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, dim_key, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop_rate=0., downsample=None, time_emb_dim=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.time_emb_dim = time_emb_dim

        shift_size = window_size // 2
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 dim_key=dim_key,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=-shift_size if (i % 2 == 0) else shift_size,
                                 mlp_ratio=mlp_ratio,
                                 dropout=drop_rate,
                                 )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim, input_resolution)
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, 2 * self.input_resolution)
            ) if exists(time_emb_dim) else None
        else:
            self.downsample = None

    def forward(self, x, time_emb):
        for blk in self.blocks:
            x = blk(x)  # STB
        if self.downsample is not None:
            down = x
            scale_shift = None
            if exists(self.mlp) and exists(time_emb):
                time_emb = self.mlp(time_emb)
                time_emb = time_emb.reshape([-1, 2 * self.input_resolution, 1])
                scale_shift = time_emb.chunk(2, dim=1)
            if exists(scale_shift):
                scale, shift = scale_shift
                down = down * (scale + 1) + shift
            down = self.downsample(down)
        else:
            down = x
        return x, down  # x=skip


class SwinV2_Unet(nn.Module):
    """
    https://github.com/HuCaoFighting/Swin-Unet/blob/46ed94d9ec114feb0fb207a7a6ca327c88742369/networks/swin_transformer_unet_skip_expand_decoder_sys.py#L333
    # # Model # #
    # 1. split signal into patches
    # 2. apply positional encoding
    # 3. STB   -> Basic layer -> STB
    # 4. Patch Merging + STB
    # 5. last conv in deep feature extraction
    # 6. reconstruct signal
    """
    def __init__(self, config, in_channels, device, additional_in_channels=0):
        super(SwinV2_Unet, self).__init__()
        self.name = "swin_unet"
        self.signal_length = config['data']['seq_len']
        self.channels = in_channels
        self.embed_dim = config['model']['embed_dim']
        self.num_heads = config['model']['num_heads']       # attention heads
        self.depths = config['model']['depths']             # swin-unet depths
        self.dim_mults = config['model']['dim_mults']       # dimension multiplikators
        self.patch_size = config['model']['patchsize']
        self.mlp_ratio = config['model']['mlp_ratio']
        # self.window_size = config['model']['window_size']
        self.window_size = min(config['model']['window_size'], config['data']['seq_len']//(2**(len(self.depths)-1)))
        self.device = device
        dim_key = 64        # TODO

        drop_path_rate = config['model']['drop_path_rate']

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        norm_layer = nn.LayerNorm

        # time embeddings
        sinusoidal_pos_emb_theta = 10000
        time_dim = self.embed_dim * 4
        sinu_pos_emb = SinusoidalPosEmb(self.embed_dim, theta=sinusoidal_pos_emb_theta)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.embed_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # patch embedding
        self.patch_embed = DataEmbedding(self.channels+additional_in_channels, self.embed_dim)
        patches_resolution = self.signal_length // self.patch_size

        # layers
        dims_down = [*map(lambda m: int(self.embed_dim * m), self.dim_mults)]
        d = dims_down[::-1]
        d.append(d[-1] // 2)
        dims_concat = list(zip(d[:-1], d[1:]))[::-1]
        num_resolutions = len(dims_down)
        self.downs = nn.ModuleList([])
        for ind, (dim, depth, n_head) in enumerate(zip(dims_down, self.depths, self.num_heads)):
            if ind == len(self.dim_mults)-1:
                continue
            #is_last = ind >= (num_resolutions - 1)
            input_resolution = patches_resolution // (2 ** ind)
            #dim = int(self.embed_dim * 2 ** ind)
            self.downs.append(BasicLayer(dim, dim_key, input_resolution,
                                         depth=depth,
                                         num_heads=n_head,
                                         window_size=self.window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         downsample=PatchMerging,
                                         time_emb_dim=time_dim,
                                         drop_rate=dpr[sum(self.depths[:ind])],
                                         )
                              )
            #                             downsample=Downsample if is_last else None,))


        self.ups = nn.ModuleList([])
        self.concat_back_dim = nn.ModuleList([])
        for ind, (dim, (dimc_in, dimc_out), depth, n_head) in enumerate(zip(*map(reversed, (dims_down, dims_concat, self.depths, self.num_heads)))):
            input_resolution = int(patches_resolution // (2 ** (num_resolutions - 1 - ind)))
            self.concat_back_dim.append(nn.Linear(dimc_in, dimc_out))
            is_last = ind == (num_resolutions - 1)

            self.ups.append(BasicLayer_up(dim, dim_key, input_resolution,
                                          depth=depth,
                                          num_heads=n_head,
                                          window_size=self.window_size,
                                          mlp_ratio=self.mlp_ratio,
                                          upsample=PatchExpand if not is_last else None,
                                          time_emb_dim=time_dim,
                                          drop_rate=dpr[sum(self.depths[:(num_resolutions - 1 - ind)])],
                                     )
                          )

        #self.final_up = FinalPatchExpand(self.embed_dim, input_resolution=self.signal_length//self.patch_size)
        self.output = nn.Conv1d(self.embed_dim, self.channels, kernel_size=1, bias=False) # TODO: bias erlauben??


    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time=None, self_condition=None, condition=None):
        assert all([d % self.downsample_factor == 0 for d in
                    x.shape[-1:]]), f'input dimensions {x.shape[-1:]} need to be divisible by {self.downsample_factor}.'
        if exists(self_condition):
            assert self_condition.shape[2:] == x.shape[2:], \
                f'self_cond has the shape {self_condition.shape} which is not compatible with the input shape {x.shape}.'
            if condition is None:
                x = torch.cat((self_condition, x), dim=1)
            else:
                assert condition.shape[2:] == x.shape[
                                              2:], f'condition has the shape {condition.shape} which is not compatible with the required shape {x.shape}.'
                x = torch.cat((condition, self_condition, x), dim=1)
        elif exists(condition):
            assert condition.shape[2:] == x.shape[
                                          2:], f'condition has the shape {condition.shape} which is not compatible with the required shape {x.shape}.'
            x = torch.cat((condition, x), dim=1)

        #x = self.patch_embed(x)     # b, s_w/patch_size, embed_dim  24,32,128
        #r = x.clone()
        x = x.transpose(1,2)    # B, C, s_w -> B, s_w, C
        x = self.patch_embed(x)
        #x = x.transpose(1, 2)
        if exists(time):
            t = self.time_mlp(time)
        else:
            t = None
        h = []  # skip connections
        for idx, block in enumerate(self.downs):
            skip, x = block(x, t)
            h.append(skip)
        #h.pop()
        # x shape: 12, 32, 1024
        for idx, block in enumerate(self.ups):
            if idx != 0:
                x = torch.cat((x, h.pop()), dim=-1)
                x = self.concat_back_dim[idx-1](x)
            x = block(x, t)


        #x = torch.cat((x, r), dim=-1)
        #x = self.final_up(x)
        x = x.view(x.shape[0], self.signal_length, -1)
        x = x.permute(0, 2, 1)  # B,C,s_w
        x = self.output(x)
        return x
