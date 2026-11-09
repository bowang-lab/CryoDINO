# Author: Tony Xu
#
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

# Edited by Ahmadreza Attarpour for CryoDINO
# Adapted for Detection task.

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.backbone_fpn_utils import BackboneWithFPN
from monai.networks.blocks.feature_pyramid_network import FeaturePyramidNetwork
from monai.networks.nets import resnet34
from ..segmentation_3d.vit_adapter import ViTAdapter


class UNETRHead(nn.Module):
    def __init__(self, feature_model, input_channels, image_size, num_classes, autocast_ctx, deep_supervision=False):
        super().__init__()

        self.autocast_ctx = autocast_ctx
        self.input_channels = input_channels
        self.deep_supervision = deep_supervision
        self.feature_model = feature_model
        self.hidden_size = self.feature_model.num_features
        self.feature_size = 32

        self.patch_size = self.feature_model.patch_embed.patch_size
        self.feat_size = [image_size // p for p in self.patch_size]

        # merges multi-channel input into a single vector output for each stage
        self.channel_merge1 = nn.Linear(self.hidden_size * self.input_channels, self.hidden_size)
        self.channel_merge2 = nn.Linear(self.hidden_size * self.input_channels, self.hidden_size)
        self.channel_merge3 = nn.Linear(self.hidden_size * self.input_channels, self.hidden_size)
        self.channel_merge4 = nn.Linear(self.hidden_size * self.input_channels, self.hidden_size)
        self.act_fn = nn.GELU()

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=input_channels,
            out_channels=self.feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=self.hidden_size,
            out_channels=self.feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=self.hidden_size,
            out_channels=self.feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=self.hidden_size,
            out_channels=self.feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=True,
            res_block=True,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.hidden_size,
            out_channels=self.feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_size * 8,
            out_channels=self.feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=self.feature_size * 4,
            out_channels=self.feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        # AA: decoder2 (H/2 → H) and out removed — detection stops at H/2 (dec1)
        # self.decoder2 = UnetrUpBlock(
        #     spatial_dims=3,
        #     in_channels=self.feature_size * 2,
        #     out_channels=self.feature_size,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name='instance',
        #     res_block=True,
        # )
        # self.out = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=num_classes)

        # AA: deep supervision removed
        # if deep_supervision:
        #     self.out_ds1 = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size*2, out_channels=num_classes)  # H/2
        #     self.out_ds2 = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size*4, out_channels=num_classes)  # H/4
        #     self.out_ds3 = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size*8, out_channels=num_classes)  # H/8

        # AA: detection heads — tapped at dec1 (H/2, feature_size*2 channels)
        # AA: two 3×3 conv stems (spatial context) before final 1×1 prediction heads
        _ch = self.feature_size * 2
        self.cls_stem = nn.Sequential(
            nn.Conv3d(_ch, _ch, kernel_size=3, padding=1), nn.SiLU(inplace=True), nn.InstanceNorm3d(_ch),
            nn.Conv3d(_ch, _ch, kernel_size=3, padding=1), nn.SiLU(inplace=True), nn.InstanceNorm3d(_ch),
        )
        self.cls_head = nn.Conv3d(_ch, num_classes, kernel_size=1)
        self.off_stem = nn.Sequential(
            nn.Conv3d(_ch, _ch, kernel_size=3, padding=1), nn.SiLU(inplace=True), nn.InstanceNorm3d(_ch),
            nn.Conv3d(_ch, _ch, kernel_size=3, padding=1), nn.SiLU(inplace=True), nn.InstanceNorm3d(_ch),
        )
        self.off_head = nn.Conv3d(_ch, 3, kernel_size=1)
        # AA: init: cls biased toward background; offsets start at zero
        nn.init.zeros_(self.cls_head.weight); nn.init.constant_(self.cls_head.bias, -4)
        nn.init.zeros_(self.off_head.weight); nn.init.zeros_(self.off_head.bias)

        self.proj_axes = (0, 3 + 1) + tuple(d + 1 for d in range(3))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward_features_multi(self, x_in):
        """Pass multi-channel input through feature model by reshaping batch and merging channels"""
        assert x_in.shape[1] == self.input_channels
        B = x_in.shape[0]

        # Change feature channel into individual batches B, C, H, W, D -> B*C, 1, H, W, D
        x_reshape = x_in.reshape(-1, 1, *x_in.shape[2:])
        with self.autocast_ctx():
            x2, x3, x4, x = self.feature_model.get_intermediate_layers(
                x_reshape,
                n=[5, 11, 17, 23],
                return_class_token=False
            )

        # reshape to merge channels B*C, N, F -> B, N, C*F
        x2 = x2.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)
        x3 = x3.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)
        x4 = x4.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)
        x = x.permute(0, 2, 1).reshape(B, self.hidden_size*self.input_channels, -1).permute(0, 2, 1)

        # Merge channels B, N, C*F -> B, N, F
        x2 = self.act_fn(self.channel_merge1(x2))
        x3 = self.act_fn(self.channel_merge2(x3))
        x4 = self.act_fn(self.channel_merge3(x4))
        x = self.act_fn(self.channel_merge4(x))

        return x2, x3, x4, x

    def forward(self, x_in):

        x2, x3, x4, x = self.forward_features_multi(x_in)
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.proj_feat(x2))
        enc3 = self.encoder3(self.proj_feat(x3))
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)  # H/8, 8F
        dec2 = self.decoder4(dec3, enc3)  # H/4, 4F
        dec1 = self.decoder3(dec2, enc2)  # H/2, 2F  ← stride-2 feature map
        # AA: out = self.decoder2(dec1, enc1)  # H, F  ← removed, detection stops here

        # AA: if self.deep_supervision and self.training:
        # AA:     return [self.out(out), self.out_ds1(dec1), self.out_ds2(dec2), self.out_ds3(dec3)]
        # AA: return self.out(out)
        return self.cls_head(self.cls_stem(dec1)), self.off_head(self.off_stem(dec1)).tanh() * 2


class LinearDecoderHead(nn.Module):
    def __init__(self, feature_model, input_channels, image_size, num_classes, autocast_ctx, n_last_layers=4):
        super().__init__()

        self.autocast_ctx = autocast_ctx
        self.input_channels = input_channels

        self.feature_model = feature_model
        self.n_last_layers = n_last_layers
        self.image_size = image_size

        self.hidden_size = self.feature_model.num_features

        # merges multi-channel input into a single vector output
        self.channel_merge = nn.Conv3d(self.hidden_size * self.input_channels, self.hidden_size, kernel_size=1)
        self.act_fn = nn.GELU()

        self.bn_channels = self.hidden_size * n_last_layers
        self.bn = nn.BatchNorm3d(self.bn_channels)
        # AA: conv_seg → two parallel heads; resize to H/2 (stride-2) instead of H
        # self.conv_seg = nn.Conv3d(self.bn_channels, num_classes, kernel_size=1)
        # self.resize = nn.Upsample(size=(self.image_size,) * 3, mode="trilinear")
        self.cls_head = nn.Conv3d(self.bn_channels, num_classes, kernel_size=1)
        self.off_head = nn.Conv3d(self.bn_channels, 3, kernel_size=1)
        self.resize = nn.Upsample(size=(self.image_size // 2,) * 3, mode="trilinear")

    def forward_features_multi(self, inputs):
        """Pass multi-channel input through feature model one-by-one and concatenate + merge outputs."""

        assert inputs.shape[1] == self.input_channels
        B = inputs.shape[0]

        # Change feature channel into individual batches B, C, H, W, D -> B*C, 1, H, W, D
        inputs_reshape = inputs.reshape(-1, 1, *inputs.shape[2:])

        with self.autocast_ctx():
            features = self.feature_model.get_intermediate_layers(
                inputs_reshape,
                n=self.n_last_layers,
                return_class_token=False,
                reshape=True
            )

        # reshape to merge channels, B*C, F, H, W, D -> B, C*F, H, W, D
        features = [f.reshape(B, self.hidden_size*self.input_channels, *f.shape[2:]) for f in features]

        # Merge channels B, C*F, H, W, D -> B, C, H, W, D
        merged_feats = [self.act_fn(self.channel_merge(f)) for f in features]
        return merged_feats

    def forward(self, inputs):
        """Forward function."""
        features = self.forward_features_multi(inputs)
        cat_feats = torch.cat(features, dim=1)
        cat_feats = self.bn(cat_feats)
        # AA: logits = self.conv_seg(cat_feats); return self.resize(logits)
        return self.resize(self.cls_head(cat_feats)), self.resize(self.off_head(cat_feats))


def _add_detection_heads(module, ch, num_classes):
    """Attach the dense stride-2 detection head (cls + offset branches) to `module`.

    Shared by ViTAdapterUNETRHead and ResNetFPNHead so both end in exactly the same head;
    attribute names are fixed (cls_stem/cls_head/off_stem/off_head) so state-dict keys
    of existing checkpoints are unchanged.
    """
    # AA: two 3×3 conv stems (spatial context) before final 1×1 prediction heads
    module.cls_stem = nn.Sequential(
        nn.Conv3d(ch, ch, kernel_size=3, padding=1), nn.SiLU(inplace=True), nn.InstanceNorm3d(ch),
        nn.Conv3d(ch, ch, kernel_size=3, padding=1), nn.SiLU(inplace=True), nn.InstanceNorm3d(ch),
    )
    module.cls_head = nn.Conv3d(ch, num_classes, kernel_size=1)  # raw logits
    module.off_stem = nn.Sequential(
        nn.Conv3d(ch, ch, kernel_size=3, padding=1), nn.SiLU(inplace=True), nn.InstanceNorm3d(ch),
        nn.Conv3d(ch, ch, kernel_size=3, padding=1), nn.SiLU(inplace=True), nn.InstanceNorm3d(ch),
    )
    module.off_head = nn.Conv3d(ch, 3, kernel_size=1)  # (Δx, Δy, Δz) — channel 0 = first axis (X), matches anchors_for_offsets_feature_map
    # AA: init: cls biased toward background; offsets start at zero
    nn.init.zeros_(module.cls_head.weight); nn.init.constant_(module.cls_head.bias, -4)
    nn.init.zeros_(module.off_head.weight); nn.init.zeros_(module.off_head.bias)


def _detection_outputs(module, feat):
    """feat [B, ch, D/2, H/2, W/2] -> (cls_map [B, C, ...] logits, off_map [B, 3, ...] in [-2, 2])."""
    cls_map = module.cls_head(module.cls_stem(feat))
    off_map = module.off_head(module.off_stem(feat)).tanh() * 2
    return cls_map, off_map


def _pretrain_size_from(vit_model, fallback):
    """Side length whose //16 grid matches the ViT's pos_embed token count.

    ViTAdapter reshapes pos_embed to (pretrain_size // 16)^3; if that disagrees with the
    checkpoint the reshape raises, so infer it from the weights rather than guessing.
    """
    try:
        n_tokens = vit_model.pos_embed.shape[1] - 1          # drop the cls token
        patch = vit_model.patch_embed.patch_size[0]
        grid = round(n_tokens ** (1.0 / 3.0))
        if grid ** 3 == n_tokens:
            return grid * patch
    except (AttributeError, IndexError, TypeError):
        pass
    return fallback


class ViTAdapterUNETRHead(nn.Module):

    # AA: deep_supervision removed; num_classes = number of particle classes (e.g. 6)
    def __init__(self, feature_model, input_channels, image_size, num_classes, autocast_ctx):
        super().__init__()

        self.autocast_ctx = autocast_ctx
        self.input_channels = input_channels
        # self.deep_supervision = deep_supervision  # AA: removed
        # AA: ViTAdapter._get_pos_embed reshapes pos_embed to (pretrain_size // 16)^3, so
        # pretrain_size must describe the CHECKPOINT's pos_embed grid — not the finetune
        # image size. The default 112 gives 7^3 = 343 and blows up on a 128-crop checkpoint
        # (8^3 = 512 tokens). segmentation_heads.py:243 (commit ef2cdfd) passes image_size,
        # which only works when the two happen to coincide; derive it from the weights instead.
        self.feature_model = ViTAdapter(feature_model, input_channels,
                                        pretrain_size=_pretrain_size_from(feature_model, image_size))
        self.hidden_size = self.feature_model.vit_model.num_features
        self.feature_size = 32
        self.patch_size = self.feature_model.vit_model.patch_embed.patch_size
        self.feat_size = [image_size // p for p in self.patch_size]

        self.act_fn = nn.GELU()

        self.encoder1 = UnetrBasicBlock(spatial_dims=3, in_channels=input_channels, out_channels=self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.encoder2 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=self.feature_size,
                                        kernel_size=3,  stride=1, norm_name='instance', res_block=True)
        self.encoder3 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=2*self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.encoder4 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=4*self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.encoder5 = UnetrBasicBlock(spatial_dims=3, in_channels=self.hidden_size, out_channels=8*self.feature_size,
                                        kernel_size=3, stride=1, norm_name='instance', res_block=True)
        self.decoder4 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size*8, out_channels=self.feature_size*4,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        self.decoder3 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size*4, out_channels=self.feature_size*2,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        self.decoder2 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size*2, out_channels=self.feature_size,
                                     kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        # AA: decoder1 (H/2 → H) and out removed — detection stops at dec0 (H/2)
        # self.decoder1 = UnetrUpBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=self.feature_size,
        #                              kernel_size=3, upsample_kernel_size=2, norm_name='instance', res_block=True)
        # self.out = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size, out_channels=num_classes)

        # AA: deep supervision removed
        # if deep_supervision:
        #     self.out_ds1 = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size,   out_channels=num_classes)  # H/2
        #     self.out_ds2 = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size*2, out_channels=num_classes)  # H/4
        #     self.out_ds3 = UnetOutBlock(spatial_dims=3, in_channels=self.feature_size*4, out_channels=num_classes)  # H/8

        # AA: detection heads tapped at dec0 (H/2, feature_size channels)
        _add_detection_heads(self, self.feature_size, num_classes)

    def forward(self, x_in):

        f1, f2, f3, f4 = self.feature_model(x_in)
        # AA: enc0 not used — decoder1 (which needed it as skip) is removed
        # enc0 = self.encoder1(x_in)  # H, W, D, F
        enc1 = self.encoder2(f1)  # H/2, W/2, D/2, F
        enc2 = self.encoder3(f2)  # H/4, W/4, D/4, 2F
        enc3 = self.encoder4(f3)  # H/8, W/8, D/8, 4F
        enc4 = self.encoder5(f4)  # H/16, W/16, D/16, 8F

        dec2 = self.decoder4(enc4, enc3)  # H/8, W/8, D/8, 4F
        dec1 = self.decoder3(dec2, enc2)  # H/4, W/4, D/4, 2F
        dec0 = self.decoder2(dec1, enc1)  # H/2, W/2, D/2, F  ← stride-2 feature map
        # AA: out = self.decoder1(dec0, enc0)  # H, W, D, F  ← removed

        # AA: if self.deep_supervision and self.training:
        # AA:     return [self.out(out), self.out_ds1(dec0), self.out_ds2(dec1), self.out_ds3(dec2)]
        # AA: return self.out(out)
        return _detection_outputs(self, dec0)                      # [B, C, D/2, H/2, W/2], [B, 3, ...]


class ResNetFPNHead(nn.Module):
    """MONAI ResNet34-FPN backbone (trained from scratch) + the same dense detection head.

    Same-loss CNN baseline for ViTAdapterUNETRHead: identical cls/offset head at stride 2,
    trained by the same object_detection_loss and decoded/evaluated by the same code, so
    the backbone is the only difference. conv1 stride 2 with no max-pool puts layer1 at
    stride 2 (layer2..4 at 4, 8, 16); the FPN fuses them top-down and its stride-2 level
    feeds the head with feature_size=32 channels, as ViTAdapterUNETR's dec0 does.
    """

    def __init__(self, input_channels, image_size, num_classes, autocast_ctx, feature_size=32):  # noqa: ARG002
        super().__init__()
        self.autocast_ctx = autocast_ctx
        backbone = resnet34(spatial_dims=3, n_input_channels=input_channels,
                            conv1_t_stride=2, no_max_pool=True, pretrained=False)
        # BackboneWithFPN's IntermediateLayerGetter runs the backbone's children in order and
        # ignores no_max_pool, so the max-pool must be removed explicitly — otherwise the
        # head lands at stride 4 while the loss/decoder assume stride 2.
        backbone.maxpool = nn.Identity()
        self.feature_model = BackboneWithFPN(
            backbone,
            return_layers={'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'},
            in_channels_list=[64, 128, 256, 512],
            out_channels=feature_size,
            spatial_dims=3,
        )
        _add_detection_heads(self, feature_size, num_classes)

    def forward(self, x_in):
        with self.autocast_ctx():
            feat = self.feature_model(x_in)['0']                   # stride-2 FPN level
            # loss/decoder hard-code stride 2; a mismatch would silently misplace every detection
            assert tuple(feat.shape[-3:]) == tuple(s // 2 for s in x_in.shape[-3:]), \
                f"ResNetFPN feature map {tuple(feat.shape[-3:])} is not stride 2 of input {tuple(x_in.shape[-3:])}"
            cls_map, off_map = _detection_outputs(self, feat)
        return cls_map.float(), off_map.float()                    # loss/decode in fp32



# ---------------------------------------------------------------------------------------------
# Opt-in plain-ViT multi-block decoders (detection3d.py --decoder-fusion concat|fpn).
# The default (--decoder-fusion adapter) still builds ViTAdapterUNETRHead above, unchanged.
# ---------------------------------------------------------------------------------------------

DECODER_FUSIONS = ('adapter', 'concat', 'fpn', 'adapter_multi', 'adapter_fpn')
# adapter_multi: second-to-last block of each adapter stage; with the stage-end taps (5,11,17,23)
# the adapter already uses, every stage then sees its last two blocks
DEFAULT_ENCODER_LAYERS = {'concat': '-2,-1', 'fpn': '5,11,17,23', 'adapter_multi': '4,10,16,22'}
# ResNet34 layer1..4 widths at strides 2/4/8/16 — the exact FPN inputs of ResNetFPNHead
_FPN_IN_CHANNELS = [64, 128, 256, 512]


def _vit_depth(vit):
    """Number of real transformer blocks (BlockChunk pads chunks with nn.Identity)."""
    if getattr(vit, 'chunked_blocks', False):
        return sum(1 for chunk in vit.blocks for b in chunk if not isinstance(b, nn.Identity))
    return len(vit.blocks)


def parse_encoder_layers(spec, depth):
    """'5,11,17,23' / '-2,-1' -> sorted unique non-negative block indices (0-based, negatives from the end).

    get_intermediate_layers() only accepts non-negative indices when given a list (it tests
    `i in blocks_to_take`), so negatives are resolved here.
    """
    idx = []
    for tok in str(spec).replace(' ', '').split(','):
        if tok == '':
            continue
        i = int(tok)
        j = i + depth if i < 0 else i
        if not 0 <= j < depth:
            raise ValueError(f"encoder layer {i} out of range for a {depth}-block ViT")
        idx.append(j)
    idx = sorted(set(idx))
    if not idx:
        raise ValueError(f"empty --encoder-layers: {spec!r}")
    return idx


class _ViTHolder(nn.Module):
    """Holds the ViT as `.vit_model` so parameter names (feature_model.vit_model.*) and the
    freezing / train-last-n-blocks code in detection3d.py are the same as for ViTAdapterUNETRHead."""

    def __init__(self, vit_model):
        super().__init__()
        self.vit_model = vit_model


def _pyramid_branch(in_ch, out_ch, n_up):
    """1x1 projection of a stride-16 ViT map, then n_up x (2x deconv) -> stride 16 / 2**n_up (ViTDet simple pyramid)."""
    layers = [nn.Conv3d(in_ch, out_ch, kernel_size=1), nn.InstanceNorm3d(out_ch), nn.GELU()]
    for _ in range(n_up):
        layers += [nn.ConvTranspose3d(out_ch, out_ch, kernel_size=2, stride=2), nn.InstanceNorm3d(out_ch), nn.GELU()]
    return nn.Sequential(*layers)


class ViTFPNHead(nn.Module):
    """Plain ViT (no adapter / spatial-prior module) -> 4-level simple pyramid -> FPN -> detection head.

    Mirrors ResNetFPNHead: the pyramid levels have ResNet34's C2..C5 widths (64/128/256/512) at
    strides 2/4/8/16 and go through the same MONAI FeaturePyramidNetwork (1x1 lateral, nearest
    top-down add, 3x3 output, 32 channels); the stride-2 level P2 feeds the shared cls/offset head.

    fusion='fpn'   : exactly 4 encoder blocks, shallow -> deep mapped to strides 2, 4, 8, 16
                     (UNETR / ViT-Adapter style multi-block taps; default 5,11,17,23).
    fusion='concat': any number of blocks concatenated along channels into one stride-16 map,
                     from which all four levels are built (ViTDet simple pyramid; default -2,-1).
    Token maps are taken through get_intermediate_layers(norm=True), i.e. after the ViT's final
    LayerNorm, as in UNETRHead / LinearDecoderHead.
    """

    def __init__(self, feature_model, input_channels, image_size, num_classes, autocast_ctx,
                 fusion='fpn', encoder_layers=None, feature_size=32):  # noqa: ARG002 (image_size)
        super().__init__()
        if fusion not in ('concat', 'fpn'):
            raise ValueError(f"ViTFPNHead fusion must be concat|fpn, got {fusion!r}")
        self.autocast_ctx = autocast_ctx
        self.input_channels = input_channels
        self.fusion = fusion
        self.feature_model = _ViTHolder(feature_model)
        vit = self.feature_model.vit_model
        self.hidden_size = vit.num_features
        self.patch_size = vit.patch_embed.patch_size
        self.encoder_layers = parse_encoder_layers(
            encoder_layers if encoder_layers is not None else DEFAULT_ENCODER_LAYERS[fusion], _vit_depth(vit))

        n_levels = len(_FPN_IN_CHANNELS)                        # strides 2, 4, 8, 16
        tap_ch = self.hidden_size * input_channels
        if fusion == 'fpn':
            if len(self.encoder_layers) != n_levels:
                raise ValueError(f"--decoder-fusion fpn needs exactly {n_levels} encoder layers "
                                 f"(one per stride 2/4/8/16), got {self.encoder_layers}")
            in_chs = [tap_ch] * n_levels
        else:
            in_chs = [tap_ch * len(self.encoder_layers)] * n_levels
        self.pyramid = nn.ModuleList([
            _pyramid_branch(in_chs[k], _FPN_IN_CHANNELS[k], n_up=n_levels - 1 - k) for k in range(n_levels)
        ])
        self.fpn = FeaturePyramidNetwork(spatial_dims=3, in_channels_list=_FPN_IN_CHANNELS, out_channels=feature_size)
        _add_detection_heads(self, feature_size, num_classes)

    def _vit_taps(self, x_in):
        """[B, C, X, Y, Z] -> list of [B, C*F, X/16, Y/16, Z/16] (fp32), one per encoder layer."""
        B, C = x_in.shape[:2]
        grid = [s // p for s, p in zip(x_in.shape[2:], self.patch_size)]
        with self.autocast_ctx():
            taps = self.feature_model.vit_model.get_intermediate_layers(
                x_in.reshape(B * C, 1, *x_in.shape[2:]), n=self.encoder_layers, return_class_token=False)
        # B*C, N, F -> B, C*F, X, Y, Z  (same channel merge layout as UNETRHead)
        return [t.float().permute(0, 2, 1).reshape(B, C * self.hidden_size, *grid) for t in taps]

    def forward(self, x_in):
        taps = self._vit_taps(x_in)
        if self.fusion == 'fpn':
            srcs = taps                                         # shallow -> deep = stride 2 -> 16
        else:
            cat = torch.cat(taps, dim=1)
            srcs = [cat] * len(self.pyramid)
        levels = OrderedDict((str(k), branch(s)) for k, (branch, s) in enumerate(zip(self.pyramid, srcs)))
        feat = self.fpn(levels)['0']                            # stride-2 FPN level (P2), as ResNetFPNHead
        assert tuple(feat.shape[-3:]) == tuple(s // 2 for s in x_in.shape[-3:]), \
            f"ViTFPN feature map {tuple(feat.shape[-3:])} is not stride 2 of input {tuple(x_in.shape[-3:])}"
        return _detection_outputs(self, feat)


def build_vit_detection_head(feature_model, input_channels, image_size, num_classes, autocast_ctx,
                             fusion='adapter', encoder_layers=None):
    """--segmentation-head ViTAdapterUNETR factory. fusion='adapter' (default) is the original
    ViTAdapterUNETRHead, bit-for-bit; its block taps are fixed by the adapter's 4 interaction
    stages (blocks 0-5, 6-11, 12-17, 18-23), so --encoder-layers is rejected there."""
    if fusion == 'adapter':
        if encoder_layers is not None:
            raise ValueError("--encoder-layers needs --decoder-fusion concat|fpn|adapter_multi; the ViT-Adapter "
                             "already taps the ends of its 4 interaction stages (blocks 5, 11, 17, 23)")
        return ViTAdapterUNETRHead(feature_model, input_channels, image_size, num_classes, autocast_ctx)
    if fusion in ('adapter_multi', 'adapter_fpn'):
        return ViTAdapterUNETRPlusHead(feature_model, input_channels, image_size, num_classes, autocast_ctx,
                                       fusion=fusion, encoder_layers=encoder_layers)
    return ViTFPNHead(feature_model, input_channels, image_size, num_classes, autocast_ctx,
                      fusion=fusion, encoder_layers=encoder_layers)


# ---------------------------------------------------------------------------------------------
# Opt-in ViT-Adapter variants that keep ViTAdapterUNETRHead intact (detection3d.py
# --decoder-fusion adapter_multi|adapter_fpn). All original modules are built first, by the
# parent constructor, so their initialisation is the same as the default head's. Every added
# path ends in a zero-initialised conv and is added residually, so at init the forward equals
# the original head; the added paths are learned from there.
# ---------------------------------------------------------------------------------------------


def _zero_conv(in_ch, out_ch, kernel_size=1):
    conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class _TopDownResidual(nn.Module):
    """FPN-style lateral + top-down fusion over the UNETR skip features enc1..enc4 (strides 2..16,
    channels F, 2F, 4F, 8F), returned as residual corrections enc_k + out_k(p_k).

    Same structure as MONAI's FeaturePyramidNetwork in ResNetFPNHead (1x1 lateral to a common
    width, nearest-upsampled top-down sum, 3x3 output conv), but the 3x3 output conv maps back to
    each level's own width and is zero-initialised, so the UNETR decoder receives its original
    inputs at init.
    """

    def __init__(self, in_channels_list, width):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv3d(c, width, kernel_size=1) for c in in_channels_list])
        self.output = nn.ModuleList([_zero_conv(width, c, kernel_size=3) for c in in_channels_list])

    def forward(self, feats):
        p = self.lateral[-1](feats[-1])
        outs = [feats[-1] + self.output[-1](p)]
        for k in range(len(feats) - 2, -1, -1):
            lat = self.lateral[k](feats[k])
            p = lat + F.interpolate(p, size=lat.shape[2:], mode='nearest')
            outs.insert(0, feats[k] + self.output[k](p))
        return outs


class ViTAdapterUNETRPlusHead(ViTAdapterUNETRHead):
    """ViTAdapterUNETRHead (adapter, SPM, interaction stages, UNETR decoder, detection head all
    unchanged) plus one zero-initialised residual path:

    adapter_multi: extra ViT block outputs (--encoder-layers, default 4,10,16,22) are captured with
        forward hooks while the adapter runs. Each is assigned to the interaction stage containing
        it; per stage they are concatenated, merged by a zero-init 1x1 conv, upsampled like that
        stage's own ViT tap (x8/x4/x2/x1) and added to the adapter output f1..f4.
    adapter_fpn: _TopDownResidual over enc1..enc4 (the UNETR decoder's skip inputs) at width 32,
        like the ResNet34-FPN baseline, so deep context reaches every skip directly.
    """

    def __init__(self, feature_model, input_channels, image_size, num_classes, autocast_ctx,
                 fusion='adapter_multi', encoder_layers=None):
        super().__init__(feature_model, input_channels, image_size, num_classes, autocast_ctx)
        if fusion not in ('adapter_multi', 'adapter_fpn'):
            raise ValueError(f"ViTAdapterUNETRPlusHead fusion must be adapter_multi|adapter_fpn, got {fusion!r}")
        self.fusion = fusion
        adapter = self.feature_model
        vit = adapter.vit_model
        self.encoder_layers = []
        if fusion == 'adapter_fpn':
            if encoder_layers is not None:
                raise ValueError("--decoder-fusion adapter_fpn takes no --encoder-layers")
            fs = self.feature_size
            self.topdown = _TopDownResidual([fs, 2 * fs, 4 * fs, 8 * fs], width=fs)
            return

        self.encoder_layers = parse_encoder_layers(
            encoder_layers if encoder_layers is not None else DEFAULT_ENCODER_LAYERS['adapter_multi'], _vit_depth(vit))
        blocks = [b for chunk in vit.blocks for b in chunk if not isinstance(b, nn.Identity)] \
            if vit.chunked_blocks else list(vit.blocks)
        # stage of a block = adapter.indexes entry containing it (chunk index when chunked)
        self.stage_layers = [[b for b in self.encoder_layers if st <= b <= ed] for st, ed in adapter.indexes]
        dim = vit.embed_dim
        self.extra_merge = nn.ModuleList([
            _zero_conv(len(ls) * dim * input_channels, dim) if ls else nn.Identity() for ls in self.stage_layers
        ])
        self._taps = {}
        self._hooks = [blocks[b].register_forward_hook(self._make_hook(b)) for b in self.encoder_layers]

    def _make_hook(self, b):
        def hook(module, inputs, output):
            self._taps[b] = output[:, 1:]       # inside the adapter the cls token is prepended: drop it
        return hook

    def _adapter_features(self, x_in):
        if self.fusion != 'adapter_multi':
            return self.feature_model(x_in)
        self._taps = {}
        f = list(self.feature_model(x_in))
        taps, self._taps = self._taps, {}
        B, C = x_in.shape[:2]
        grid = [s // p for s, p in zip(x_in.shape[2:], self.patch_size)]
        n_stages = len(self.stage_layers)
        for k, layers in enumerate(self.stage_layers):
            if not layers:
                continue
            # B*C, N, dim -> B, C*dim, X, Y, Z per tap (same layout as ViTAdapter's channel_merge), concat taps
            t = torch.cat([taps[b].transpose(1, 2).reshape(B, C * self.hidden_size, *grid) for b in layers], dim=1)
            t = self.extra_merge[k](t)
            scale = 2 ** (n_stages - 1 - k)     # stage k feeds level k: stride 16 -> 2, 4, 8, 16
            if scale > 1:
                t = F.interpolate(t, scale_factor=scale, mode='trilinear', align_corners=False)
            f[k] = f[k] + t
        return f

    def forward(self, x_in):
        f1, f2, f3, f4 = self._adapter_features(x_in)
        enc1 = self.encoder2(f1)  # H/2, F
        enc2 = self.encoder3(f2)  # H/4, 2F
        enc3 = self.encoder4(f3)  # H/8, 4F
        enc4 = self.encoder5(f4)  # H/16, 8F
        if self.fusion == 'adapter_fpn':
            enc1, enc2, enc3, enc4 = self.topdown([enc1, enc2, enc3, enc4])
        dec2 = self.decoder4(enc4, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)  # H/2, F  <- stride-2 feature map
        return _detection_outputs(self, dec0)

