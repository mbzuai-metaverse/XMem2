"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.group_modules import *
from model import resnet
from model.cbam import CBAM


class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]

        g = self.distributor(x, g)
        g = self.block1(g)
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        g = self.block2(g+r)

        return g


class HiddenUpdater(nn.Module):
    # Used in the decoder, multi-scale feature + GRU
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)

        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class HiddenReinforcer(nn.Module):
    # Used in the value encoder, a single GRU
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        # defined slightly differently than standard GRU, 
        # namely the new value is generated before the forget gate.
        # might provide better gradient but frankly it was initially just an 
        # implementation error that I never bothered fixing
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, stop_at=3, single_object=False):
        """
        stop_at: what backbone stage to get features from (1 -> 4, 2 -> 8, or 3 -> 16)
        """
        assert 1 <= stop_at <= 3

        super().__init__()
        
        self.stop_at = stop_at
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        if stop_at > 1:
            self.layer2 = network.layer2 # 1/8, 128
            if stop_at > 2:
                self.layer3 = network.layer3 # 1/16, 256

        self.distributor = MainToGroupDistributor()

        channels_modifiers = {
            # stop_at -> num_channels denominator
            3: 1,
            2: 2,
            1: 4
        }

        reduce_channels_factor = channels_modifiers[stop_at]
        # 1024 -> channels of features maps from key_encoder backbone
        # 256 -> same thing, but form value_encoder backbone
        x_in_dim = 1024 // reduce_channels_factor
        g_in_dim = 256 // reduce_channels_factor

        self.fuser = FeatureFusionBlock(x_in_dim, g_in_dim, value_dim, value_dim)
        if hidden_dim > 0 and stop_at == 3:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_fx, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g_1 = torch.stack([masks, others], 2)
        else:
            g_1 = masks.unsqueeze(2)
        g_2 = self.distributor(image, g_1)

        batch_size, num_objects = g_2.shape[:2]
        g = g_2.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g) # 1/2, 64
        g = self.maxpool(g)  # 1/4, 64
        g = self.relu(g) 

        g = self.layer1(g) # 1/4
        if self.stop_at > 1:
            g = self.layer2(g) # 1/8
            if self.stop_at > 2:
                g = self.layer3(g) # 1/16

        g = g.view(batch_size, num_objects, *g.shape[1:])
        g = self.fuser(image_feat_fx, g)

        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)

        return g, h
 

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f)
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g


class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x, need_s, need_e):
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection


# https://pastebin.com/QVRTR2nr for HolisticFeatures

class Decoder(nn.Module):
    def __init__(self, val_dims: list, hidden_dim):
        super().__init__()
        val_dim_f16, val_dim_f8, val_dim_f4 = val_dims

        self.fuser_f16 = FeatureFusionBlock(1024, val_dim_f16+hidden_dim, 512, 512)
        self.fuser_f8 = FeatureFusionBlock(512, val_dim_f8, 256, 256)
        self.fuser_f4 = FeatureFusionBlock(256, val_dim_f4, 128, 128)

        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None
        
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, mem_f16, mem_f8, mem_f4, h_out=True):
        """
        mem_f[x] = memory_readout from specific backbone feature stage
        """
        batch_size, num_objects = mem_f16.shape[:2]

        # TODO: add holistic memory features into the fuser
        # holistic memory features are calculated ONCE before the inference 
        # they are the same for every single call of this `forward` method
        if self.hidden_update is not None:
            # should be [1024, val_dim_f16+hidden_dim]
            g16 = self.fuser_f16(f16, torch.cat([mem_f16, hidden_state], 2))
        else:
            g16 = self.fuser_f16(f16, mem_f16)

        # should be [512, val_dim_f8]
        g8 = self.fuser_f8(f8, mem_f8)
        g8_mixed = self.up_16_8(g8, g16)

        # should be [256, val_dim_f4]
        g4 = self.fuser_f4(f4, mem_f4)
        g4_mixed = self.up_8_4(g4, g8_mixed)

        logits = self.pred(F.relu(g4_mixed.flatten(start_dim=0, end_dim=1)))

        if h_out and self.hidden_update is not None:
            g4_mixed = torch.cat([g4_mixed, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8_mixed, g4_mixed], hidden_state)
        else:
            hidden_state = None
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, logits