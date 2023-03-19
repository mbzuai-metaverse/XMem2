"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 
from efficientnet_pytorch import EfficientNet

from model.group_modules import *
from model import resnet
from model.cbam import CBAM
from model.u2netfast import U2NET, _upsample_like

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
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 # 1/4, 64
        self.layer2 = network.layer2 # 1/8, 128
        self.layer3 = network.layer3 # 1/16, 256

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g) # 1/2, 64
        g = self.maxpool(g)  # 1/4, 64
        g = self.relu(g) 

        g = self.layer1(g) # 1/4
        g = self.layer2(g) # 1/8
        g = self.layer3(g) # 1/16

        g = g.view(batch_size, num_objects, *g.shape[1:])
        g = self.fuser(image_feat_f16, g)

        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)

        return g, h
 

class ValueEncoder_2(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False, restore_path=""):
        super().__init__()
        
        self.single_object = single_object
        #insert u2net 
        network = U2NET(in_ch=5)  
        
        self.stage1 = network.stage1
        self.pool12 = network.pool12  # see where this is happening: 1/2, 64
        self.stage2 = network.stage2

        self.pool23 = network.pool23 
        self.stage3 = network.stage3 
        self.pool34 = network.pool34 
        self.stage4 = network.stage4
        self.pool45 = network.pool45
        self.stage5 = network.stage5
        
        self.pool56 = network.pool56
        self.stage6 = network.stage6

        # decoder
        self.stage5d = network.stage5d
        self.conv1 = nn.Conv2d(512, 256, 1) 


        #end u2net 
        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)
        #insert u2net 

        hx = g

        # hx = self.pool_in(hxin)

        # stage 1
        if (hx.shape[1] == 4): 
            shape = [*hx.shape]
            shape[1] = 1 
            extra = torch.zeros(shape)
            extra = extra.to("cuda")
            hx = torch.cat((hx, extra), 1) 
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))

        hx5d_ = self.conv1(hx5d) # 1/16, 1024
        g = hx5d_
        #end u2net 

        g = g.view(batch_size, num_objects, *g.shape[1:])
        g_view = g 
        g = self.fuser(image_feat_f16, g)

        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)

        return g, h
 
class ValueEncoder_effb5(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        
        self.single_object = single_object
        model = EfficientNet.from_pretrained('efficientnet-b5', in_channels=5)
        
        self._conv_stem = model._conv_stem
        self._bn0 = model._bn0
        self.mb_blocks_0 = model._blocks[0]
        self.mb_blocks_1 = model._blocks[1]
        self.mb_blocks_2 = model._blocks[2]
        self.mb_blocks_3 = model._blocks[3]
        self.mb_blocks_4 = model._blocks[4]
        self.mb_blocks_5 = model._blocks[5]
        self.mb_blocks_6 = model._blocks[6]
        self.mb_blocks_7 = model._blocks[7]
        self.mb_blocks_8 = model._blocks[8]
        self.mb_blocks_9 = model._blocks[9]
        self.mb_blocks_10 = model._blocks[10]
        self.mb_blocks_11 = model._blocks[11]
        self.mb_blocks_12 = model._blocks[12]
        self.mb_blocks_13 = model._blocks[13]
        self.mb_blocks_14 = model._blocks[14]
        self.conv1 = nn.Conv2d(128, 256, 1) 

        # self.conv1 = network.conv1
        # self.bn1 = network.bn1
        # self.relu = network.relu  # 1/2, 64
        # self.maxpool = network.maxpool

        # self.layer1 = network.layer1 # 1/4, 64
        # self.layer2 = network.layer2 # 1/8, 128
        # self.layer3 = network.layer3 # 1/16, 256

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        # image_feat_f16 is the feature from the key encoder
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        if (g.shape[1] == 4): 
            shape = [*g.shape]
            shape[1] = 1 
            extra = torch.zeros(shape)
            extra = extra.to("cuda")
            g = torch.cat((g, extra), 1) 
        g = self._conv_stem(g)
        g = self._bn0(g)
        g = self.mb_blocks_0(g)
        g = self.mb_blocks_1(g)
        g = self.mb_blocks_2(g)
        g = self.mb_blocks_3(g)
        g = self.mb_blocks_4(g)
        g = self.mb_blocks_5(g)
        g = self.mb_blocks_6(g)
        g = self.mb_blocks_7(g)
        g = self.mb_blocks_8(g)
        g = self.mb_blocks_9(g)
        g = self.mb_blocks_10(g)
        g = self.mb_blocks_11(g)
        g = self.mb_blocks_12(g)
        g = self.mb_blocks_13(g)
        g = self.mb_blocks_14(g) # 1/16, 128 channe;s
        g = self.conv1(g) #from 128 to 256
        g = g.view(batch_size, num_objects, *g.shape[1:])
        g = self.fuser(image_feat_f16, g)

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
    
class KeyEncoder_effb7(nn.Module):
    def __init__(self):
        super().__init__()
        model = EfficientNet.from_pretrained('efficientnet-b7', in_channels=3).cuda()

        self._conv_stem = model._conv_stem
        self._bn0 = model._bn0
        self.mb_blocks_0 = model._blocks[0]
        self.mb_blocks_1 = model._blocks[1]
        self.mb_blocks_2 = model._blocks[2]
        self.mb_blocks_3 = model._blocks[3]
        self.mb_blocks_4 = model._blocks[4]
        self.mb_blocks_5 = model._blocks[5]
        self.mb_blocks_6 = model._blocks[6]
        self.mb_blocks_7 = model._blocks[7]
        self.mb_blocks_8 = model._blocks[8]
        self.mb_blocks_9 = model._blocks[9]
        self.mb_blocks_10 = model._blocks[10] #1/4, 48 channels
        self.mb_blocks_11 = model._blocks[11]
        self.mb_blocks_12 = model._blocks[12]
        self.mb_blocks_13 = model._blocks[13]
        self.mb_blocks_14 = model._blocks[14]
        self.mb_blocks_15 = model._blocks[15]
        self.mb_blocks_16 = model._blocks[16]
        self.mb_blocks_17 = model._blocks[17] #1/8, 80 channels
        self.mb_blocks_18 = model._blocks[18]
        self.mb_blocks_19 = model._blocks[19]
        self.mb_blocks_20 = model._blocks[20]
        self.mb_blocks_21 = model._blocks[21]
        self.mb_blocks_22 = model._blocks[22]
        self.mb_blocks_23 = model._blocks[23]
        self.mb_blocks_24 = model._blocks[24]
        self.mb_blocks_25 = model._blocks[25]
        self.mb_blocks_26 = model._blocks[26]
        self.mb_blocks_27 = model._blocks[27]
        self.mb_blocks_28 = model._blocks[28]
        self.mb_blocks_29 = model._blocks[29]
        self.mb_blocks_30 = model._blocks[30]
        self.mb_blocks_31 = model._blocks[31]
        self.mb_blocks_32 = model._blocks[32]
        self.mb_blocks_33 = model._blocks[33]
        self.mb_blocks_34 = model._blocks[34]
        self.mb_blocks_35 = model._blocks[35]
        self.mb_blocks_36 = model._blocks[36]
        self.mb_blocks_37 = model._blocks[37] #1/16, 224 channels
        self.conv1 = nn.Conv2d(48, 256, 1) 
        self.conv2 = nn.Conv2d(80, 512, 1) 
        self.conv3 = nn.Conv2d(224, 1024, 1) 

    def forward(self, f):
        g = self._conv_stem(f)
        g = self._bn0(g)
        g = self.mb_blocks_0(g)
        g = self.mb_blocks_1(g)
        g = self.mb_blocks_2(g)
        g = self.mb_blocks_3(g)
        g = self.mb_blocks_4(g)
        g = self.mb_blocks_5(g)
        g = self.mb_blocks_6(g)
        g = self.mb_blocks_7(g)
        g = self.mb_blocks_8(g)
        g = self.mb_blocks_9(g)
        f4 = self.mb_blocks_10(g)  #1/4, 48 channels
        g = self.mb_blocks_11(f4)
        g = self.mb_blocks_12(g)
        g = self.mb_blocks_13(g)
        g = self.mb_blocks_14(g) 
        g = self.mb_blocks_15(g)
        g = self.mb_blocks_16(g)
        f8 = self.mb_blocks_17(g) #1/8, 80 channels
        g = self.mb_blocks_18(f8)
        g = self.mb_blocks_19(g) 
        g = self.mb_blocks_20(g)
        g = self.mb_blocks_21(g)
        g = self.mb_blocks_22(g)
        g = self.mb_blocks_23(g)
        g = self.mb_blocks_24(g) 
        g = self.mb_blocks_25(g)
        g = self.mb_blocks_26(g)
        g = self.mb_blocks_27(g)
        g = self.mb_blocks_28(g)
        g = self.mb_blocks_29(g) 
        g = self.mb_blocks_30(g)
        g = self.mb_blocks_31(g)
        g = self.mb_blocks_32(g)
        g = self.mb_blocks_33(g)
        g = self.mb_blocks_34(g) 
        g = self.mb_blocks_35(g)
        g = self.mb_blocks_36(g)
        f16 = self.mb_blocks_37(g) #1/16, 224 channels
        f4 = self.conv1(f4)
        f8 = self.conv2(f8)
        f16 = self.conv3(f16)
        return f16, f8, f4
    
class KeyEncoder_2(nn.Module): #this is what ari needs to edit 
    def __init__(self, restore_path=""):
        super().__init__()
        network = U2NET()  

        self.stage1 = network.stage1
        self.pool12 = network.pool12  # see where this is happening: 1/2, 64
        self.stage2 = network.stage2

        self.pool23 = network.pool23 
        self.stage3 = network.stage3 
        self.pool34 = network.pool34 
        self.stage4 = network.stage4
        self.pool45 = network.pool45
        self.stage5 = network.stage5
        
        self.pool56 = network.pool56
        self.stage6 = network.stage6

        # decoder
        self.stage5d = network.stage5d
        self.stage4d = network.stage4d

        self.conv1 = nn.Conv2d(512, 1024, 1) # 1/16, 1024


    def forward(self, f):
        hx = f
        # hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx5d = self.conv1(hx5d) # 1/16, 1024

        # TODO: maybe return hx4d, hx3d, hx2d
        #new propsal 
        return  hx5d, hx5dup, hx4dup
    


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


class Decoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None
        
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2]

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)
        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        return hidden_state, logits
