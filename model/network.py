"""
This file defines XMem, the highest level nn.Module interface
During training, it is used by trainer.py
During evaluation, it is used by inference_core.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

import torch
import torch.nn as nn

from model.aggregate import aggregate
from model.modules import *
from model.memory_util import *
from util.tensor_util import KeyFeatures
from util.tensor_util import pad_divide_by


class XMem(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)

        self.single_object = config.get('single_object', False)
        # print(f'Single object mode: {self.single_object}')

        # backbone
        self.key_encoder = KeyEncoder()

        # Value encoders (3 different memories -> 3 different sizes of values)
        self.value_encoder_f16 = ValueEncoder(self.value_dim_f16, self.hidden_dim, stop_at=3, single_object=self.single_object)
        self.value_encoder_f8 = ValueEncoder(self.value_dim_f8, self.hidden_dim, stop_at=2, single_object=self.single_object)
        self.value_encoder_f4 = ValueEncoder(self.value_dim_f4, self.hidden_dim, stop_at=1, single_object=self.single_object)

        # Projection from fx feature space to key/value space
        self.key_proj_f16 = KeyProjection(1024, self.key_dim_f16)
        self.key_proj_f8 = KeyProjection(512, self.key_dim_f8)
        self.key_proj_f4 = KeyProjection(256, self.key_dim_f4)

        self.decoder = Decoder([self.value_dim_f16, self.value_dim_f8, self.value_dim_f4], self.hidden_dim)

        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)

    @staticmethod
    def _reshape(b, t, key, shrinkage=None, selection=None):
        key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
        if shrinkage is not None:
            shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
        if selection is not None:
            selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

        return key, shrinkage, selection

    def encode_key(self, frame, need_sk=True, need_ek=True) -> KeyFeatures: 
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
    
        f16, f8, f4 = self.key_encoder(frame)
        key_f16, shrinkage_f16, selection_f16 = self.key_proj_f16(f16, need_sk, need_ek)
        key_f8, shrinkage_f8, selection_f8 = self.key_proj_f8(f8, need_sk, need_ek)
        key_f4, shrinkage_f4, selection_f4 = self.key_proj_f4(f4, need_sk, need_ek)

        if need_reshape:
            key_f16, shrinkage_f16, selection_f16 = self._reshape(b, t, key_f16, shrinkage_f16, selection_f16)
            key_f8, shrinkage_f8, selection_f8 = self._reshape(b, t, key_f8, shrinkage_f8, selection_f8)
            key_f4, shrinkage_f4, selection_f4 = self._reshape(b, t, key_f4, shrinkage_f4, selection_f4)

            # B*T*C*H*W
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])

        res = KeyFeatures(
            f16=f16,
            f8=f8,
            f4=f4,

            key_f16=key_f16,
            key_f8=key_f8,
            key_f4=key_f4,

            shrinkage_f16=shrinkage_f16,
            shrinkage_f8=shrinkage_f8,
            shrinkage_f4=shrinkage_f4,

            selection_f16=selection_f16,
            selection_f8=selection_f8,
            selection_f4=selection_f4,
        )

        return res

    def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True): 
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i!=j]]
                , dim=1, keepdim=True)
            for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        g16, h16 = self.value_encoder_f16(frame, image_feat_f16, h16, masks, others, is_deep_update)

        # TODO: return multiscale
        return g16, h16

    def encode_holistic_features(self, frames: list, masks: list):
        """Takes a list of frames + masks and -> a block of same shape as `value` features, Nx more channels (N=4 by default)

        Runs a recurrent unit with memory 
        Args:
            frames (list): List of torch.Tensor objects containing RGB images
            masks (list): List of torch.Tensor objects containing binary GT masks
        """
        assert len(frames) > 0, "No frames provided for holistic features extraction!"
        assert len(frames) == len(masks), f"Equal number of frames and masks should be provided, got {len(frames)} and {len(masks)}"
        
        holistic_features = None
        for image, mask in zip(frames, masks):
            image, _ = pad_divide_by(image, 16)
            image = image.unsqueeze(0) # add the batch dimension
                
            mask, _ = pad_divide_by(mask, 16)
            mask = aggregate(mask, dim=0)
            mask = mask[1:].unsqueeze(0)
            
            assert len(image.shape) == 4
            key, shrinkage, selection, f16, f8, f4 = self.encode_key(image, need_ek=False, need_sk=False)
            
            # hidden state is ignored if is_deep_update==False
            # value.shape = [B, num_groups, 512, 30, 54]
            value, _ = self.encode_value(image, f16, h16=None, masks=mask, is_deep_update=False)

            holistic_features = self.holistic_encoder((f16, f8, f4), value, hidden_features=holistic_features)
  
        return holistic_features
        
    # Used in training only. 
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key, query_selection, memory_key, 
                    memory_shrinkage, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)

        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
        memory = readout(affinity, memory_value)
        memory = memory.view(batch_size, num_objects, self.value_dim_f16, *memory.shape[-2:])

        return memory

    def segment(self, multi_scale_features, memory_readout,
                    hidden_state, selector=None, h_out=True, strip_bg=True, holistic_features=None): 

        hidden_state, logits = self.decoder(*multi_scale_features, hidden_state, memory_readout, holistic_features=holistic_features, h_out=h_out)
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector
            
        logits, prob = aggregate(prob, dim=1, return_logits=True)
        if strip_bg:
            # Strip away the background
            prob = prob[:, 1:]

        return hidden_state, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        If model_path is provided, we load these from the model weights
        The actual parameters are then updated to the config in-place

        Otherwise we load it either from the config or default
        """
        
        # TODO: add holistic features config here, hardcoded for now
        if model_path is not None:
            # load the model and key/value/hidden dimensions with some hacks
            # config is updated with the loaded parameters
            # TODO: fix names for multi-scale memory 
            model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim_f16 = model_weights['key_proj.key_proj.weight'].shape[0]
            self.value_dim_f16 = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'decoder.hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['decoder.hidden_update.transform.weight'].shape[0]//3
            # print(f'Hyperparameters read from the model weights: '
                    # f'C^k={self.key_dim}, C^v={self.value_dim}, C^h={self.hidden_dim}')
        else:
            model_weights = None
            # load dimensions from config or default
            # ===============================key_dims======================================
            if 'key_dim_f16' not in config:
                self.key_dim_f16 = 64
                print(f'key_dim_f16 not found in config. Set to default {self.key_dim_f16}')
            else:
                self.key_dim_f16 = config['key_dim_f16']

            if 'key_dim_f8' not in config:
                self.key_dim_f8 = 32
                print(f'key_dim_f8 not found in config. Set to default {self.key_dim_f8}')
            else:
                self.key_dim_f8 = config['key_dim_f8']

            if 'key_dim_f4' not in config:
                self.key_dim_f4 = 16
                print(f'key_dim_f4 not found in config. Set to default {self.key_dim_f4}')
            else:
                self.key_dim_f4 = config['key_dim_f4']

            # ============================value_dims=======================================
            if 'value_dim_f16' not in config:
                self.value_dim_f16 = 512
                print(f'value_dim_f16 not found in config. Set to default {self.value_dim_f16}')
            else:
                self.value_dim_f16 = config['value_dim_f16']

            if 'value_dim_f8' not in config:
                self.value_dim_f8 = 256
                print(f'value_dim_f8 not found in config. Set to default {self.value_dim_f8}')
            else:
                self.value_dim_f8 = config['value_dim_f8']

            if 'value_dim_f4' not in config:
                self.value_dim_f4 = 128
                print(f'value_dim_f4 not found in config. Set to default {self.value_dim_f4}')
            else:
                self.value_dim_f4 = config['value_dim_f4']

            # ============================hidden_dims======================================
            if 'hidden_dim' not in config:
                self.hidden_dim = 64
                print(f'hidden_dim not found in config. Set to default {self.hidden_dim}')
            else:
                self.hidden_dim = config['hidden_dim']

            self.disable_hidden = (self.hidden_dim <= 0)

        config['key_dim_f16'] = self.key_dim_f16
        config['key_dim_f8'] = self.key_dim_f8
        config['key_dim_f4'] = self.key_dim_f4

        config['value_dim_f16'] = self.value_dim_f16
        config['value_dim_f8'] = self.value_dim_f8
        config['value_dim_f4'] = self.value_dim_f4

        config['hidden_dim'] = self.hidden_dim

        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    print('Converting weights from single object to multiple objects.')
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.load_state_dict(src_dict)
