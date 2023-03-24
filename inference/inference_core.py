from time import perf_counter

import torch
from inference.memory_manager import MemoryManager
from model.network import XMem
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by, unpad


class InferenceCore:
    def __init__(self, network:XMem, config):
        self.config = config
        self.network = network
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)

        self.clear_memory()
        self.all_labels = None

        # warmup
        self.network.encode_key(torch.zeros((1, 3, 480, 854), device='cuda:0'))

    def clear_memory(self, keep_permanent=False):
        self.curr_ti = -1
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        if keep_permanent:
            new_memory = self.memory.copy_perm_mem_only()
        else:
            new_memory = MemoryManager(config=self.config)

        self.memory = new_memory

    def update_config(self, config):
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)
        self.memory.update_config(config)

    def set_all_labels(self, all_labels):
        # self.all_labels = [l.item() for l in all_labels]
        self.all_labels = all_labels

    def encode_frame_key(self, image):
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension

        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image,
                                                                         need_ek=True,
                                                                         need_sk=True)

        return key, shrinkage, selection
    def step(self, image, mask=None, valid_labels=None, end=False, manually_curated_masks=False, disable_memory_updates=False, do_not_add_mask_to_memory=False, return_key_and_stuff=False):
        # For feedback:
        #   1. We run the model as usual
        #   2. We get feedback: 2 lists, one with good prediction indices, one with bad
        #   3. We force the good frames (+ annotated frames) to stay in working memory forever
        #   4. We force the bad frames to never even get added to the working memory
        #   5. Rerun with these settings 
        # image: 3*H*W
        # mask: num_objects*H*W or None
        self.curr_ti += 1
            
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0) # add the batch dimension
        if manually_curated_masks:
            is_mem_frame = (mask is not None) and (not end)
        else:
            is_mem_frame = ((self.curr_ti-self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end)

        is_ignore = do_not_add_mask_to_memory  # to avoid adding permanent memory frames twice, since they are alredy in the memory

        need_segment = (valid_labels is None) or (len(self.all_labels) != len(valid_labels))
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame) or  # synchronized
            (not self.deep_update_sync and self.curr_ti-self.last_deep_update_ti >= self.deep_update_every) # no-sync
        ) and (not end)
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (not end)

        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image, 
                                                    need_ek=(self.enable_long_term or need_segment), 
                                                    need_sk=True)
        multi_scale_features = (f16, f8, f4)

        if disable_memory_updates:
            is_normal_update = False
            is_deep_update = False
            is_mem_frame = False
            self.curr_ti -= 1  # do not advance the iteration further

        # segment the current frame is needed
        if need_segment:
            memory_readout = self.memory.match_memory(key, selection, disable_usage_updates=disable_memory_updates).unsqueeze(0)
            hidden, _, pred_prob_with_bg = self.network.segment(multi_scale_features, memory_readout, 
                                    self.memory.get_hidden(), h_out=is_normal_update, strip_bg=False)
            # remove batch dim
            pred_prob_with_bg = pred_prob_with_bg[0]
            pred_prob_no_bg = pred_prob_with_bg[1:]
            if is_normal_update:
                self.memory.set_hidden(hidden)
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        # use the input mask if any
        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)

            if pred_prob_no_bg is not None:
                # if we have a predicted mask, we work on it
                # make pred_prob_no_bg consistent with the input mask
                mask_regions = (mask.sum(0) > 0.5)
                pred_prob_no_bg[:, mask_regions] = 0
                # shift by 1 because mask/pred_prob_no_bg do not contain background
                mask = mask.type_as(pred_prob_no_bg)
                if valid_labels is not None:
                    shift_by_one_non_labels = [i for i in range(pred_prob_no_bg.shape[0]) if (i+1) not in valid_labels]
                    # non-labelled objects are copied from the predicted mask
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]
            pred_prob_with_bg = aggregate(mask, dim=0)

            # also create new hidden states
            if not disable_memory_updates:
                self.memory.create_hidden_state(len(self.all_labels), key)

        # save as memory if needed
        if is_mem_frame:
            value, hidden = self.network.encode_value(image, f16, self.memory.get_hidden(), 
                                    pred_prob_with_bg[1:].unsqueeze(0), is_deep_update=is_deep_update)
            self.memory.add_memory(key, shrinkage, value, self.all_labels, 
                                    selection=selection if self.enable_long_term else None, ignore=is_ignore)
            
            self.last_mem_ti = self.curr_ti

            if is_deep_update:
                self.memory.set_hidden(hidden)
                self.last_deep_update_ti = self.curr_ti

        res = unpad(pred_prob_with_bg, self.pad)

        if return_key_and_stuff:
            return res, key, shrinkage, selection
        else:
            return res

    def put_to_permanent_memory(self, image, mask, ti=None):
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0) # add the batch dimension
        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image, 
                                            need_ek=True, 
                                            need_sk=True)

        mask, _ = pad_divide_by(mask, 16)

        pred_prob_with_bg = aggregate(mask, dim=0)
        self.memory.create_hidden_state(len(self.all_labels), key)

        value, hidden = self.network.encode_value(image, f16, self.memory.get_hidden(), 
                                    pred_prob_with_bg[1:].unsqueeze(0), is_deep_update=False)
        
        is_update = self.memory.frame_already_saved(ti)
        # print(ti, f"update={is_update}")
        if self.memory.frame_already_saved(ti):
            self.memory.update_permanent_memory(ti, key, shrinkage, value, selection=selection if self.enable_long_term else None)
        else:                       
            self.memory.add_memory(key, shrinkage, value, self.all_labels, 
                                        selection=selection if self.enable_long_term else None, permanent=True, ti=ti)
            
        # print(self.memory.permanent_work_mem.key.shape)

        return is_update
    
    def remove_from_permanent_memory(self, frame_idx):
        self.memory.remove_from_permanent_memory(frame_idx)
    
    @property
    def permanent_memory_frames(self):
        return list(self.memory.frame_id_to_permanent_mem_idx.keys())