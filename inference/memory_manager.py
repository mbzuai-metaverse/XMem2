import torch
import warnings

from inference.kv_memory_store import KeyValueMemoryStore
from model.memory_util import *


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """

    def __init__(self, config):
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.top_k = config['top_k']

        self.enable_long_term = config['enable_long_term']
        self.enable_long_term_usage = config['enable_long_term_count_usage']
        if self.enable_long_term:
            self.max_mt_frames = config['max_mid_term_frames']  # maximum work memory size
            self.min_mt_frames = config['min_mid_term_frames']  # minimum number of frames to keep in work memory when consolidating
            self.num_prototypes = config['num_prototypes']
            self.max_long_elements = config['max_long_term_elements']

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The hidden state will be stored in a single tensor for all objects
        # B x num_objects x CH x H x W
        self.hidden = None

        self.temporary_work_mem = KeyValueMemoryStore(count_usage=self.enable_long_term)
        self.permanent_work_mem = KeyValueMemoryStore(count_usage=False)
        self.frame_id_to_permanent_mem_idx = dict()
        if self.enable_long_term:
            self.long_mem = KeyValueMemoryStore(count_usage=self.enable_long_term_usage)

        self.reset_config = True

    def update_config(self, config):
        self.reset_config = True
        self.hidden_dim = config['hidden_dim']
        self.top_k = config['top_k']

        assert self.enable_long_term == config['enable_long_term'], 'cannot update this'
        assert self.enable_long_term_usage == config['enable_long_term_count_usage'], 'cannot update this'

        self.enable_long_term_usage = config['enable_long_term_count_usage']
        if self.enable_long_term:
            self.max_mt_frames = config['max_mid_term_frames']
            self.min_mt_frames = config['min_mid_term_frames']
            self.num_prototypes = config['num_prototypes']
            self.max_long_elements = config['max_long_term_elements']

    def _readout(self, affinity, v):
        # this function is for a single object group
        return v @ affinity

    def match_memory(self, query_key, selection, disable_usage_updates=False):
        # query_key: B x C^k x H x W
        # selection:  B x C^k x H x W
        # TODO: keep groups in both..?
        # 1x64x30x54

        # = permanent_work_mem.num_groups, since it's always >= temporary_work_mem.num_groups
        num_groups = max(self.temporary_work_mem.num_groups, self.permanent_work_mem.num_groups)
        h, w = query_key.shape[-2:]

        query_key = query_key.flatten(start_dim=2)
        selection = selection.flatten(start_dim=2) if selection is not None else None

        """
        Memory readout using keys
        """

        temp_work_mem_size = self.temporary_work_mem.size
        if self.enable_long_term and self.long_mem.engaged():
            # Use long-term memory
            long_mem_size = self.long_mem.size

            memory_key = torch.cat([self.long_mem.key, self.temporary_work_mem.key, self.permanent_work_mem.key], -1)
            shrinkage = torch.cat([self.long_mem.shrinkage, self.temporary_work_mem.shrinkage, self.permanent_work_mem.shrinkage], -1)

            similarity = get_similarity(memory_key, shrinkage, query_key, selection)

            long_mem_similarity = similarity[:, :long_mem_size]
            temp_work_mem_similarity = similarity[:, long_mem_size:long_mem_size+temp_work_mem_size]
            perm_work_mem_similarity = similarity[:, long_mem_size+temp_work_mem_size:]

            # get the usage with the first group
            # the first group always have all the keys valid
            affinity, usage = do_softmax(
                torch.cat([long_mem_similarity[:, -self.long_mem.get_v_size(0):], temp_work_mem_similarity, perm_work_mem_similarity], 1),
                top_k=self.top_k, inplace=True, return_usage=True)
            affinity = [affinity]

            # compute affinity group by group as later groups only have a subset of keys
            for gi in range(1, num_groups):
                temp_group_v_size = self.temporary_work_mem.get_v_size(gi)
                perm_group_v_size = self.permanent_work_mem.get_v_size(gi)
                temp_sim_size = temp_work_mem_similarity.shape[1] 
                perm_sim_size = perm_work_mem_similarity.shape[1] 

                if gi < self.long_mem.num_groups:
                    # merge working and lt similarities before softmax
                    affinity_one_group = do_softmax(
                        torch.cat([long_mem_similarity[:, -self.long_mem.get_v_size(gi):],
                                   temp_work_mem_similarity[:, temp_sim_size-temp_group_v_size:],
                                   perm_work_mem_similarity[:, perm_sim_size-perm_group_v_size:]],
                                dim=1),
                        top_k=self.top_k, inplace=True)
                else:
                    # no long-term memory for this group
                    affinity_one_group = do_softmax(torch.cat([
                            temp_work_mem_similarity[:, temp_sim_size-temp_group_v_size:], 
                            perm_work_mem_similarity[:, perm_sim_size-perm_group_v_size:]],
                            1),
                        top_k=self.top_k, inplace=(gi == num_groups-1))
                affinity.append(affinity_one_group)

            all_memory_value = []
            for gi in range(num_groups):
                # merge the working and lt values before readout
                if gi < self.long_mem.num_groups:
                    all_memory_value.append(torch.cat([self.long_mem.value[gi], self.temporary_work_mem.value[gi], self.permanent_work_mem.value[gi]], -1))
                else:
                    all_memory_value.append(torch.cat([self.temporary_work_mem.value[gi], self.permanent_work_mem.value[gi]], -1))

            """
            Record memory usage for working and long-term memory
            """
            if not disable_usage_updates:
            # ignore the index return for long-term memory
                work_usage = usage[:, long_mem_size:long_mem_size+temp_work_mem_size]  # no usage for permanent memory
                self.temporary_work_mem.update_usage(work_usage.flatten())

                if self.enable_long_term_usage:
                    # ignore the index return for working memory
                    long_usage = usage[:, :long_mem_size]
                    self.long_mem.update_usage(long_usage.flatten())
        else:
            memory_key = torch.cat([self.temporary_work_mem.key, self.permanent_work_mem.key], -1)
            shrinkage = torch.cat([self.temporary_work_mem.shrinkage, self.permanent_work_mem.shrinkage], -1)
            # No long-term memory
            similarity = get_similarity(memory_key, shrinkage, query_key, selection)
            temp_work_mem_similarity = similarity[:, :temp_work_mem_size]
            perm_work_mem_similarity = similarity[:, temp_work_mem_size:]

            if self.enable_long_term:
                affinity, usage = do_softmax(similarity, inplace=(num_groups == 1),
                                             top_k=self.top_k, return_usage=True)
                if not disable_usage_updates:
                    # Record memory usage for working memory
                    self.temporary_work_mem.update_usage(usage[:, :temp_work_mem_size].flatten())
            else:
                affinity = do_softmax(similarity, inplace=(num_groups == 1),
                                      top_k=self.top_k, return_usage=False)

            affinity = [affinity]

            # compute affinity group by group as later groups only have a subset of keys
            for gi in range(1, num_groups):
                temp_group_v_size = self.temporary_work_mem.get_v_size(gi)
                perm_group_v_size = self.permanent_work_mem.get_v_size(gi)
                temp_sim_size = temp_work_mem_similarity.shape[1] 
                perm_sim_size = perm_work_mem_similarity.shape[1] 

                affinity_one_group = do_softmax(
                    torch.cat([
                        # concats empty tensor if the group is also empty for temporary memory
                        temp_work_mem_similarity[:, temp_sim_size-temp_group_v_size:], 
                        perm_work_mem_similarity[:, perm_sim_size-perm_group_v_size:], 
                    ], dim=1),
                    top_k=self.top_k, inplace=(gi == num_groups-1)
                )
                affinity.append(affinity_one_group)

            all_memory_value = []
            for gi in range(num_groups):
                group_v_cat = torch.cat([self.temporary_work_mem.value[gi], self.permanent_work_mem.value[gi]], -1)
                all_memory_value.append(group_v_cat)

        # Shared affinity within each group
        all_readout_mem = torch.cat([
            self._readout(affinity[gi], gv)
            for gi, gv in enumerate(all_memory_value)
        ], 0)

        return all_readout_mem.view(all_readout_mem.shape[0], self.CV, h, w)

    def update_permanent_memory(self, frame_idx, key, shrinkage, value, selection=None):
        saved_pos = self.frame_id_to_permanent_mem_idx[frame_idx]

        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2)
        value = value[0].flatten(start_dim=2)

        if selection is not None:
            selection = selection.flatten(start_dim=2)

        self.permanent_work_mem.replace_at(saved_pos, key, value, shrinkage, selection)

    def remove_from_permanent_memory(self, frame_idx):
        elem_size = self.HW
        saved_pos = self.frame_id_to_permanent_mem_idx[frame_idx]

        self.permanent_work_mem.remove_at(saved_pos, elem_size)

        del self.frame_id_to_permanent_mem_idx[frame_idx]

    def add_memory(self, key, shrinkage, value, objects, selection=None, permanent=False, ignore=False, ti=None):
        # key: 1*C*H*W
        # value: 1*num_objects*C*H*W
        # objects contain a list of object indices
        if self.H is None or self.reset_config:
            self.reset_config = False
            self.H, self.W = key.shape[-2:]
            self.HW = self.H*self.W
            if self.enable_long_term:
                # convert from num. frames to num. nodes
                self.min_work_elements = self.min_mt_frames*self.HW
                self.max_work_elements = self.max_mt_frames*self.HW

        # key:   1*C*N
        # value: num_objects*C*N
        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2)
        value = value[0].flatten(start_dim=2)

        self.CK = key.shape[1]
        self.CV = value.shape[1]

        if selection is not None:
            if not self.enable_long_term:
                warnings.warn('the selection factor is only needed in long-term mode', UserWarning)
            selection = selection.flatten(start_dim=2)

        if ignore:
            pass # all permanent frames are pre-placed into permanent memory (when using our memory modification) 
                # also ignores the first frame (#0) when using original memory mechanism, since it's already in the permanent memory
        elif permanent:
            pos = self.permanent_work_mem.add(key, value, shrinkage, selection, objects)
            if ti is not None:
                self.frame_id_to_permanent_mem_idx[ti] = pos
        else:
            self.temporary_work_mem.add(key, value, shrinkage, selection, objects)
            
        
        num_temp_groups = self.temporary_work_mem.num_groups
        num_perm_groups = self.permanent_work_mem.num_groups

        if not self.temporary_work_mem.engaged() or (num_temp_groups != num_perm_groups):
            # print(f"PERM_NUM_GROUPS={num_perm_groups} vs TEMP_NUM_GROUPS={num_temp_groups}", end=' ')

            # first frame or new group; we need to have both memories engaged to avoid crashes when concating
            # so we just initialize the temporary one with an empty tensor
            key0 = key[..., 0:0]
            value0 = value[..., 0:0]
            shrinkage0 = shrinkage[..., 0:0]
            selection0 = selection[..., 0:0]
            if num_perm_groups > num_temp_groups:
                # for preloading into permanent memory
                self.temporary_work_mem.add(key0, value0, shrinkage0, selection0, objects)
            else:
                # for original memory mechanism
                self.permanent_work_mem.add(key0, value0, shrinkage0, selection0, objects)
            
            # print(f"AFTER->PERM_NUM_GROUPS={self.permanent_work_mem.num_groups} vs TEMP_NUM_GROUPS={self.temporary_work_mem.num_groups}")
            
        # long-term memory cleanup
        if self.enable_long_term:
            # Do memory compressed if needed
            if self.temporary_work_mem.size >= self.max_work_elements:
                # if we have more then N elements in the work memory
                # Remove obsolete features if needed
                if self.long_mem.size >= (self.max_long_elements-self.num_prototypes):
                    self.long_mem.remove_obsolete_features(self.max_long_elements-self.num_prototypes)

                # We NEVER remove anything from the working memory
                self.compress_features()

    def create_hidden_state(self, n, sample_key):
        # n is the TOTAL number of objects
        h, w = sample_key.shape[-2:]
        if self.hidden is None:
            self.hidden = torch.zeros((1, n, self.hidden_dim, h, w), device=sample_key.device)
        elif self.hidden.shape[1] != n:
            self.hidden = torch.cat([
                self.hidden,
                torch.zeros((1, n-self.hidden.shape[1], self.hidden_dim, h, w), device=sample_key.device)
            ], 1)

        assert (self.hidden.shape[1] == n)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def get_hidden(self):
        return self.hidden
    
    def frame_already_saved(self, ti):
        return ti in self.frame_id_to_permanent_mem_idx

    # def slices_excluding_permanent(self, group_value, start, end):
    #     HW = self.HW
    #     group_value[:,:,HW:-self.min_work_elements+HW]

    #     slices = []

    #     # this won't work because after just 1 consolidation all permanent frames are going to be god know where
    #     # and their indices would mean nothing
    #     # How about have 2 separate tensors and concatenate them just for memory reading?
    #     all_indices = torch.arange(self.temporary_work_mem.size // HW)  # all frames indices from 0 to ...

    def compress_features(self):
        HW = self.HW
        candidate_value = []
        total_work_mem_size = self.temporary_work_mem.size
        for gv in self.temporary_work_mem.value:
            # Some object groups might be added later in the video
            # So not all keys have values associated with all objects
            # We need to keep track of the key->value validity
            mem_size_in_this_group = gv.shape[-1]
            if mem_size_in_this_group == total_work_mem_size:
                # full LT
                candidate_value.append(gv[:, :, :-self.min_work_elements])
            else:
                # mem_size is smaller than total_work_mem_size, but at least HW
                assert HW <= mem_size_in_this_group < total_work_mem_size
                if mem_size_in_this_group > self.min_work_elements:
                    # part of this object group still goes into LT
                    candidate_value.append(gv[:, :, :-self.min_work_elements])
                else:
                    # this object group cannot go to the LT at all
                    candidate_value.append(None)

        # perform memory consolidation
        # now starts at zero, because the 1st frame is going into permanent memory
        prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
            *self.temporary_work_mem.get_all_sliced(0, -self.min_work_elements), candidate_value)

        # remove consolidated working memory
        self.temporary_work_mem.sieve_by_range(0, -self.min_work_elements, min_size=self.min_work_elements+HW)

        # add to long-term memory
        self.long_mem.add(prototype_key, prototype_value, prototype_shrinkage, selection=None, objects=None)

    def consolidation(self, candidate_key, candidate_shrinkage, candidate_selection, usage, candidate_value):
        # keys: 1*C*N
        # values: num_objects*C*N
        N = candidate_key.shape[-1]

        # find the indices with max usage
        _, max_usage_indices = torch.topk(usage, k=self.num_prototypes, dim=-1, sorted=True)
        prototype_indices = max_usage_indices.flatten()

        # Prototypes are invalid for out-of-bound groups
        validity = [prototype_indices >= (N-gv.shape[2]) if gv is not None else None for gv in candidate_value]

        prototype_key = candidate_key[:, :, prototype_indices]
        prototype_selection = candidate_selection[:, :, prototype_indices] if candidate_selection is not None else None

        """
        Potentiation step
        """
        similarity = get_similarity(candidate_key, candidate_shrinkage, prototype_key, prototype_selection)

        # convert similarity to affinity
        # need to do it group by group since the softmax normalization would be different
        affinity = [
            do_softmax(similarity[:, -gv.shape[2]:, validity[gi]]) if gv is not None else None
            for gi, gv in enumerate(candidate_value)
        ]

        # some values can be have all False validity. Weed them out.
        affinity = [
            aff if aff is None or aff.shape[-1] > 0 else None for aff in affinity
        ]

        # readout the values
        prototype_value = [
            self._readout(affinity[gi], gv) if affinity[gi] is not None else None
            for gi, gv in enumerate(candidate_value)
        ]

        # readout the shrinkage term
        prototype_shrinkage = self._readout(affinity[0], candidate_shrinkage) if candidate_shrinkage is not None else None

        return prototype_key, prototype_value, prototype_shrinkage
    
    def copy_perm_mem_only(self):
        new_mem = MemoryManager(config=self.config)

        if self.permanent_work_mem.key is None or self.permanent_work_mem.key.size(-1) == 0:
            return new_mem
        
        new_mem.permanent_work_mem = self.permanent_work_mem
        new_mem.frame_id_to_permanent_mem_idx = self.frame_id_to_permanent_mem_idx
        
        key0 = self.permanent_work_mem.key[..., 0:0]
        value0 = self.permanent_work_mem.value[0][..., 0:0]
        shrinkage0 = self.permanent_work_mem.shrinkage[..., 0:0] if self.permanent_work_mem.shrinkage is not None else None
        selection0 = self.permanent_work_mem.selection[..., 0:0] if self.permanent_work_mem.selection is not None else None

        new_mem.temporary_work_mem.add(key0, value0, shrinkage0, selection0, self.permanent_work_mem.all_objects)

        new_mem.CK = self.permanent_work_mem.key.shape[1]
        new_mem.CV = self.permanent_work_mem.value[0].shape[1]

        key_shape = self.permanent_work_mem.key.shape
        sample_key = self.permanent_work_mem.key[..., 0:self.HW].view(*key_shape[:-1], self.H, self.W)
        new_mem.create_hidden_state(len(self.permanent_work_mem.all_objects), sample_key)

        new_mem.temporary_work_mem.obj_groups = self.temporary_work_mem.obj_groups
        new_mem.temporary_work_mem.all_objects = self.temporary_work_mem.all_objects


        new_mem.CK = self.CK
        new_mem.CV = self.CV
        new_mem.H = self.H
        new_mem.W = self.W
        new_mem.HW = self.HW

        return new_mem

