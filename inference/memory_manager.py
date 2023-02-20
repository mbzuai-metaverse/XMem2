from typing import List
import torch
import warnings

from inference.kv_memory_store import KeyValueMemoryStore
from model.memory_util import *
from util.tensor_util import MultiscaleFeatures_16_8_4, MutliscaleValues_16_8_4


class MemoryUnit:
    def __init__(self, enable_long_term: bool, enable_long_term_usage: bool) -> None:
        self.temporary_work_mem = KeyValueMemoryStore(
            count_usage=enable_long_term)
        self.permanent_work_mem = KeyValueMemoryStore(count_usage=False)
        if enable_long_term:
            self.long_mem = KeyValueMemoryStore(
                count_usage=enable_long_term_usage)

        self.CK = None
        self.CV = None
        self.H = None
        self.W = None
        self.HW = None


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """

    def __init__(self, config):
        self.hidden_dim = config['hidden_dim']
        self.top_k = config['top_k']

        self.enable_long_term = config['enable_long_term']
        self.enable_long_term_usage = config['enable_long_term_count_usage']
        if self.enable_long_term:
            # maximum work memory size
            self.max_mt_frames = config['max_mid_term_frames']
            # minimum number of frames to keep in work memory when consolidating
            self.min_mt_frames = config['min_mid_term_frames']
            self.num_prototypes = config['num_prototypes']
            self.max_long_elements = config['max_long_term_elements']

        # dimensions will be inferred from input later
        # self.CK = self.CV = None
        # self.H = self.W = None

        # The hidden state will be stored in a single tensor for all objects
        # B x num_objects x CH x H x W
        self.hidden = None

        self.memory_16 = MemoryUnit(enable_long_term=self.enable_long_term, enable_long_term_usage=self.enable_long_term_usage)
        self.memory_8 = MemoryUnit(enable_long_term=self.enable_long_term, enable_long_term_usage=self.enable_long_term_usage)
        self.memory_4 = MemoryUnit(enable_long_term=self.enable_long_term, enable_long_term_usage=self.enable_long_term_usage)

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

    def match_memory(self, multiscale_features: MultiscaleFeatures_16_8_4, disable_usage_updates=False) -> List[torch.tensor]:
        # query_key: B x C^k x H x W
        # selection:  B x C^k x H x W
        # TODO: keep groups in both..?
        # 1x64x30x54

        """
        Memory readout using keys
        """

        readouts = []
        mem_units = self.get_memories()

        for features, mem_unit in zip(multiscale_features.get_all_scales(), mem_units):
            query_key = features.key
            selection = features.selection

            temporary_work_mem = mem_unit.temporary_work_mem
            permanent_work_mem = mem_unit.permanent_work_mem
            long_mem = mem_unit.long_mem

            num_groups = temporary_work_mem.num_groups
            h, w = query_key.shape[-2:]
            temp_work_mem_size = temporary_work_mem.size

            query_key = query_key.flatten(start_dim=2)
            selection = selection.flatten(start_dim=2) if selection is not None else None
            
            if self.enable_long_term and long_mem.engaged():
                # Use long-term memory
                long_mem_size = long_mem.size

                memory_key = torch.cat(
                    [long_mem.key, temporary_work_mem.key, permanent_work_mem.key], -1)
                shrinkage = torch.cat(
                    [long_mem.shrinkage, temporary_work_mem.shrinkage, permanent_work_mem.shrinkage], -1)

                similarity = get_similarity(
                    memory_key, shrinkage, query_key, selection)

                long_mem_similarity = similarity[:, :long_mem_size]
                temp_work_mem_similarity = similarity[:,
                                                      long_mem_size:long_mem_size+temp_work_mem_size]
                perm_work_mem_similarity = similarity[:,
                                                      long_mem_size+temp_work_mem_size:]

                # get the usage with the first group
                # the first group always have all the keys valid
                affinity, usage = do_softmax(
                    torch.cat([long_mem_similarity[:, -long_mem.get_v_size(0):],
                              temp_work_mem_similarity, perm_work_mem_similarity], 1),
                    top_k=self.top_k, inplace=True, return_usage=True)
                affinity = [affinity]

                # compute affinity group by group as later groups only have a subset of keys
                for gi in range(1, num_groups):
                    if gi < long_mem.num_groups:
                        # merge working and lt similarities before softmax
                        affinity_one_group = do_softmax(
                            torch.cat([long_mem_similarity[:, -long_mem.get_v_size(gi):],
                                       temp_work_mem_similarity[:, -
                                                                temporary_work_mem.get_v_size(gi):],
                                       perm_work_mem_similarity[:, -permanent_work_mem.get_v_size(gi):]],
                                      1),
                            top_k=self.top_k, inplace=True)
                    else:
                        # no long-term memory for this group
                        affinity_one_group = do_softmax(torch.cat([
                            temp_work_mem_similarity[:, -
                                                     temporary_work_mem.get_v_size(gi):],
                            perm_work_mem_similarity[:, -permanent_work_mem.get_v_size(gi):]],
                            1),
                            top_k=self.top_k, inplace=(gi == num_groups-1))
                    affinity.append(affinity_one_group)

                all_memory_value = []
                for gi, gv in enumerate(temporary_work_mem.value):
                    # merge the working and lt values before readout
                    if gi < long_mem.num_groups:
                        all_memory_value.append(torch.cat(
                            [long_mem.value[gi], temporary_work_mem.value[gi], permanent_work_mem.value[gi]], -1))
                    else:
                        all_memory_value.append(torch.cat(
                            [temporary_work_mem.value[gi], permanent_work_mem.value[gi]], -1))

                """
                Record memory usage for working and long-term memory
                """
                if not disable_usage_updates:
                    # ignore the index return for long-term memory
                    # no usage for permanent memory
                    work_usage = usage[:,
                                       long_mem_size:long_mem_size+temp_work_mem_size]
                    temporary_work_mem.update_usage(work_usage.flatten())

                    if self.enable_long_term_usage:
                        # ignore the index return for working memory
                        long_usage = usage[:, :long_mem_size]
                        long_mem.update_usage(long_usage.flatten())
            else:
                memory_key = torch.cat(
                    [temporary_work_mem.key, permanent_work_mem.key], -1)
                shrinkage = torch.cat(
                    [temporary_work_mem.shrinkage, permanent_work_mem.shrinkage], -1)
                # No long-term memory
                similarity = get_similarity(
                    memory_key, shrinkage, query_key, selection)

                if self.enable_long_term:
                    affinity, usage = do_softmax(similarity, inplace=(num_groups == 1),
                                                 top_k=self.top_k, return_usage=True)
                    if not disable_usage_updates:
                        # Record memory usage for working memory
                        temporary_work_mem.update_usage(
                            usage[:, :temp_work_mem_size].flatten())
                else:
                    affinity = do_softmax(similarity, inplace=(num_groups == 1),
                                          top_k=self.top_k, return_usage=False)

                affinity = [affinity]

                # compute affinity group by group as later groups only have a subset of keys
                for gi in range(1, num_groups):
                    affinity_one_group = do_softmax(similarity[:, -temporary_work_mem.get_v_size(gi):],
                                                    top_k=self.top_k, inplace=(gi == num_groups-1))
                    affinity.append(affinity_one_group)

                all_memory_value = []
                for gi, gv in enumerate(temporary_work_mem.value):
                    all_memory_value.append(torch.cat(
                        [temporary_work_mem.value[gi], permanent_work_mem.value[gi]], -1))

            # Shared affinity within each group
            all_readout_mem = torch.cat([
                self._readout(affinity[gi], gv)
                for gi, gv in enumerate(all_memory_value)
            ], 0)

            readout_correct_shape = all_readout_mem.view(all_readout_mem.shape[0], mem_unit.CV, h, w)
            readouts.append(readout_correct_shape)

        return readouts

    def add_memory(self, multiscale_features: MultiscaleFeatures_16_8_4, multiscale_values: MutliscaleValues_16_8_4, objects, permanent=False, ignore=False):
        # key: 1*C*H*W
        # value: 1*num_objects*C*H*W
        # objects contain a list of object indices

        # keys =          [key_features.key_f16,          key_features.key_f8,        key_features.key_f4]
        # selections =    [key_features.selection_f16,    key_features.selection_f8,  key_features.selection_f4]
        # mem_units =     [self.memory_16,                self.memory_8,              self.memory_4]

        mem_units = self.get_memories()

        for features, value, mem_unit in zip(multiscale_features.get_all_scales(), multiscale_values.get_all_scales(), mem_units):
            # key:   1*C*N
            # value: num_objects*C*N
            key = features.key
            shrinkage = features.shrinkage
            selection = features.selection
            
            if mem_unit.H is None or self.reset_config:
                self.reset_config = False
                mem_unit.H, mem_unit.W = key.shape[-2:]
                mem_unit.HW = mem_unit.H*mem_unit.W
                if self.enable_long_term:
                    # convert from num. frames to num. nodes
                    mem_unit.min_work_elements = self.min_mt_frames*mem_unit.HW
                    mem_unit.max_work_elements = self.max_mt_frames*mem_unit.HW

            key = key.flatten(start_dim=2)
            shrinkage = shrinkage.flatten(start_dim=2)
            value = value[0].flatten(start_dim=2)

            mem_unit.CK = key.shape[1]
            mem_unit.CV = value.shape[1]

            if selection is not None:
                if not self.enable_long_term:
                    warnings.warn(
                        'the selection factor is only needed in long-term mode', UserWarning)
                    selection = None
                else:
                    selection = selection.flatten(start_dim=2)

            if ignore:
                # all permanent frames are pre-placed into permanent memory (when using our memory modification)
                pass
                # also ignores the first frame (#0) when using original memory mechanism, since it's already in the permanent memory
            elif permanent:
                mem_unit.permanent_work_mem.add(
                    key, value, shrinkage, selection, objects)
            else:
                mem_unit.temporary_work_mem.add(
                    key, value, shrinkage, selection, objects)

            if not mem_unit.temporary_work_mem.engaged():
                # first frame; we need to have both memories engaged to avoid crashes when concating
                # so we just initialize the temporary one with an empty tensor
                key0 = key[..., 0:0]
                value0 = value[..., 0:0]
                shrinkage0 = shrinkage[..., 0:0]
                selection0 = selection[..., 0:0]

                mem_unit.temporary_work_mem.add(
                    key0, value0, shrinkage0, selection0, objects)

            # long-term memory cleanup
            if self.enable_long_term:
                # Do memory compressed if needed
                if mem_unit.temporary_work_mem.size >= mem_unit.max_work_elements:
                    # if we have more then N elements in the work memory
                    # Remove obsolete features if needed
                    if mem_unit.long_mem.size >= (mem_unit.max_long_elements-mem_unit.num_prototypes):
                        mem_unit.long_mem.remove_obsolete_features(
                            mem_unit.max_long_elements-mem_unit.num_prototypes)

                    # We NEVER remove anything from the working memory
                    mem_unit.compress_features()

    def create_hidden_state(self, n, sample_key):
        # n is the TOTAL number of objects
        # [B, num_objects, hidden_dim, h, w]
        h, w = sample_key.shape[-2:]
        if self.hidden is None:
            self.hidden = torch.zeros(
                (1, n, self.hidden_dim, h, w), device=sample_key.device)
        elif self.hidden.shape[1] != n:
            # ONLY if the shape[1] (num_objects) != total number of objects
            self.hidden = torch.cat([
                self.hidden,
                torch.zeros(
                    (1, n-self.hidden.shape[1], self.hidden_dim, h, w), device=sample_key.device)
            ], 1)

        assert (self.hidden.shape[1] == n)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def get_hidden(self):
        return self.hidden

    def get_memories(self):
        return (self.memory_16, self.memory_8, self.memory_4)

    def compress_features(self):
        for mem_unit in self.get_memories():
            HW = mem_unit.HW
            candidate_value = []
            total_work_mem_size = mem_unit.temporary_work_mem.size
            for gv in mem_unit.temporary_work_mem.value:
                # Some object groups might be added later in the video
                # So not all keys have values associated with all objects
                # We need to keep track of the key->value validity
                mem_size_in_this_group = gv.shape[-1]
                if mem_size_in_this_group == total_work_mem_size:
                    # full LT
                    candidate_value.append(gv[:, :, :-mem_unit.min_work_elements])
                else:
                    # mem_size is smaller than total_work_mem_size, but at least HW
                    assert HW <= mem_size_in_this_group < total_work_mem_size
                    if mem_size_in_this_group > mem_unit.min_work_elements:
                        # part of this object group still goes into LT
                        candidate_value.append(gv[:, :, :-mem_unit.min_work_elements])
                    else:
                        # this object group cannot go to the LT at all
                        candidate_value.append(None)

            # perform memory consolidation
            # now starts at zero, because the 1st frame is going into permanent memory
            prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
                *mem_unit.temporary_work_mem.get_all_sliced(0, -mem_unit.min_work_elements), candidate_value)

            # remove consolidated working memory
            mem_unit.temporary_work_mem.sieve_by_range(
                0, -mem_unit.min_work_elements, min_size=mem_unit.min_work_elements+HW)

            # add to long-term memory
            mem_unit.long_mem.add(prototype_key, prototype_value,
                            prototype_shrinkage, selection=None, objects=None)

    def consolidation(self, candidate_key, candidate_shrinkage, candidate_selection, usage, candidate_value):
        # keys: 1*C*N
        # values: num_objects*C*N
        N = candidate_key.shape[-1]

        # find the indices with max usage
        _, max_usage_indices = torch.topk(
            usage, k=self.num_prototypes, dim=-1, sorted=True)
        prototype_indices = max_usage_indices.flatten()

        # Prototypes are invalid for out-of-bound groups
        validity = [prototype_indices >= (
            N-gv.shape[2]) if gv is not None else None for gv in candidate_value]

        prototype_key = candidate_key[:, :, prototype_indices]
        prototype_selection = candidate_selection[:, :,
                                                  prototype_indices] if candidate_selection is not None else None

        """
        Potentiation step
        """
        similarity = get_similarity(
            candidate_key, candidate_shrinkage, prototype_key, prototype_selection)

        # convert similarity to affinity
        # need to do it group by group since the softmax normalization would be different
        affinity = [
            do_softmax(similarity[:, -gv.shape[2]:,
                       validity[gi]]) if gv is not None else None
            for gi, gv in enumerate(candidate_value)
        ]

        # some values can be have all False validity. Weed them out.
        affinity = [
            aff if aff is None or aff.shape[-1] > 0 else None for aff in affinity
        ]

        # readout the values
        prototype_value = [
            self._readout(
                affinity[gi], gv) if affinity[gi] is not None else None
            for gi, gv in enumerate(candidate_value)
        ]

        # readout the shrinkage term
        prototype_shrinkage = self._readout(
            affinity[0], candidate_shrinkage) if candidate_shrinkage is not None else None

        return prototype_key, prototype_value, prototype_shrinkage
