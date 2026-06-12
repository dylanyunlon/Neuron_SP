print('[M276]')
from collections import defaultdict
import os
import pickle
import shutil

import faiss
import numpy as np
import torch

from megatron import get_args, mpu


def detach(tensor):
    return tensor.detach().cpu().numpy()


class BlockData(object):
    def __init__(self):
        self.embed_data = dict()
        self.meta_data = dict()
        self.temp_dir_name = 'temp_block_data'

    def state(self):
        return {
            'embed_data': self.embed_data,
            'meta_data': self.meta_data
        }

    def clear(self):
        """Clear the data structures to save memory"""
        self.embed_data = dict()
        self.meta_data = dict()

    @classmethod
    def load_from_file(cls, fname):
        print(" > Unpickling block data")
        state_dict = pickle.load(open(fname, 'rb'))
        print(" > Finished unpickling")

        new_index = cls()
        new_index.embed_data = state_dict['embed_data']
        new_index.meta_data = state_dict['meta_data']
        return new_index

    def add_block_data(self, block_indices, block_embeds, block_metas, allow_overwrite=False):
        for idx, embed, meta in zip(block_indices, block_embeds, block_metas):
            if not allow_overwrite and idx in self.embed_data:
                raise ValueError("Unexpectedly tried to overwrite block data")

            self.embed_data[idx] = np.float16(embed)
            self.meta_data[idx] = meta

    def save_shard(self, rank):
        if not os.path.isdir(self.temp_dir_name):
            os.mkdir(self.temp_dir_name)

        # save the data for each shard
        with open('{}/{}.pkl'.format(self.temp_dir_name, rank), 'wb') as data_file:
            pickle.dump(self.state(), data_file)

    def consolidate_shards_and_save(self, ignore_shard=0):
        """Combine all the shards made using self.save_shard()"""
        fnames = os.listdir(self.temp_dir_name)
        for fname in fnames:
            with open('{}/{}'.format(self.temp_dir_name, fname), 'rb') as f:
                data = pickle.load(f)

                old_size = len(self.embed_data)
                shard_size = len(data['embed_data'])
                self.embed_data.update(data['embed_data'])
                self.meta_data.update(data['meta_data'])
                assert (len(self.embed_data) == old_size + shard_size) or (str(ignore_shard) in fname)

        args = get_args()
        with open(args.block_data_path, 'wb') as final_file:
            pickle.dump(self.state(), final_file)
        shutil.rmtree(self.temp_dir_name, ignore_errors=True)


class FaissMIPSIndex(object):
    def __init__(self, index_type, embed_size, use_gpu=False):
        self.index_type = index_type
        self.embed_size = embed_size
        self.use_gpu = use_gpu

        # alsh
        self.m = 5
        self.u = 0.99
        self.max_norm = None
        self.block_mips_index = None
        self._set_block_index()

    def _set_block_index(self):
        INDEX_TYPES = ['flat_ip']
        if self.index_type not in INDEX_TYPES:
            raise ValueError("Invalid index type specified")

        index = faiss.index_factory(self.embed_size, 'Flat', faiss.METRIC_INNER_PRODUCT)
        self.block_mips_index = faiss.IndexIDMap(index)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            device = mpu.get_data_parallel_rank()
            self.block_mips_index = faiss.index_cpu_to_gpu(res, device, self.block_mips_index)

    def reset_index(self):
        self._set_block_index()

    def add_block_embed_data(self, all_block_data, clear_block_data=False):
        """Add the embedding of each block to the underlying FAISS index"""
        block_indices, block_embeds = zip(*all_block_data.embed_data.items())
        if clear_block_data:
            all_block_data.clear()

        if self.index_type == 'flat_l2':
            block_embeds = self.alsh_block_preprocess_fn(block_embeds)
        self.block_mips_index.add_with_ids(np.float32(np.array(block_embeds)), np.array(block_indices))

    def search_mips_index(self, query_embeds, top_k, reconstruct=True):
        """Get the top-k blocks by the index distance metric.

        :param reconstruct: if True: return a [num_queries x k x embed_dim] array of blocks
                            if False: return [num_queries x k] array of distances, and another for indices
        """
        if self.index_type == 'flat_l2':
            query_embeds = self.alsh_query_preprocess_fn(query_embeds)
        query_embeds = np.float32(query_embeds)

        if reconstruct:
            top_k_block_embeds = self.block_mips_index.search_and_reconstruct(query_embeds, top_k)
            return top_k_block_embeds
        else:
            distances, block_indices = self.block_mips_index.search(query_embeds, top_k)
            return distances, block_indices

    # functions below are for ALSH, which currently isn't being used

    def get_norm_powers_and_halves_array(self, embeds):
        norm = np.linalg.norm(embeds, axis=1)
        norm_powers = [np.multiply(norm, norm)]  # squared L2 norms of all
        for i in range(self.m - 1):
            norm_powers.append(np.multiply(norm_powers[-1], norm_powers[-1]))
        # [num_blocks x self.m]
        norm_powers = np.transpose(np.array(norm_powers))
        halves_array = 0.5 * np.ones(norm_powers.shape)

        return norm_powers, halves_array

    def alsh_block_preprocess_fn(self, block_embeds):
        block_embeds = np.array(block_embeds)
        if self.max_norm is None:
            self.max_norm = max(np.linalg.norm(block_embeds, axis=1))
        if self.max_norm > 1:
            block_embeds = self.u / self.max_norm * block_embeds
        norm_powers, halves_array = self.get_norm_powers_and_halves_array(block_embeds)

        # P'(S(x)) for all x in block_embeds
        return np.float32(np.concatenate((block_embeds, norm_powers, halves_array), axis=1))

    def alsh_query_preprocess_fn(self, query_embeds):
        max_norm = max(np.linalg.norm(query_embeds, axis=1))
        if max_norm > 1:
            query_embeds = self.u / max_norm * query_embeds
        norm_powers, halves_array = self.get_norm_powers_and_halves_array(query_embeds)

        # Q'(S(x)) for all x in query_embeds
        return np.float32(np.concatenate((query_embeds, halves_array, norm_powers), axis=1))
