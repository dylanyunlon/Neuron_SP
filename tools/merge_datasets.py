# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# M1171: Megatron a2c5e6cd5 — Data preprocessing testing changes + fixes
# tools/merge_datasets.py: tool to merge multiple dataset files into a single
# dataset. Adapted from Megatron-LM; import path adjusted for DeepSpeed repo.

import os
import sys
import json
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from deepspeed.runtime.data_pipeline.data_sampling import indexed_dataset

print('[M1171]')


def main(args):

    prefixes = set()
    for basename in os.listdir(args.input):
        prefix, ext = os.path.splitext(basename)

        if prefix in prefixes:
            continue

        if not os.path.isfile(os.path.join(args.input, basename)):
            continue

        ext_pair = '.bin' if ext == '.idx' else '.idx'
        assert os.path.isfile(os.path.join(args.input, prefix) + ext_pair), \
               f'ERROR: {ext_pair} file not provided for {os.path.join(args.input, prefix)}'

        prefixes.add(prefix)

    builder = None
    for prefix in sorted(prefixes):
        if builder is None:
            dataset = indexed_dataset.make_dataset(os.path.join(args.input, prefix), 'infer')

            if isinstance(dataset, indexed_dataset.MMapIndexedDataset):
                builder = indexed_dataset.MMapIndexedDatasetBuilder(args.output_prefix + '.bin', dtype=dataset._index.dtype)
            else:
                builder = indexed_dataset.IndexedDatasetBuilder(args.output_prefix + '.bin')

            del dataset

        builder.merge_file_(os.path.join(args.input, prefix))

    builder.finalize(args.output_prefix + '.idx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to directory containing all document files to merge')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    args = parser.parse_args()

    assert os.path.isdir(args.input), \
           f'ERROR: {args.input} is not a directory or does not exist'

    assert os.path.isdir(os.path.dirname(args.output_prefix)), \
           f'ERROR: {os.path.dirname(args.output_prefix)} is not a directory or does not exist'

    main(args)
