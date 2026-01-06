# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines project-wide constants."""

NUM_IMAGE_TRAIN = 782
NUM_IMAGE_VALID = 98
NUM_IMAGE_TEST = 1
IMAGE_SHAPE = (256, 496, 3)

IS_CHANNEL_WISED = False  # RGB False; Gray True
CHUNK_SHAPE_2D = (16, 16)  # 图像块的高度和宽度

CHUNK_SIZE_BYTES = CHUNK_SHAPE_2D[0] * CHUNK_SHAPE_2D[1]
CHUNK_SIZE_BYTES *= 1 if IS_CHANNEL_WISED else 3
PATCHES_PER_IMAGE = (IMAGE_SHAPE[0] // CHUNK_SHAPE_2D[0]) * (IMAGE_SHAPE[1] // CHUNK_SHAPE_2D[1])
PATCHES_PER_IMAGE *= 3 if IS_CHANNEL_WISED else 1

NUM_CHUNKS_TRAIN = PATCHES_PER_IMAGE * NUM_IMAGE_TRAIN
NUM_CHUNKS_VALID = PATCHES_PER_IMAGE * NUM_IMAGE_VALID
NUM_CHUNKS_TEST = PATCHES_PER_IMAGE * NUM_IMAGE_TEST

ALPHABET_SIZE = 256
ARITHMETIC_CODER_BASE = 2
ARITHMETIC_CODER_PRECISION = 32
