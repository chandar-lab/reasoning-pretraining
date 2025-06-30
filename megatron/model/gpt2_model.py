# # Copyright (c) 2025, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""GPT-2 model."""

import math
import torch
import torch.nn as nn
from collections import defaultdict

from functools import partial
from megatron.model.utils import Lambda, SequentialWrapper, recursive_setattr
from megatron.model.norms import get_norm
from megatron.model.init_functions import get_init_methods

from megatron import mpu
from megatron.mpu import ParallelRelativePositionBias
from megatron.model.transformer import (
    ParallelTransformerLayerPipe,
    NormPipe,
    ParallelLinearPipe,
    parallel_lm_logits,
    ParallelLinear,
)
from megatron.model.gmlp import GMLPBlock
from megatron.model.rwkv.v6 import RWKVResidualLayerPipe
from megatron.model.mamba import ParallelMambaResidualLayerPipe
from megatron.model.word_embeddings import EmbeddingPipe, SoftEmbedding

# Pipeline parallelism
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec
from typing import Union, List


def gpt2_attention_mask_func(attention_scores, ltor_mask):
    mask_value = torch.finfo(attention_scores.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(
        mask_value, dtype=attention_scores.dtype, device=attention_scores.device
    )
    attention_scores.masked_fill_(ltor_mask, mask_value)
    return attention_scores


def cross_entropy(output, labels, _fp16=False):
    """From pretrain_gpt2:forward_step()"""
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    """
    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert output.dtype == torch.half and loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss


def _pre_transformer_block(args):
    # data format change for hidden_states to avoid explicit tranposes : [b s h] --> [s b h]
    assert len(args) == 2, "Incorrect number of arguments to _pre_transformer_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous(), *_args[1:])
    return fn(args)


def _post_transformer_block(args):
    # from (hidden_states, attention_mask)
    # to (hidden_states.T)
    assert len(args) == 2, "Incorrect number of arguments to _post_transformer_block"
    fn = lambda _args: (_args[0].transpose(0, 1).contiguous())
    return fn(args)

class LatentSpaceLayer(nn.Module):
    def __init__(self, latent_dim):
        super(LatentSpaceLayer, self).__init__()
        self.latent_dim = latent_dim
        self.latent_projection = nn.Linear(latent_dim, latent_dim)  # Projection to latent space

    def forward(self, x):
        return self.latent_projection(x)
    def init_latent_state(self, prefix=None):
        """Initialize latent vectors for reasoning."""
        # For illustration, return zero vectors of the latent_dim
        return torch.zeros((1, self.latent_dim)).to(prefix.device)

    def sample_latent(self, batch_size, device):
        """
        Sample latent vectors from a Gaussian distribution.
        
        :param batch_size: The number of samples to generate.
        :param device: The device to generate the latent vectors on.
        :return: Sampled latent vectors.
        """
        return torch.randn(batch_size, self.latent_dim).to(device)


    def decode_with_latents(self, combined_input):
        """
        Decode the latent representation back to logits.

        :param prefix: The prefix of tokens or initial inputs.
        :param latent_gaussians: The latent space representations.
        :return: logits for token prediction
        """
        # You can concatenate the prefix with the latent representations if needed
        logits = self.decoder(combined_input)  # Use a linear layer to generate logits from the latent vector
        return logits

class GPT2ModelPipe(PipelineModule, torch.nn.Module):
    """GPT2Model adapted for pipeline parallelism."""

    def __init__(
        self,
        neox_args,
        num_tokentypes=0,
        parallel_output=True,
        topology=None,
        use_cache=False,
    ):
        self.neox_args = neox_args

        self.use_cache = use_cache
        self.parallel_output = parallel_output
        self.hidden_size = self.neox_args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method, self.output_layer_init_method = get_init_methods(self.neox_args)
        self.__topology__ = topology

        # Initialize specs first
        self.specs = []
        self.init_specs()  # Initialize the layer specs (basically a fancy nn.Sequential)

        # Initialize the parent class after specs are initialized
        super().__init__(
            layers=self.specs,
            loss_fn=partial(cross_entropy, _fp16=self.neox_args.fp16_lm_cross_entropy),
            topology=topology,
            activation_checkpoint_interval=self.neox_args.checkpoint_num_layers
            if self.neox_args.checkpoint_activations
            else 0,
            partition_method=neox_args.pipe_partition_method,
            checkpointable_layers=[
                "GMLPBlock",
                "ParallelTransformerLayerPipe",
                "ParallelMambaResidualLayerPipe",
            ],
        )

        # Now initialize latent space layer after parent class initialization
        self.latent_dim = neox_args.latent_dim  # Add latent dimension to the config
        self.latent_space_layer = LatentSpaceLayer(self.latent_dim)  # Initialize the latent space layer

        # Insert the latent space layer into the model specs at index 1 (after embedding)
        self.specs.insert(-1, self.latent_space_layer)

    def insert_layers(
        self, layers: Union[nn.Module, nn.ModuleList, nn.Sequential, List], idx
    ):
        """
        inserts the layers in `layers` into the pipe model at `idx`.
        """
        if isinstance(layers, nn.Module):
            self.specs.insert(idx, layers)
        elif any(
            [isinstance(layers, nn.ModuleList), isinstance(layers, nn.Sequential)]
        ):
            self.specs[idx:idx] = layers
        elif isinstance(layers, list):
            assert all(
                [hasattr(l, "__call__") for l in layers]
            ), "all items in `layers` must be Callables"
            self.specs[idx:idx] = layers
        else:
            raise ValueError(
                f"layer passed into {self.__class__.__name__}.insert_layer() should be either an nn.Module, an nn.ModuleList, an nn.Sequential object, or a list of callables not a {type(layers)}"
            )

        # re-initialize parent class
        super().__init__(
            layers=self.specs,
            loss_fn=self.loss_fn,
            topology=self.__topology__,
            activation_checkpoint_interval=self.activation_checkpoint_interval,
            partition_method=self.neox_args.pipe_partition_method,
            checkpointable_layers=[
                "GMLPBlock",
                "ParallelTransformerLayerPipe",
                "ParallelMambaResidualLayerPipe",
                "RWKVResidualLayerPipe",
            ],
        )


    def decode_with_latents(self, prefix, latent_gaussians):
        """
        Decode predicted tokens using latent representations (Gaussians).

        :param prefix: The prefix input tensor (could be token embeddings or hidden states).
        :param latent_gaussians: The latent vectors sampled or generated for decoding.
        :return: Decoded logits (predicted token probabilities/logits).
        """
        x = latent_gaussians
        for layer in self.specs:
            x = layer(x)  
        logits = x 
        return logits

    def init_specs(self):
        weight_tying = not self.neox_args.no_weight_tying
        self.specs = []

        # Embedding layer
        if weight_tying:
            self.specs.append(
                TiedLayerSpec(
                    "embed",
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                    tied_weight_attr="word_embeddings_weight",
                )
            )
        else:
            self.specs.append(
                LayerSpec(
                    EmbeddingPipe,
                    self.neox_args,
                    self.hidden_size,
                    self.neox_args.padded_vocab_size,
                    self.neox_args.max_position_embeddings,
                    self.neox_args.hidden_dropout,
                    self.init_method,
                    self.num_tokentypes,
                )
            )

        self.specs.append(_pre_transformer_block)

        # Transformer layers
        for i in range(self.neox_args.num_layers):
            layer_type = self.neox_args.attention_config[i]
            if layer_type in ["gmlp", "amlp"]:
                self.specs.append(
                    LayerSpec(
                        GMLPBlock,
                        init_method=self.init_method,
                        layer_number=i,
                        output_layer_init_method=self.output_layer_init_method,
                        neox_args=self.neox_args,
                        mask_fn=gpt2_attention_mask_func,
                    )
                )
            else:
                self.specs.append(
                    LayerSpec(
                        ParallelTransformerLayerPipe,
                        neox_args=self.neox_args,
                        attention_mask_func=gpt2_attention_mask_func,
                        init_method=self.init_method,
                        output_layer_init_method=self.output_layer_init_method,
                        layer_number=i,
                    )
                )

        self.specs.append(_post_transformer_block)

        # NormPipe
        norm, eps = get_norm(self.neox_args)
        self.specs.append(
            LayerSpec(NormPipe, norm, self.neox_args.hidden_size, eps=eps)
        )

        # Logits layer
        self.specs.append(
            LayerSpec(
                ParallelLinearPipe,
                neox_args=self.neox_args,
                init_method=self.init_method,
                parallel_output=self.parallel_output,
                is_last_layer=True,
            )
        )

    def _set_parallel_output(self, value):
        # sets the parallel output value of the final layer to value
        final_layer = list(self.forward_funcs)[-1]
        if isinstance(final_layer, (ParallelLinearPipe, ParallelLinear)):
            final_layer.final_linear.set_parallel_output(value)

    def inference_mode(self, use_cache=True):
        """
        Sets up the model for inference by turning on k/v caching (if specified) and setting `parallel output` of the final layer to false,
        so logits are gathered across model parallel ranks.
        """
        # first set caching to true if specified
        recursive_setattr(self.forward_funcs, "use_cache", use_cache, assert_type=bool)
        # then set parallel output of the final layer to false so we don't have to gather the output manually
        self._set_parallel_output(False)
        recursive_setattr(self.forward_funcs, "training", False)

    def train_mode(self):
        """
        Sets up the model for training by turning off k/v caching and setting `parallel output` of the final layer to True,
        so logits are not gathered across model parallel ranks, and loss is computed in parallel (more efficient).
        """
        # set caching to false
        recursive_setattr(self.forward_funcs, "use_cache", False)
        # then set parallel output to true (more efficient training)
        self._set_parallel_output(True)
        recursive_setattr(self.forward_funcs, "training", True)

    def clear_cache(self):
        """
        Recursively clears the kv cache on all layers
        """
        recursive_setattr(self.forward_funcs, "layer_past", None)

    def to_sequential(self):
        """
        Transforms the PipelineModule to a plain nn.Sequential module
        :return:
        """
        layers = []
        tied_layers = defaultdict(list)
        for n, spec in enumerate(self.specs):
            if isinstance(spec, TiedLayerSpec):
                if spec.key in tied_layers:
                    # receiver
                    layers.append(
                        Lambda(lambda x: spec.forward_fn(tied_layers[spec.key][0], x))
                    )
                else:
                    # owner
                    module = spec.build(log=False)
                    layers.append(module)
                    tied_layers[spec.key].append(module)
            elif isinstance(spec, LayerSpec):
                layers.append(spec.build(log=False))
            elif hasattr(spec, "__call__"):
                # check that it's a callable function
                layers.append(Lambda(spec))
            else:
                raise ValueError(f"Layer number {n} ({spec}) Not recognized")
        model = SequentialWrapper(
            layers,
            self.activation_checkpoint_interval,
            self.activation_checkpoint_func,
            parent_class_name=self.__class__.__name__,
        )
        return model
