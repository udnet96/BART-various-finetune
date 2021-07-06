# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter
import numpy as np


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        do_fairseq_MHA = False
        # print('q, k, v : ', query.shape, key.shape, value.shape)
        if (
                not self.onnx_trace
                and not is_tpu  # don't use PyTorch version on TPUs
                and incremental_state is None
                and not static_kv
                # A workaround for quantization to work. Otherwise JIT compilation
                # treats bias in linear module as method.
                and not torch.jit.is_scripting()
                and not do_fairseq_MHA
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # Q * K^T
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )  # softmax (Q*K^T / root(d_k))
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        # from ud..
        # lower attn value token masking
        """
        # masking : attn_prob dim 2 (key) wrt dim 1 (query) / for all batch items.
        if self.training:
            lower_masking_prob = 0.01
            attn_mask_ud = attn_probs.sort().values
            attn_mask_ud = attn_mask_ud[:, :, int(attn_probs.size(2) * lower_masking_prob)]
            attn_mask_ud = attn_mask_ud.unsqueeze(2)
            attn_mask_ud = attn_mask_ud.expand(attn_probs.size())
            attn_probs = torch.where(attn_probs > attn_mask_ud, attn_probs,
                                       torch.tensor(0, dtype=torch.half, device='cuda'))
        """
        # from ud..
        # higher n % att shuffle
        """
        if self.training:
            shuffle_prob = 0.1
            # idx_upper = attn_probs.sort().indices[:, :, int(attn_probs.size(2) * (1 - shuffle_prob)):]
            idx_upper = attn_probs.sort().indices[:, :, int(attn_probs.size(2) * (1 - shuffle_prob)):]
            # idx_upper = idx_upper_orig.clone().detach()
            np.random.shuffle(idx_upper)
            idx_upper = torch.cat((attn_probs.sort().indices[:, :, :int(attn_probs.size(2) * (1 - shuffle_prob))],
                                   idx_upper), dim=2)
            # lower, upper divide and (lower, upper shuffle) concat

            idx_upper = torch.gather(attn_probs, 2, idx_upper).to('cuda')
            idx_upper = torch.gather(idx_upper, 2, attn_probs.sort().indices.argsort()).to('cuda')
            # for check shuffling
            
            d = np.array((idx_upper == attn_probs).to('cpu'))
            unique, counts = np.unique(d, return_counts=True)
            dict(zip(unique, counts))
            print('shuffled rate : ', counts[0] / (counts[0] + counts[1]))
            
            attn_probs = idx_upper.clone().detach()
            del idx_upper
        """
        # from ud..
        # attn value random shuffling
        """
        if self.training:
            shuffle_prob = 0.2
            shuffle_num = int(attn_probs.size(2) * shuffle_prob)
            ran = set()
            while len(ran) < shuffle_num:
                ranone = np.random.randint(attn_probs.size(2))
                ran.add(ranone)
            ran = list(ran)
            ran.sort()
            before_shuffle = ran
            np.random.shuffle(ran)
            after_shuffle = ran
            idxs = []
            cnt = 0
            for i in range(attn_probs.size(2)):
                if i in before_shuffle:
                    idxs.append(after_shuffle[cnt])
                    cnt += 1
                else:
                    idxs.append(i)
            idx_upper = attn_probs[:, :, idxs]
            v = v[:, idxs, :]

            # for debugging..
            cnt = 0
            for i in range(attn_probs.size(2)):
                if idxs[i] != i:
                    cnt += 1
            shuffled_rate = cnt / attn_probs.size(2)
            print(shuffled_rate)
            # for check shuffling
            d = np.array((idx_upper == attn_probs))
            unique, counts = np.unique(d, return_counts=True)
            dict(zip(unique, counts))
            print('shuffled rate : ', counts[0] / (counts[0] + counts[1]))

            attn_probs = idx_upper.clone().detach()
            del idx_upper
        """
        # from ud..
        # higher attn-value pair random shuffling
        """
        if self.training:
            shuffle_prob = 0.05
            idx_upper = attn_probs.sort().indices[:, :, int(attn_probs.size(2) * (1 - shuffle_prob)):]

            # higher idx shuffling
            idxs = torch.tensor([])
            for i in range(idx_upper.size(1)):
                idxs = torch.cat((idxs, torch.randperm(idx_upper.size(2)).unsqueeze(0)),
                                 dim=0)
            idxs = idxs.expand(idx_upper.size(0), -1, -1).type(torch.int64).to('cuda')
            idx_upper = torch.gather(idx_upper, 2, idxs)
            del idxs
            # idx_upper1 = torch.cat((attn_probs.sort().indices[:, :, :int(attn_probs.size(2) * (1 - shuffle_prob))],
            #                        idxs_before_shuffle), dim=-1)
            # d = (idxs_before_shuffle == idxs_after_shuffle)  # shuffle checking
            idx_upper = torch.cat(
                (attn_probs.sort().indices[:, :, :int(attn_probs.size(2) * (1 - shuffle_prob))],
                 idx_upper), dim=-1)
            idx_upper2 = torch.gather(attn_probs, 2, idx_upper)
            idx_upper2 = torch.gather(idx_upper2, 2, attn_probs.sort().indices.argsort())  # sorted -> orig

            # v sort
            v_concat1 = torch.tensor([]).to('cuda')
            for i in range(idx_upper.size(0)):
                v_concat2 = torch.tensor([]).to('cuda')
                for j in range(idx_upper.size(1)):
                    v_temp = v[i, idx_upper[i, j, :], :].unsqueeze(0).to('cuda')
                    # v_temp = v[i, :, :].unsqueeze(0)
                    v_concat2 = torch.cat((v_concat2, v_temp), dim=0)
                v_concat1 = torch.cat((v_concat1, v_concat2.unsqueeze(0)), dim=0)
            del v_concat2, idx_upper, v

            # v restore (to shuffled idxs)
            id2 = attn_probs.sort().indices.argsort()
            v_concat5 = torch.tensor([]).to('cuda')
            for i in range(id2.size(0)):
                v_concat4 = torch.tensor([]).to('cuda')
                for j in range(id2.size(1)):
                    v_temp = v_concat1[i, j, id2[i, j, :], :].unsqueeze(0).to('cuda')
                    v_concat4 = torch.cat((v_concat4, v_temp), dim=0)
                v_concat5 = torch.cat((v_concat5, v_concat4.unsqueeze(0)), dim=0)
            del v_concat1, v_concat4, attn_probs

            # for check v shuffling
            # just expanded v for shuffle check 
            v_expanded = v.unsqueeze(dim=1)
            v_expanded = v_expanded.expand(-1, attn_probs.size(1), -1, -1)  # (bsz*numhead, tgt, src, headdim)
            d = np.array((v_expanded == v_concat5))
            unique, counts = np.unique(d, return_counts=True)
            dict(zip(unique, counts))
            print('v shuffled rate : ', counts[0] / (counts[0] + counts[1]))

            d = np.array((attn_probs == attn_probs_shuffled))
            unique, counts = np.unique(d, return_counts=True)
            dict(zip(unique, counts))
            print('attn shuffled rate : ', counts[0] / (counts[0] + counts[1]))

            # lower, upper divide and (lower, upper shuffle) concat
            attn_probs3 = torch.tensor([]).to('cuda')
            for i, atten in enumerate(idx_upper2):
                attn_probs2 = torch.tensor([]).to('cuda')
                for j, att in enumerate(atten):
                    attn_probs2 = torch.cat(
                        (attn_probs2,
                         torch.bmm(att.unsqueeze(0).unsqueeze(0).to(torch.float16),
                                   v_concat5[i, j].unsqueeze(0).to(torch.float16))[0]
                         ), dim=0).to(torch.float16)
                attn_probs3 = torch.cat((attn_probs3, attn_probs2.unsqueeze(0)), dim=0)
            attn = attn_probs3.to(torch.float16)
            del attn_probs3, attn_probs2, v_concat5, idx_upper2
        else:
            assert v is not None
            attn = torch.bmm(attn_probs, v)
        """

        assert v is not None
        attn = torch.bmm(attn_probs, v)  # softmax (Q*K^T / root(d_k)) v
        # print('q, k.tran(1,2), attn_weights, : ', q.shape, k.transpose(1, 2).shape, attn_weights.shape)
        # print('attn_probs, v, attn : ', attn_probs.shape, v.shape, attn.shape)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)  # Feed Forward

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                            0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
            self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
            self,
            incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
            buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                                                           dim: 2 * dim
                                                           ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value
