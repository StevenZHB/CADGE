import sys

sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import dgl
import os
import numpy as np
import sys
import pickle
sys.path.append(".")

from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from modeling_unilm import LabelSmoothingLoss, BertEncoder, UnilmPreTrainedModel, UnilmModel, UnilmModelIncr, \
    BertDecoder, UnilmConfig
from graph_attention import GATlayer

BertLayerNorm = nn.LayerNorm


class chatbotModel(UnilmPreTrainedModel):
    def __init__(self, config):
        super(chatbotModel, self).__init__(config)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.bert = UnilmModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None
        self.init_weights()
        self.tie_weights()
        self.graph_fc1 = nn.Linear(config.graph_in_feat, config.graph_out_feat, bias=False)
        self.graph_model = nn.ModuleList([GATlayer(config.graph_in_feat, config.graph_out_feat, config.hidden_size,
                                                   config.graph_drop, config.leaky_negative_slope) for _ in
                                          range(config.num_attention_graphs)])

    def tie_weights(self):
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_attention_mask = extended_attention_mask.to(
            dtype=torch.float16 if self.config.fp16 else torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, graph, input_ids, token_type_ids, encoder_mask=None, decoder_mask=None,
                masked_lm_labels=None, masked_pos=None, masked_weights=None, num_tokens_a=None, num_tokens_b=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, encoder_mask)

        embedding_output = self.bert.embeddings(
            input_ids, token_type_ids)
        hidden_states = embedding_output
        for l in range(int(self.num_hidden_layers / 2)):
            hidden_states = self.bert.encoder.layer[l](hidden_states, extended_attention_mask)
        sequence_output = hidden_states
        cls = self.bert.pooler(sequence_output)

        for gindex, g in enumerate(graph):
            g.ndata['q'] = cls[gindex].expand((g.num_nodes(), self.config.hidden_size))

        graph_batch = dgl.batch(graph)

        graph_batch.ndata['h'] = self.bert.embeddings(graph_batch.ndata['h']).view(-1, self.config.hidden_size * 4)
        graph_batch.ndata['h'] = self.graph_fc1(graph_batch.ndata['h'])
        for graph_layer in self.graph_model:
            graph_batch = graph_layer(graph_batch)
            graphs = dgl.unbatch(graph_batch)
            knowledge_distill = torch.stack([g.ndata['h'][0].view((4,-1)) for g in graphs])
            ks_loss = sum([g.ndata['loss'][0] / (g.ndata['tar_num'][0]+1) for g in graphs])
            ks_loss = ks_loss/len(graphs)

        hidden_states = torch.cat((knowledge_distill, sequence_output), dim=1)
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, decoder_mask)
        for l in range(int(self.num_hidden_layers / 2), self.num_hidden_layers):
            hidden_states = self.bert.encoder.layer[l](hidden_states, extended_attention_mask)
        sequence_output = hidden_states[:, 4 * self.config.num_attention_graphs:, :]

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        if masked_lm_labels is None:
            if masked_pos is None:
                prediction_scores = self.cls(sequence_output)
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output, masked_pos)
                prediction_scores = self.cls(sequence_output_masked)
            return prediction_scores

        sequence_output_masked = gather_seq_out_by_pos(
            sequence_output, masked_pos)
        prediction_scores_masked = self.cls(sequence_output_masked)
        if self.crit_mask_lm_smoothed:
            masked_lm_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
        else:
            masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
        masked_lm_loss = loss_mask_and_normalize(
            masked_lm_loss.float(), masked_weights)

        return masked_lm_loss, ks_loss
