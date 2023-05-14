import copy
import json
import os
import math
import random
from random import random as rand

import pickle
import numpy as np
import torch
import pandas as pd

import dgl
import nltk
import jsonlines
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from ast import literal_eval
from preprocess import save_json, load_json, reverse_dict_key_val

torch.multiprocessing.set_sharing_strategy('file_system')

DEFAULT_VOCAB = ['[PAD]', '[UNK]', '[MASK]', '[SEP]', '[CLS]']
PAD_IDX, UNK_IDX, MASK_IDX, SEP_IDX, CLS_IDX = 0, 100, 103, 102, 101
NAF_TRIPLE = [UNK_IDX, UNK_IDX, UNK_IDX, UNK_IDX]


class commondataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizor, data_name='train'):
        assert data_name in ['train', 'test', 'valid'], "Data name should be among ['train', 'test', 'valid']."
        self.args = args
        self.data_name = data_name
        self.ent2id = load_json(args.word2id_path)
        self.rel2word = load_json(args.rel2word_path)
        self.batch_size = args.batch_size
        self.tokenizor = tokenizor
        self.csk_dict = load_json(args.csk_dict_path)
        self.max_input_len = args.max_input_len
        self.max_output_len = args.max_output_len
        nltk.download('stopwords')
        self.stopwords = stopwords.words('english')
        self.convdata = self.init_convdata()
        print(f'There are {len(self.convdata)} conversations in total.', flush=True)

    def _is_special(self, c):
        return bool(c) and (c[0] == '[')

    def decode(self, token_ids, join_str=''):
        tokens = []
        for i in token_ids:
            t = self.tokenizor.convert_ids_to_tokens(i)
            if t == '[SEP]':
                tokens.append('.')
                break
            elif not self._is_special(t):
                tokens.append(t)
                tokens.append(' ')
        return join_str.join(tokens)

    def init_convdata(self):
        print('loading conversation data', flush=True)

        data_list = []
        with self.args.data_path.open("r", encoding="utf-8") as fr:
            for line in fr:
                one_data = json.loads(line.strip())
                data_list.append(one_data)
        print(f"data from {self.args.data_path} size: {len(data_list)}")

        return data_list

    def retrieve_graph(self, words, max_triple_len=50):
        key = words.lower()
        if key in self.com_diction and key not in self.stopwords:
            graphs = self.com_diction[key]
            if len(graphs) >= max_triple_len:
                graphs.sort(key=lambda t: t[3])
                graphs = graphs[:max_triple_len]
            ret_graphs = []
            h = None
            if graphs[0][0] in self.ent2id:
                h = self.ent2id[graphs[0][0]]
            for g in graphs:
                if g[1] in self.ent2id and g[2] in self.rel2word:
                    t = self.ent2id[g[1]]
                    r = self.rel2word[g[2]]
                    if h is not None:
                        ret_graphs.append([h, t, self.ent2id[r[0]], self.ent2id[r[1]]])
            if len(ret_graphs) == 0:
                return [NAF_TRIPLE], False
            return ret_graphs, True
        else:
            return [NAF_TRIPLE], False

    def process_triple(self, all_triples, target_triples):
        src_triple = []
        target_head = []
        target_tail = []
        main_entity = []
        for triple_list in all_triples:
            processed_triple = []
            for t in triple_list:
                triple = self.csk_dict[t].split(', ')
                head = triple[0]
                rel = triple[1]
                tail = triple[2]
                rel = self.rel2word[rel].split()
                if head in self.tokenizor.vocab and tail in self.tokenizor.vocab:
                    processed_triple.append(
                        [self.tokenizor.convert_tokens_to_ids(head), self.tokenizor.convert_tokens_to_ids(rel[0]), self.tokenizor.convert_tokens_to_ids(rel[1]), self.tokenizor.convert_tokens_to_ids(tail)])
            src_triple.append(processed_triple)
            entities = [p for triple in processed_triple for p in triple]
            if len(entities) > 0:
                entity = max(set(entities), key=entities.count)
            else:
                entity = None
            main_entity.append(entity)
        for t in target_triples:
            triple = self.csk_dict[t].split(', ')
            head = triple[0]
            tail = triple[2]
            if head in self.tokenizor.vocab and tail in self.tokenizor.vocab:
                target_head.append(self.tokenizor.convert_tokens_to_ids(head))
                target_tail.append(self.tokenizor.convert_tokens_to_ids(tail))
        target_ent = target_head + target_tail
        return src_triple, main_entity, target_ent

    def make_graph(self, all_triples, main_entity, target_ent):
        word2nodeid = {}
        nodeid2wordid = {}
        edges = []
        target_node = []
        embedding = []
        #add a root node with 0 embeddings
        embedding.append([PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX])
        word2nodeid['root'] = len(word2nodeid)
        nodeid2wordid[word2nodeid['root']] = 'root'
        target_node.append(1)
        for h_index, triples in enumerate(all_triples):
            main_ent = main_entity[h_index]
            if main_ent is UNK_IDX:
                continue
            if main_ent not in word2nodeid:
                word2nodeid[main_ent] = len(word2nodeid)
                e = [PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX]
                embedding.append(e)
                nodeid2wordid[word2nodeid[main_ent]] = main_ent
                if main_ent in target_ent:
                    target_node.append(1)
                else:
                    target_node.append(0)
            edges.append([0, word2nodeid[main_ent]])

            for t_index, t in enumerate(triples):
                # define the attribute of the tail nodes and connect them to the head node
                if str(t) not in word2nodeid:
                    word2nodeid[str(t)] = len(word2nodeid)
                    embedding.append(t)
                    nodeid2wordid[word2nodeid[str(t)]] = str(t)
                    if t[0] in target_ent and t[-1] in target_ent:
                        target_node.append(1)
                    else:
                        target_node.append(0)
                edges.append([word2nodeid[main_ent], word2nodeid[str(t)]])
        # target_ent = target_head+target_tail
        if len(edges) > 0:
            edges = torch.tensor(edges)
            g = dgl.graph((edges[:, 1], edges[:, 0]), num_nodes=len(word2nodeid))
            g.ndata['h'] = torch.tensor(embedding).detach()
            g.ndata['_ID'] = torch.Tensor(np.array(range(len(word2nodeid)))).long()
            tar = torch.tensor(target_node, dtype=torch.float32)
        else:
            edges = [[0, 1]]
            edges = torch.tensor(edges)
            g = dgl.graph((edges[:, 1], edges[:, 0]), num_nodes=2)
            g.ndata['h'] = UNK_IDX * torch.ones((2, 4 * self.args.embedding_size))
            g.ndata['_ID'] = torch.Tensor(np.array(range(2))).long()
            tar = torch.tensor([0, 0], dtype=torch.float32)
        g.ndata['tar'] = tar
        g.ndata['loss'] = torch.zeros_like(tar)
        g.ndata['tar_num'] = torch.zeros_like(tar)

        return g

    def load_obj(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_random_word(self):
        i = random.randint(0, len(self.tokenizor.vocab) - 1)
        return self.tokenizor.convert_ids_to_tokens(i)

    def mask(self, src, tgt, response_target):
        tokens = src + tgt
        segment_ids = [4] * (len(src)) + [5] * (len(tgt))
        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tgt) - 1
        n_pred = min(self.args.max_pred, max(1, int(round(effective_length * self.args.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(src)) and (tk != '[CLS]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        random.shuffle(cand_pos)

        triple_list = [self.csk_dict[item].split(', ') for item in response_target if item != -1]
        ent_list = [e for tri in triple_list for e in tri]
        ent_pos = [i + len(src) for i, item in enumerate(tgt) if item in ent_list]
        # masked_pos = set(ent_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue
            st_pos, end_pos = pos, pos + 1
            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            random.shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = self.get_random_word()
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1] * len(masked_tokens)

        # Token Indexing
        masked_ids = self.tokenize(masked_tokens)
        # Token Indexing
        input_ids = self.tokenize(tokens)

        # Zero Padding
        n_pad = self.max_input_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        encoder_mask = torch.zeros(self.max_input_len, self.max_input_len, dtype=torch.long)
        encoder_mask[:, :len(src)].fill_(1)
        second_st, second_end = len(
            src), len(src) + len(tgt)
        _tril_matrix = torch.tril(torch.ones(
            (self.max_input_len, self.max_input_len), dtype=torch.long))
        encoder_mask[second_st:second_end, second_st:second_end].copy_(
            _tril_matrix[:second_end - second_st, :second_end - second_st])

        decoder_len = self.max_input_len + self.args.num_attention_graphs * 4
        decoder_mask = torch.zeros(decoder_len, decoder_len, dtype=torch.long)
        decoder_mask[:, :len(src) + 4 * self.args.num_attention_graphs].fill_(1)
        second_st, second_end = len(src) + 4 * self.args.num_attention_graphs, len(src) + len(
            tgt) + 4 * self.args.num_attention_graphs
        _tril_matrix = torch.tril(torch.ones(
            (decoder_len, decoder_len), dtype=torch.long))
        decoder_mask[second_st:second_end, second_st:second_end].copy_(
            _tril_matrix[:second_end - second_st, :second_end - second_st])

        # Zero Padding for masked target
        if self.args.max_pred > n_pred:
            n_pad = self.args.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0] * n_pad)
            if masked_pos is not None:
                masked_pos.extend([0] * n_pad)
            if masked_weights is not None:
                masked_weights.extend([0] * n_pad)
        return (
            torch.tensor(input_ids), torch.tensor(segment_ids), encoder_mask, decoder_mask, torch.tensor(masked_ids),
            torch.tensor(masked_pos), torch.tensor(masked_weights))

    def mask_eval(self, src, tgt):
        tokens = src + tgt
        segment_ids = [4] * (len(src)) + [5] * (len(tgt))
        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tgt) - 1
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(src)) and (tk != '[CLS]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)

        masked_pos = list(cand_pos)

        masked_tokens = [[tokens[pos]] for pos in masked_pos]
        batch_tokens = []
        for pos in masked_pos:
            tokens_copy = tokens.copy()
            tokens_copy[pos] = '[MASK]'
            batch_tokens.append(tokens_copy)

        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [[1]] * len(batch_tokens)

        # Token Indexing
        masked_ids = [self.tokenize(_) for _ in masked_tokens]
        # Token Indexing
        input_ids = [self.tokenize(t) for t in batch_tokens]

        # Zero Padding
        n_pad = self.max_input_len - len(input_ids[0])
        batch_ids = []
        for ids in input_ids:
            ids.extend([0] * n_pad)
            batch_ids.append(ids)
        segment_ids.extend([0] * n_pad)

        encoder_mask = torch.zeros(self.max_input_len, self.max_input_len, dtype=torch.long)
        encoder_mask[:, :len(src)].fill_(1)
        second_st, second_end = len(
            src), len(src) + len(tgt)
        _tril_matrix = torch.tril(torch.ones(
            (self.max_input_len, self.max_input_len), dtype=torch.long))
        encoder_mask[second_st:second_end, second_st:second_end].copy_(
            _tril_matrix[:second_end - second_st, :second_end - second_st])

        decoder_len = self.max_input_len + self.args.num_attention_graphs * 4
        decoder_mask = torch.zeros(decoder_len, decoder_len, dtype=torch.long)
        decoder_mask[:, :len(src) + 4 * self.args.num_attention_graphs].fill_(1)
        second_st, second_end = len(src) + 4 * self.args.num_attention_graphs, len(src) + len(
            tgt) + 4 * self.args.num_attention_graphs
        _tril_matrix = torch.tril(torch.ones(
            (decoder_len, decoder_len), dtype=torch.long))
        decoder_mask[second_st:second_end, second_st:second_end].copy_(
            _tril_matrix[:second_end - second_st, :second_end - second_st])

        # Zero Padding for masked target
        batch_size = len(masked_pos)
        masked_pos = [[pos] for pos in masked_pos]
        # print(batch_ids, torch.tensor(segment_ids).repeat(batch_size), encoder_mask,decoder_mask, torch.tensor(masked_ids),masked_pos, masked_weights)
        return (
            torch.tensor(batch_ids), torch.tensor(segment_ids).repeat(batch_size, 1),
            encoder_mask.repeat(batch_size, 1, 1),
            decoder_mask.repeat(batch_size, 1, 1), torch.tensor(masked_ids),
            torch.tensor(masked_pos), torch.tensor(masked_weights))

    def tokenize(self, src_ids):
        ids = []
        for s in src_ids:
            if s not in DEFAULT_VOCAB:
                s = s.lower()
            id = self.tokenizor.convert_tokens_to_ids(s)
            ids.append(id)
        return ids

    def pad_1d(self, list, length, pad_idx=PAD_IDX):
        """ Pad over axis 0. """
        return np.pad(list, (0, length - len(list)), 'constant', constant_values=pad_idx)

    def __len__(self):
        return len(self.convdata)

    def __getitem__(self, i):
        data = self.convdata[i]
        # print(data)
        src, tgt, match_triples, all_triples, response_target = literal_eval(data['post']), literal_eval(
            data['response']), literal_eval(data['match_triples']), literal_eval(data['all_triples']), literal_eval(
            data['response_triples'])
        src = TreebankWordDetokenizer().detokenize(src)
        tgt = TreebankWordDetokenizer().detokenize(tgt)
        src = self.tokenizor.tokenize(src)
        tgt = self.tokenizor.tokenize(tgt)

        if self.data_name == 'train':
            src = ['[CLS]'] + src + ['[SEP]']
            tgt = tgt + ['[SEP]']
            src_cat_tgt, token_type, encoder_mask, decoder_mask, masked_ids, masked_pos, masked_weights = self.mask(src,
                                                                                                                    tgt,
                                                                                                                    response_target)
            src_len = len(src)
            tgt_len = len(tgt)
            triples, main_entity, target_ent = self.process_triple(all_triples, match_triples)
            graph = self.make_graph(triples, main_entity, target_ent)
        else:
            src = ['[CLS]'] + src + ['[SEP]']
            tgt = tgt + ['[SEP]']
            src_cat_tgt, token_type, encoder_mask, decoder_mask, masked_ids, masked_pos, masked_weights = self.mask_eval(
                src,
                tgt)
            src_len = len(src)
            tgt_len = len(tgt)
            triples, target_head, target_tail = self.process_triple(all_triples, match_triples)
            graph = self.make_graph(triples, target_head, target_tail)
            graph = [copy.deepcopy(graph) for _ in range(src_cat_tgt.shape[0])]

        return {'src_cat_tgt': src_cat_tgt, 'token_type': token_type, 'encoder_mask': encoder_mask,
                'decoder_mask': decoder_mask,
                'masked_ids': masked_ids, 'masked_pos': masked_pos, 'masked_weights': masked_weights,
                'graph': graph, 'src_len': src_len, 'tgt_len': tgt_len}


def pad_2d(list, pad_idx=PAD_IDX):
    ml = max([len(i) for i in list])
    return np.array([i + [pad_idx] * (ml - len(i)) for i in list])


def collate_fn(batch):
    src_length = torch.tensor([s['src_len'] for s in batch])  # (bsz,)
    src_cat_tgt = torch.stack([s['src_cat_tgt'] for s in batch])
    token_type = torch.stack([s['token_type'] for s in batch])

    encoder_mask = torch.stack([s['encoder_mask'] for s in batch])
    decoder_mask = torch.stack([s['decoder_mask'] for s in batch])
    masked_ids = torch.stack([s['masked_ids'] for s in batch])
    masked_pos = torch.stack([s['masked_pos'] for s in batch])
    masked_weights = torch.stack([s['masked_weights'] for s in batch])

    tgt_length = torch.tensor([s['tgt_len'] for s in batch])  # (bsz,)
    graph = [s['graph'] for s in batch]  # (bsz,)

    batched_data = {
        'src_length': src_length,
        'tgt_length': tgt_length,
        'src_cat_tgt': src_cat_tgt.long(),
        'token_type': token_type.long(),
        'encoder_mask': encoder_mask.long(),
        'decoder_mask': decoder_mask.long(),
        'masked_ids': masked_ids.long(),
        'masked_pos': masked_pos.long(),
        'masked_weights': masked_weights.long(),
        'graph': graph
    }
    return batched_data


def get_dataloader(dataset,
                   batch_size=64,
                   shuffle=True,
                   num_workers=0):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              worker_init_fn=lambda _: np.random.seed(),
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              drop_last=True,
                                              )
    return data_loader


def get_distributed_data_loader(dataset,
                                train_sampler,
                                batch_size,
                                num_workers=0):
    data_loader = torch.utils.data.DataLoader(dataset,
                                              sampler=train_sampler,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              drop_last=True,
                                              )
    return data_loader
