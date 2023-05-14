# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import logging
import json
import glob
import argparse
import math
import random
from tqdm import tqdm, trange
import pickle
import numpy as np
import torch

from GCNwithUnilm import chatbotModel
from modeling_unilm import UnilmConfig
from nltk.stem import WordNetLemmatizer
from tokenization_unilm import UnilmTokenizer
from dataset import DEFAULT_VOCAB, UNK_IDX, PAD_IDX, UNK_IDX, MASK_IDX, SEP_IDX, CLS_IDX, NAF_TRIPLE, get_dataloader, \
    commondataset
import logging
from collections import OrderedDict
from log import Logger
import time


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
                  for conf in (UnilmConfig,)), ())

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_attention_mask(l_src, l_tgt):
    """计算seq2seq的mask矩阵"""
    seq_len = l_src + l_tgt
    input_mask = torch.zeros(seq_len, seq_len, dtype=torch.long)
    input_mask[:, :l_src].fill_(1)
    second_st, second_end = l_src, l_src + l_tgt
    _tril_matrix = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    input_mask[second_st:second_end, second_st:second_end].copy_(
        _tril_matrix[:second_end - second_st, :second_end - second_st])
    return input_mask


def get_dup_ngram_candidates(seq, n):
    forbid_ignore_set = []
    cands = set()
    if len(seq) < n:
        return []
    tail = seq[-(n - 1):]
    for i in range(len(seq) - (n - 1)):
        mismatch = False
        for j in range(n - 1):
            if tail[j] != seq[i + j]:
                mismatch = True
                break
        if not mismatch:
            cands.add(seq[i + n - 1])
    return list(sorted(cands))


def gen_sent(data, dataset, model, args, ngram_size=3):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    src_cat_tgt = data['src_cat_tgt'].to(args.device)
    src_length = data['src_length']
    tgt_length = data['tgt_length']
    token_type = data['token_type'].to(args.device)
    _graph = data['graph'][0]
    csk_ids = _graph.ndata['h'][:, [0, 3]]
    csk_words = set([dataset.id2word[int(ids)] for tri in csk_ids for ids in tri])

    device = args.device
    topk = args.topk
    tgt = ' '
    token_ids = list(src_cat_tgt[0, :src_length])
    target_response = dataset.tokenizor.decode(src_cat_tgt[0, src_length + 1:src_length + tgt_length - 2])
    forbid_word_mask = None
    token_type = list(token_type[0, :src_length])
    # 候选答案id
    target_ids = [[] for _ in range(topk)]
    # 候选答案分数
    target_scores = [0] * topk
    # 强制要求输出不超过max_output_len字
    model.eval()

    with torch.no_grad():
        for i in range(args.max_output_len):
            _token_ids = [token_ids + t + [MASK_IDX] for t in target_ids]
            _segment_ids = [token_type + [5] * (len(t) + 1) for t in target_ids]
            encoder_mask = torch.stack([compute_attention_mask(src_length, i + 1) for _ in target_ids]).to(args.device)
            decoder_mask = torch.stack([compute_attention_mask(src_length + 4, i + 1) for _ in target_ids]).to(
                args.device)
            input_ids = torch.tensor(_token_ids, dtype=torch.long).to(args.device)
            _segment_ids = torch.tensor(_segment_ids, dtype=torch.long).to(args.device)
            batch_index = torch.tensor(list(range(topk))).to(args.device)
            graph = [copy.deepcopy(_graph).to(args.device) for _ in range(topk)]
            outputs = model(graph, input_ids, _segment_ids, encoder_mask, decoder_mask)

            _probas = outputs[:, -1, :]
            # print(_probas)
            # 取对数，方便计算
            # _log_probas = np.log(_probas + 1e-6)
            _log_probas = _probas.cpu()
            if forbid_word_mask is not None:
                _log_probas += (forbid_word_mask * -10000.0)
            _log_probas = _log_probas.detach().numpy()
            # 每一项选出topk
            _topk_arg = _log_probas.argsort(axis=-1)[:, -topk:]
            _candidate_ids, _candidate_scores = [], []
            if args.forbid_duplicate_ngrams:
                if len(target_ids[0]) >= ngram_size:
                    dup_cands = []
                    for seq in target_ids:
                        dup_cands.append(get_dup_ngram_candidates(seq, ngram_size))
                    buf_matrix = None
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = args.vocab_size
                            buf_matrix = np.zeros(
                                (topk, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=int)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [topk, vocab_size])

            for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
                # 预测第一个字的时候，输入的topk事实上都是同一个，
                # 所以只需要看第一个，不需要遍历后面的。
                if i == 0 and j > 0:
                    continue
                for k in _topk_arg[j]:
                    _candidate_ids.append(ids + [k])
                    _candidate_scores.append(sco + _log_probas[j][k])
            _topk_arg = np.argsort(_candidate_scores)[-topk:]
            for j, k in enumerate(_topk_arg):
                # target_ids[j].append(_candidate_ids[k][-1])
                target_ids[j] = _candidate_ids[k]
                target_scores[j] = _candidate_scores[k]
            ends = [j for j, k in enumerate(target_ids) if k[-1] == 102]

            if len(ends) > 0:
                k = np.argmax([target_scores[j] for j in ends])
                return (dataset.tokenizor.decode(token_ids), dataset.tokenizor.decode(target_ids[ends[k]]), csk_words,
                        target_response)

    # 如果max_output_len字都找不到结束符，直接返回
    return (
        dataset.tokenizor.decode(token_ids), dataset.tokenizor.decode(target_ids[np.argmax(target_scores)]), csk_words,
        target_response)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    # parameters
    parser.add_argument("--data_path", default='data/testset.csv', type=str, required=False, )
    parser.add_argument("--csk_dict_path", default='data/id2triples.pkl', type=str, required=False, )

    # parser.add_argument("--model_name_or_path",default='/Users/zhoup/develop/NLPSpace/my-pre-models/chinese_wwm_pytorch', type=str,required=False, )
    parser.add_argument("--output_txt_dir", default='output.txt', type=str, required=False, )
    # Other parameters
    parser.add_argument("--cache_dir", default="", type=str, )
    parser.add_argument("--max_input_len", default=300, type=int, help="最长输入长度")
    parser.add_argument("--max_output_len", default=100, type=int, help="最长输出长度")
    parser.add_argument("--max_triple_len", default=50, type=int, help="最大三元组数量")
    parser.add_argument("--forbid_duplicate_ngrams", default=False, action="store_true", help="forbid duplicate words")
    parser.add_argument("--topk", default=8, type=int, help="beam search参数")
    parser.add_argument("--topp", default=0.95, type=float, help="核采样参数")
    parser.add_argument("--do_train", default=True, action="store_true", help="是否fine tuning")
    parser.add_argument("--do_show", default=True, action="store_true", )
    parser.add_argument("--batch_size", default=1, type=int, )
    parser.add_argument("--vocab_size", default=28996, type=int, )
    parser.add_argument("--seed", type=int, default=42, help="初始化随机种子")
    parser.add_argument("--embedding_size", default=768, type=int, help="嵌入向量大小", )
    parser.add_argument("--distributed", default=False, type=bool, help="分布式", )
    parser.add_argument("--word2id_path", default='data/word2id.txt', type=str, help="word2id 位置", )
    parser.add_argument("--rel2word_path", default='data/rel2words.txt', type=str,
                        help="rel2word 位置", )
    parser.add_argument('--recover_path', type=str,
                        default=None,)
    parser.add_argument("--num_workers", default=0, type=int, help="worker 数量", )
    parser.add_argument("--log_path", default='logs.log', type=str, help="日志文件地址", )
    parser.add_argument("--overwrite_output_dir", default=True, type=bool, help="", )
    parser.add_argument("--num_attention_graphs", default=1, type=int, help="使用图数量", )

    parser.add_argument("--model_name_or_path", default='external_models/unilm/pytorch_model.bin', type=str,
                        required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: ")

    parser.add_argument("--mask_prob", default=0.3, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument("--mask_prob_eos", default=0, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=100,
                        help="Max tokens of prediction.")
    parser.add_argument('--mask_source_words', action='store_true',
                        help="Whether to mask source words for training")
    parser.add_argument("--config_name", default="configuration_unilm.py", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="external_models/unilm/vocab.txt", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    unilmconfig = UnilmConfig()
    model = chatbotModel(unilmconfig)
    checkpoint = torch.load(
        args.recover_path,
        map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print("state dict loaded")
    tokenizor = UnilmTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=True)
    # test_loader = get_dataloader(args, data_name='test', batch_size=args.batch_size, num_workers=args.num_workers)
    print(device)
    model.to(device)
    dataset = commondataset(args, tokenizor, 'train')
    test_loader = get_dataloader(dataset, batch_size=1)

    generate_txt = open(args.output_txt_dir, 'w')
    for index, data in enumerate(test_loader):
        # print(data['graph'][0].ndata)
        a, b, csk_words, tgt_response = gen_sent(data, dataset, model, args,ngram_size=2)
        if index % 1000 == 0:
            print(index)
        # print(a)
        # print(a, '生成回复:', b, flush=True)
        generate_txt.write(b + '\n')

    print('finished', flush=True)


if __name__ == "__main__":
    main()
