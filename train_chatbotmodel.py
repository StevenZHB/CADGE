#! -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import pandas as pd
import os, json, codecs, logging, random
from tqdm import trange, tqdm
import numpy as np
import argparse
import pickle

from transformers import AdamW, get_linear_schedule_with_warmup
from modeling_unilm import UnilmConfig
from GCNwithUnilm import chatbotModel
from nltk.stem import WordNetLemmatizer
from tokenization_unilm import UnilmTokenizer, WhitespaceTokenizer
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from dataset import DEFAULT_VOCAB, PAD_IDX, UNK_IDX, MASK_IDX, SEP_IDX, CLS_IDX, NAF_TRIPLE, get_dataloader, \
    commondataset, get_distributed_data_loader
import logging
from log import Logger
import time
from collections import OrderedDict
from decode_local import gen_sent

wlem = WordNetLemmatizer()


# logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def read_text(conv_data_path):
    df = pd.read_csv(conv_data_path)
    src = df['src'].values
    tgt = df['tgt'].values

    for s, t in zip(src, tgt):
        yield t, s


def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])


def padding_triples(x):
    ml = max([len(i) for i in x])
    return [i + [NAF_TRIPLE] * (ml - len(i)) for i in x]


def load_obj(name):
    with open(name) as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--data_path", default='data/trainset.txt', type=str, required=False, )
    parser.add_argument("--csk_dict_path", default='data/id2triple.txt', type=str, required=False, )
    parser.add_argument("--output_dir", default='./output_models', type=str, required=False, )
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    # Other parameters
    parser.add_argument("--max_input_len", default=128, type=int, help="最长输入长度")
    parser.add_argument("--max_output_len", default=100, type=int, help="最长输出长度")
    parser.add_argument("--max_triple_len", default=50, type=int, help="最大三元组数量")
    parser.add_argument("--min_count", default=30, type=int, help="精简掉出现频率少于此的word")
    parser.add_argument("--topk", default=4, type=int, help="beam search参数")
    parser.add_argument("--topp", default=0., type=float, help="核采样参数")
    parser.add_argument("--do_train", default=True, action="store_true", help="是否fine tuning")
    parser.add_argument("--do_show", default=True, action="store_true", )
    parser.add_argument("--do_recover", default=False, action="store_true", )
    parser.add_argument("--batch_size", default=36, type=int, )
    parser.add_argument("--vocab_size", default=28996, type=int, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="学习率衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁减值")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="训练epochs次数", )
    parser.add_argument("--warmup_steps", default=0, type=int, help="学习率线性预热步数")
    parser.add_argument("--logging_steps", type=int, default=1000, help="每多少步打印日志")
    parser.add_argument("--seed", type=int, default=6666, help="初始化随机种子")
    parser.add_argument("--max_steps", default=800000, type=int, help="训练的总步数", )
    parser.add_argument("--save_steps", default=25000, type=int, help="保存的间隔steps", )
    parser.add_argument("--graph_num_rels", default=44, type=int, help="图关系数量", )
    parser.add_argument("--embedding_size", default=768, type=int, help="嵌入向量大小", )
    parser.add_argument("--rel2word_path", default='data/rel2word.txt', type=str, help="rel2word 位置", )
    parser.add_argument("--num_workers", default=4, type=int, help="worker 数量", )
    parser.add_argument("--log_path", default='logs.log', type=str, help="日志文件地址", )
    parser.add_argument("--overwrite_output_dir", default=True, type=bool, help="", )
    parser.add_argument("--num_attention_graphs", default=1, type=int, help="使用图数量", )
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--model_name_or_path", default='external_models/unilm/pytorch_model.bin', type=str,
                        required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--mask_prob", default=0.3, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=100,
                        help="Max tokens of prediction.")
    parser.add_argument("--tokenizer_name", default="external_models/unilm/vocab.txt", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--recover_path', type=str, default=None,
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir) and os.listdir(args.output_dir)
            and args.do_train and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.device = device
    n_gpu = torch.cuda.device_count()
    logger = Logger(args.log_path, level='info', fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s")

    # Set seed
    set_seed(args)
    logger.logger.info("Training/evaluation parameters %s", args)
    tokenizor = UnilmTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=True)
    logger.logger.info(device)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    unilmconfig = UnilmConfig()
    model = chatbotModel(unilmconfig)
    model.from_pretrained(args.model_name_or_path, config=unilmconfig)
    if args.do_recover:
        model_recover = torch.load(args.recover_path, map_location='cpu')
        model.load_state_dict(model_recover['state_dict'], strict=False)

    model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                    find_unused_parameters=True)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    dataset = commondataset(args, tokenizor, 'train')
    if n_gpu > 1:
        train_sampler = DistributedSampler(dataset)
        train_loader = get_distributed_data_loader(dataset, train_sampler, batch_size=args.batch_size,
                                                   num_workers=args.num_workers)
    else:
        train_loader = get_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)

    t_total = int(len(train_loader) * args.num_train_epochs /
                  args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * t_total),
                                                num_training_steps=t_total)
    if args.do_recover:
        optimizer.load_state_dict(model_recover['optimizer'])
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    logger.logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    global_step = 0


    if args.do_train:
        logger.logger.info("***** Running training *****")
        logger.logger.info("  Num epochs = %d", args.num_train_epochs)
        logger.logger.info("  Num steps = %d", args.max_steps)

        model.train()
        for i_epoch in trange(1, int(args.num_train_epochs) + 1, desc="Epoch"):
            if n_gpu > 1:
                train_sampler.set_epoch(i_epoch)
            if args.local_rank == 0:
                print(i_epoch)
            train_dataloader_iterator = tqdm(enumerate(train_loader),
                                             total=len(train_loader)) if args.local_rank == 0 else enumerate(
                train_loader)
            for step, batch in enumerate(train_dataloader_iterator):
                if args.local_rank == 0:
                    print(step)
                model.train()
                input_ids = batch['src_cat_tgt'].cuda()
                token_type = batch['token_type'].cuda()
                graph = batch['graph']
                encoder_mask = batch['encoder_mask'].cuda()
                decoder_mask = batch['decoder_mask'].cuda()
                lm_label_ids = batch['masked_ids'].cuda()
                masked_pos = batch['masked_pos'].cuda()
                masked_weights = batch['masked_weights'].cuda()
                batch_index = torch.tensor(list(range(args.batch_size))).cuda()
                graph = graph[int(batch_index[0]): int(batch_index[-1]) + 1]
                graph = [g.to(batch_index.device) for g in graph]

                # input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, _ = batch
                masked_lm_loss, ks_loss = model(graph, input_ids, token_type, encoder_mask, decoder_mask,
                                                lm_label_ids, masked_pos=masked_pos, masked_weights=masked_weights)
                loss = masked_lm_loss + ks_loss

                if n_gpu > 1:  # mean() to average on multi-gpu.
                    loss = loss.mean()

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1

                if args.local_tank == 0:
                    train_dataloader_iterator.set_postfix(epoch=i_epoch, current_step=step,
                                                          total_step=len(train_loader),
                                                          loss=round(loss.item(), 3), MLMloss=round(masked_lm_loss, 3),
                                                          KSloss=round(ks_loss.mean().item(), 3))

                if args.save_steps > 0 and global_step % args.save_steps == 1:
                    if args.local_rank == 0:
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        # model_to_save.save_pretrained(output_dir)
                        launchTimestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                        torch.save(
                            {'epoch': i_epoch, 'state_dict': model_to_save.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'args': args},
                            args.output_dir + '/m-' + launchTimestamp + '-' + str("%.4f" % loss_scalar) + '.pth.tar')
                        if args.fp16:
                            output_amp_file = os.path.join(
                                args.output_dir, "amp.{0}.bin".format(i_epoch))
                            torch.save(amp.state_dict(), output_amp_file)

                        logger.logger.info("Saving model checkpoint to %s", args.output_dir)
                        logger.logger.info("***** CUDA.empty_cache() *****")
                        torch.cuda.empty_cache()
                    if n_gpu > 1:
                        dist.barrier()


if __name__ == '__main__':
    main()
