#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=2 train_chatbotmodel.py --