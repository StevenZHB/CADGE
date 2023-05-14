# CADGE: Context-Aware Dialogue Generation Enhanced with Graph-Structured Knowledge Aggregation
This repository is the code and resources for the paper [CADGE: Context-Aware Dialogue Generation Enhanced with Graph-Structured Knowledge Aggregation](https://arxiv.org/abs/2305.06294) 

## Instructions

This project is mainly implemented with following tools:
- **Pytorch** 
- **DGL** 
- The initial checkpoints of pretrained models come from [Hugginface](https://huggingface.co).

So if you want to run this code, you must have following preliminaries:
- Python 3 or Anaconda (mine is 3.8)
- [Pytorch](https://pytorch.org/) (mine is 1.11.0)
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base)) (mine is 4.21.3)
- **DGL** (mine is 0.9) 

**It is worth mentioning that the installation of previous DGL is not so easy. I found out an interesting fact that if DGL is not compatible to Pytorch, when running this pytorch code with cuda it may give some interesting errors. I have no solution to this issue, as it depends on the individual environment.**

## Datasets and Resources

### Directly Download Dataset and Resources
To reproduce our work you need to download following files:

- The raw data come from the paper [CCM](https://github.com/thu-coai/ccm)

You need to download the raw data only if you want to reproduce the dataset by yourself.

### Preprocess Dataset From Scratch

Make sure you have `resources/commonsense_conversation_dataset` ready.

Download [rel2words.txt](https://www.dropbox.com/s/0wetcr2o1wa7z5f/rel2words.txt?dl=0) from Dropbox, and put it to `resources/rel2words.txt`.

Run `python preprocess.py --model_name_or_path=microsoft/unilm-base-cased` to get the dataset at `datasets/cadge`.

### The introduction of the dataset
The structure of `datasets`should be like this:
```markdown
├── datasets/cadge
      └── `id2triple.txt`    
      └── `id2word.txt`     
      └── `rel2word.txt` 
      └── `testset.txt` 
      └── `trainset.txt` 
      └── `triple2id.txt`
      └── `valset.txt`
      └── `word2id.txt`
```

## Quick Start

### 1. Install packages
```shell
pip install -r requirements.txt
```
And you have to install **Pytorch** from their [homepage](https://pytorch.org/get-started/locally/).

### 2. Collect Datasets and Resources

As mentioned above.

### 3. Run the code for training or testing

Please refer to the command examples listed in `python_commands.sh`:

For example, for our model:
```shell
bash run_train.sh
```

```shell
bash run_generate.sh
```

Revise the parameters according to your demand.

## Notation
Nothing is difficult. For technical details please refer to our paper.

## Citation
If you found this repository or paper is helpful to you, please cite our paper. 
Currently we only have arxiv citation listed as follows:

This is the arxiv citation:
```angular2
@misc{zhanga2023cadge,
      title={CADGE: Context-Aware Dialogue Generation Enhanced with Graph-Structured Knowledge Aggregation}, 
      author={Hongbo Zhanga and Chen Tang and Tyler Loakmana and Chenghua Lina and Stefan Goetze},
      year={2023},
      eprint={2305.06294},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



