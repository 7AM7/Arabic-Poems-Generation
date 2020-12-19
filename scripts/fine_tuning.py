import os
import subprocess
import logging
import random
import csv
import numpy as np
import pandas as pd
import ast
import re
from glob import glob
import sys
import gc
import time
import argparse

import torch
from tqdm import notebook
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import trange, tqdm, notebook
from sklearn.utils import shuffle
from transformers import BertForMaskedLM, BertTokenizer, XLNetTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import sentencepiece as spm

# xla imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp # To read more, http://pytorch.org/xla/index.html#running-on-multiple-xla-devices-with-multithreading
import torch_xla.distributed.xla_multiprocessing as xmp # To read more, http://pytorch.org/xla/index.html#running-on-multiple-xla-devices-with-multiprocessing
import torch_xla.distributed.parallel_loader as pl


class TPUFineTuning:
    def __init__(self,
                output_dir,
                checkpoint_path,
                bert_config
    ):
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.bert_config = bert_config

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def checkpoint(self):
        TF_OUTPUT_PATH = os.path.join(self.output_dir, "tf_checkpoints")
        TORCH_OUTPUT_PATH = os.path.join(self.output_dir, "torch_checkpoints")
        if not os.path.exists(TF_OUTPUT_PATH):
            os.makedirs(TF_OUTPUT_PATH)

        if not os.path.exists(TORCH_OUTPUT_PATH):
            os.makedirs(TORCH_OUTPUT_PATH)

        checkpoint_name = self.checkpoint_path.split('/')[-1]
        logging.info("Downloading Tensorflow checkpoints ...")
        subprocess.call(['gsutil', 'cp', self.checkpoint_path + '.*', TF_OUTPUT_PATH])

        logging.info("Converting Tensorflow checkpoints to Pytorch...")
        tf_path = os.path.join(TF_OUTPUT_PATH, checkpoint_name)
        pt_path = os.path.join(TORCH_OUTPUT_PATH, 'torch_' + checkpoint_name + '.bin')
        subprocess.call(['python3', '-m', 'pytorch_pretrained_bert', 'convert_tf_checkpoint_to_pytorch',
                        tf_path, self.bert_config, pt_path])

        subprocess.call(['rm', '-rf', TF_OUTPUT_PATH])
        logging.info("Converted Successfully {}".format(pt_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--output_dir',
      type=str,
      required=True,
      default=None,
      help='fine-tuning output folder path',
    )
    parser.add_argument(
      '--checkpoint_path',
      type=str,
      required=True,
      default=None,
      help='Google cloud storage checkpoint path ',
    )
    parser.add_argument(
      '--bert_config',
      type=str,
      required=True,
      default=None,
      help='The config json file corresponding to the pre-trained BERT model.',
    )
    args = parser.parse_args()

    tpu_fineTune = TPUFineTuning(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        bert_config=args.bert_config
    )
    tpu_fineTune.checkpoint()