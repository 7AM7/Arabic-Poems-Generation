
import os
import random
import numpy as np
from tqdm import tqdm
import argparse
from configparser import ConfigParser
import bert_utils

import torch
from torch.utils.data import (DataLoader,Dataset, random_split, RandomSampler, SequentialSampler,
                              TensorDataset)

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForMaskedLM,BertTokenizer


def main():
  parser = argparse.ArgumentParser(description="Finetuning Bert Model")
  parser.add_argument(
    "-i",
    "--train_dataset",
    help="Path your train dataset",
    required=True,
    type=str,
  )
  parser.add_argument(
    "-t",
    "--test_dataset",
    help="Path yout test dataset",
    required=True,
    type=str,
  )
  parser.add_argument(
    "-s",
    "--output_dir",
    help="Path to save the model files",
    required=True,
    type=str,
  )
  args = parser.parse_args()



  #read model name and hyperparamter from config file
  config_object = ConfigParser()
  config_object.read("config.ini")
  model1 = config_object["arabert"]
  model_name = model1["model_name"]
  epochs = model1["epochs"]
  lr = model1["lr"]
  adam_epsilon = model1["adam_epsilon"]

  #GPU Connection
  if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
  else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

  #load tokenizer and model
  tokenizer = bert_utils.Load_tokenizer(model_name)
  model = bert_utils.Load_model(model_name)
  model.cuda()

  #read train and test dataset
  dataset_train = bert_utils.create_dataset(tokenizer,args.train_dataset,block_size=24)
  dataset_test = bert_utils.create_dataset(tokenizer,args.test_dataset,block_size=24)

  #train_evaluate
  bert_utils.set_seed(seed=42)
  model , tokenizer , loss = train_evaluate(model,tokenizer,dataset_train,dataset_test,batch_size,lr,adam_epsilon,epochs)

  #save model
  bert_utils.Save_model_after_training(args.output_dir)
  

if __name__ == "__main__":
    main()