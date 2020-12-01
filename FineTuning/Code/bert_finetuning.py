
import os
import random
import numpy as np
from tqdm import tqdm
from configparser import ConfigParser
import bert_utils

import torch
from torch.utils.data import (DataLoader,Dataset, random_split, RandomSampler, SequentialSampler,
                              TensorDataset)

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForMaskedLM,BertTokenizer


def main():
  #read model name and hyperparamter from config file
  #NOET : use model 1 and model 2 only because model 3 you can use it only in GT2 finetuning file
  config_object=ConfigParser()
  config_object.read("config.ini")
  model1 = config_object["model1"]
  model_name = model1["model_name"]
  epochs=model1["epochs"]
  lr=model1["lr"]
  adam_epsilon=model1["adam_epsilon"]

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
  tokenizer=bert_utils.Load_tokenizer(model_name)
  model=bert_utils.Load_model(model_name)
  model.cuda()

  #read train and test dataset
  dataset_train = bert_utils.create_dataset(tokenizer,"/content/drive/MyDrive/train_dataset1.txt",block_size=24)
  dataset_test = bert_utils.create_dataset(tokenizer,"/content/drive/MyDrive/test_dataset1.txt",block_size=24)

  #train
  bert_utils.set_seed(seed=42)
  model,tokenizer=train(model,tokenizer,dataset_train,batch_size,lr,adam_epsilon,epochs)

  #save model
  output_dir = '/model_save64k'
  bert_utils.Save_model_after_training(output_dir)

  #evaluate
  model, tokenizer=bert_utils.Load_model_after_training(output_dir)
  model = model.cuda()
  eval_loss = evaluate(model, tokenizer, dataset_test, batch_size)
  print(eval_loss)
  

if __name__ == "__main__":
    main()