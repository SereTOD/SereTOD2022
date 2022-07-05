# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
import os
from pathlib import Path
import pdb
import sys
sys.path.append("../")
import json
import torch
import random

import numpy as np

# from transformers import set_seed
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback

from arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from backbone import get_backbone
from data_processor import (
    SLProcessor
)
from model import get_model
from metric import (
    compute_span_F1
)
from trainer import Trainer
from dump_result import dump_result_sl
from input_utils import get_bio_labels


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# argument parser
parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# output dir
model_name_or_path = model_args.model_name_or_path.split("/")[-1]
output_dir = Path(
    os.path.join(os.path.join(os.path.join(training_args.output_dir, training_args.task_name), model_args.paradigm),
                 model_name_or_path))
output_dir.mkdir(exist_ok=True, parents=True)
training_args.output_dir = output_dir

# set seed
set_seed(training_args)

# local rank
# training_args.local_rank = int(os.environ["LOCAL_RANK"])

# prepare labels
type2id_path = data_args.type2id_path
data_args.type2id = json.load(open(type2id_path))
model_args.num_labels = len(data_args.type2id)
training_args.label_name = ["labels"]

data_args.type2id = get_bio_labels(data_args.type2id)
data_args.id2type = {id: label for label, id in data_args.type2id.items()}
training_args.id2type = data_args.id2type
model_args.num_labels = len(data_args.type2id)

# markers 
data_args.markers = ["<entity>", "</entity>"]

print(data_args, model_args, training_args)


# writter
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                              early_stopping_threshold=training_args.early_stopping_threshold)

# model 
backbone, tokenizer, config = get_backbone(model_args.model_type, model_args.model_name_or_path,
                                           model_args.model_name_or_path, data_args.markers,
                                           new_tokens=data_args.markers)
model = get_model(model_args, backbone)
model.cuda()
data_class = None
metric_fn = None

data_class = SLProcessor
metric_fn = compute_span_F1
# dataset 
train_dataset = data_class(data_args, tokenizer, data_args.train_file, False)
eval_dataset = data_class(data_args, tokenizer, data_args.validation_file, False)

# Trainer 
trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=metric_fn,
    data_collator=train_dataset.collate_fn,
    tokenizer=tokenizer,
    callbacks=[earlystoppingCallBack]
)
trainer.train()

if training_args.do_predict:
    test_dataset = data_class(data_args, tokenizer, data_args.test_file, True)
    logits, labels, metrics = trainer.predict(
        test_dataset=test_dataset,
        ignore_keys=["loss"]
    )   
    if data_args.test_exists_labels:
        print(metrics)
    else:
        preds = np.argmax(logits, axis=-1)
        dump_result_sl(preds, labels, test_dataset.is_overflow, data_args)


