import os
from pathlib import Path
import pdb
import sys
sys.path.append("../")
import json
import torch
import random
import logging

import numpy as np
from tqdm import tqdm
from collections import defaultdict

# from transformers import set_seed
from transformers.integrations import TensorBoardCallback
from transformers import EarlyStoppingCallback

from arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from backbone import get_backbone
from data_processor import (
    SCProcessor
)
from model import get_model
from metric import (
    compute_F1
)
from input_utils import get_bio_labels
from trainer import Trainer
from dump_result import dump_result

# from torch.utils.tensorboard import SummaryWriter

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

# logging config 
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# labels
model_args.num_labels = 2

# markers 
markers = defaultdict(list)
markers["slot"] = ["<slot>", "</slot>"]
markers["user"] = ["<user>", "</user>"]
markers["entity"] = ["<entity>", "</entity>"]
data_args.markers = markers
insert_markers = [m for ms in data_args.markers.values() for m in ms]

print(data_args, model_args, training_args)

# writter 
earlystoppingCallBack = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience, \
                                              early_stopping_threshold=training_args.early_stopping_threshold)

# model 
backbone, tokenizer, config = get_backbone(model_args.model_type, model_args.model_name_or_path, \
                                           model_args.model_name_or_path, insert_markers, new_tokens=insert_markers)
model = get_model(model_args, backbone)
model.cuda()
data_class = SCProcessor
metric_fn = compute_F1

# dataset 
train_dataset = data_class(data_args, tokenizer, data_args.train_file, False)
eval_dataset = data_class(data_args, tokenizer, data_args.validation_file, False)

# set event types
# training_args.data_for_evaluation = eval_dataset.get_data_for_evaluation()

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
    # training_args.data_for_evaluation = test_dataset.get_data_for_evaluation()
    logits, labels, metrics = trainer.predict(
        test_dataset=test_dataset,
        ignore_keys=["loss"]
    )
    if data_args.test_exists_labels:
        print(metrics)
    else:
        preds = logits[:, 1]
        dump_result(data_args, preds, data_args.test_file)
