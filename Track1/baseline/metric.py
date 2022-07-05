# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
import numpy as np 
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as span_f1_score
from seqeval.scheme import IOB2


def compute_F1(logits, labels, **kwargs):
    predictions = np.argmax(logits, axis=-1)
    pos_labels = list(set(labels.tolist()))
    pos_labels.remove(0)
    micro_f1 = f1_score(labels, predictions, labels=pos_labels, average="micro") * 100.0
    return {"micro_f1": micro_f1}


def convert_to_names(instances, id2label):
    name_instances = []
    for instance in instances:
        name_instances.append([id2label[item] for item in instance])
    return name_instances


def select_start_position(preds, labels, merge=True):
    final_preds = []
    final_labels = []

    if merge:
        final_preds = preds[labels != -100].tolist()
        final_labels = labels[labels != -100].tolist()
    else:
        for i in range(labels.shape[0]):
            final_preds.append(preds[i][labels[i] != -100].tolist())
            final_labels.append(labels[i][labels[i] != -100].tolist())

    return final_preds, final_labels


def compute_span_F1(logits, labels, **kwargs):
    if len(logits.shape) == 3:
        preds = np.argmax(logits, axis=-1)
    else:
        preds = logits
    # convert id to name
    training_args = kwargs["training_args"]
    id2label = training_args.id2role if training_args.task_name == "SF" else training_args.id2type
    final_preds, final_labels = select_start_position(preds, labels, False)
    final_preds = convert_to_names(final_preds, id2label)
    final_labels = convert_to_names(final_labels, id2label)
    micro_f1 = span_f1_score(final_labels, final_preds, mode='strict', scheme=IOB2) * 100.0
    return {"micro_f1": micro_f1}