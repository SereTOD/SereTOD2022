# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
import os
import re 
import pdb
import json
import torch
import logging

from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for event extraction."""

    def __init__(self, example_id, text, labels=None):
        self.example_id = example_id
        self.text = text
        self.labels = labels


class InputFeatures(object):
    """Input features of an instance."""
    
    def __init__(self, example_id, input_ids, attention_mask, token_type_ids=None, labels=None):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels


class DataProcessor(Dataset):
    """Base class of data processor."""

    def __init__(self, config, tokenizer, is_testing):
        self.config = config
        self.tokenizer = tokenizer
        self.is_testing = is_testing
        self.examples = []
        self.input_features = []

    def read_examples(self, input_file):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def _truncate(self, outputs, max_seq_length):
        is_truncation = False
        if len(outputs["input_ids"]) > max_seq_length:
            print("An instance exceeds the maximum length.")
            is_truncation = True
            for key in ["input_ids", "attention_mask", "token_type_ids", "offset_mapping"]:
                if key not in outputs:
                    continue
                outputs[key] = outputs[key][:max_seq_length]
        return outputs, is_truncation

    def get_ids(self):
        ids = []
        for example in self.examples:
            ids.append(example.example_id)
        return ids

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, index):
        features = self.input_features[index]
        data_dict = dict(
            input_ids=torch.tensor(features.input_ids, dtype=torch.long),
            attention_mask=torch.tensor(features.attention_mask, dtype=torch.float32)
        )
        if features.token_type_ids is not None and self.config.return_token_type_ids:
            data_dict["token_type_ids"] = torch.tensor(features.token_type_ids, dtype=torch.long)
        if features.labels is not None:
            data_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
        return data_dict

    def collate_fn(self, batch):
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        if self.config.truncate_in_batch:
            input_length = int(output_batch["attention_mask"].sum(-1).max())
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key not in output_batch:
                    continue
                output_batch[key] = output_batch[key][:, :input_length]
            if "labels" in output_batch and len(output_batch["labels"].shape) == 2:
                output_batch["labels"] = output_batch["labels"][:, :input_length]
        return output_batch


class SLProcessor(DataProcessor):
    """Data processor for sequence labeling."""

    def __init__(self, config, tokenizer, input_file, is_testing):
        super().__init__(config, tokenizer, is_testing)
        self.is_overflow = []
        self.config.type2id["X"] = -100
        self.read_examples_turn(input_file)
        self.convert_examples_to_features()

    def compute_offset_in_turn(self, turn, original_pos):
        speaker1 = list(turn.keys())[0]
        speaker2 = list(turn.keys())[1]
        start, end = None, None 
        if original_pos[0] == 1:
            start, end = original_pos[1:]
        else:
            start = original_pos[1] + len(turn[speaker1])
            end = original_pos[2] + len(turn[speaker1])
        return [start, end]
    
    def read_examples_turn(self, input_file):
        self.examples = []
        data = json.load(open(input_file))
        for item in tqdm(data, desc="Reading from %s" % input_file):
            for turn in item["content"]:
                # text in turn
                text_in_turn_list = []
                for key in list(turn.keys())[:2]:
                    text_in_turn_list.append(turn[key])
                text_in_turn = list("".join(text_in_turn_list))
                labels = ["O"] * len(text_in_turn)
                if self.is_testing and not self.config.test_exists_labels:
                    example = InputExample(
                        example_id=None,
                        text=text_in_turn,
                        labels=labels 
                    )
                else:
                    # triples per entity
                    if len(turn["info"]["ents"]) == 0 and not self.is_testing:
                        continue
                    for ent in turn["info"]["ents"]:     
                        for position in ent["pos"]:
                            offset = self.compute_offset_in_turn(turn, position)
                            assert "".join(text_in_turn[offset[0]:offset[1]]) == ent["name"]
                            labels[offset[0]] = f"B-{ent['type']}"
                            for p in range(offset[0]+1, offset[1]):
                                labels[p] = f"I-{ent['type']}"
                    example = InputExample(
                        example_id=None,
                        text=text_in_turn,
                        labels=labels
                    )
                self.examples.append(example)


    def get_final_labels(self, labels, word_ids_of_each_token, label_all_tokens=False):
        final_labels = []
        pre_word_id = None
        for word_id in word_ids_of_each_token:
            if word_id is None:
                final_labels.append(-100)
            elif word_id != pre_word_id:  # first split token of a word
                final_labels.append(self.config.type2id[labels[word_id]])
            else:
                final_labels.append(self.config.type2id[labels[word_id]] if label_all_tokens else -100)
            pre_word_id = word_id

        return final_labels

    @staticmethod
    def insert_marker(text, entities, labels, markers):
        sorted_entities = sorted(entities, key=lambda e: e["position"][0])
        marked_text = []
        marked_labels = []
        curr_pos = 0
        for ent in sorted_entities:
            marked_text.extend(text[curr_pos:ent["position"][0]])
            marked_text.append(markers[ent["type"]][0])
            marked_text.extend(text[ent["position"][0]:ent["position"][1]])
            marked_text.append(markers[ent["type"]][1])
            if labels is not None:
                marked_labels.extend(labels[curr_pos:ent["position"][0]])
                marked_labels.append("X")
                marked_labels.extend(labels[ent["position"][0]:ent["position"][1]])
                marked_labels.append("X")
            curr_pos = ent["position"][1]
        
        if text[curr_pos:]:
            marked_text.extend(text[curr_pos:])
            if labels is not None:
                marked_labels.extend(labels[curr_pos:])
        return "".join(marked_text), marked_labels


    def convert_examples_to_features(self):
        self.input_features = []
        self.is_overflow = []

        for example in tqdm(self.examples, desc="Processing features for SL"):
            text, labels = example.text, example.labels
            outputs = self.tokenizer(text,
                                     padding="max_length",
                                     truncation=False,
                                     max_length=self.config.max_seq_length,
                                     is_split_into_words=True)
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
            outputs, is_overflow = self._truncate(outputs, self.config.max_seq_length)
            self.is_overflow.append(is_overflow)

            if labels is not None:
                word_ids_of_each_token = outputs.word_ids()[: self.config.max_seq_length]
                final_labels = self.get_final_labels(labels, word_ids_of_each_token, label_all_tokens=False)
            else:
                final_labels = None 

            features = InputFeatures(
                example_id=example.example_id,
                input_ids=outputs["input_ids"],
                attention_mask=outputs["attention_mask"],
                token_type_ids=outputs["token_type_ids"],
                labels=final_labels
            )
            self.input_features.append(features)