# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
import os 
import pdb 
import json
import math 
import copy 
from re import L
from string import whitespace
from scipy.fftpack import shift
import torch 
import logging
import collections

from collections import defaultdict

from typing import List
from tqdm import tqdm 
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for event extraction."""

    def __init__(self, example_id, text, entities, triple, labels=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            text: List of str. The untokenized text.
            triggerL: Left position of trigger.
            triggerR: Light position of tigger.
            labels: Event type of the trigger
        """
        self.example_id = example_id
        self.text = text
        self.entities = entities
        self.triple = triple 
        self.labels = labels


class InputFeatures(object):
    """Input features of an instance."""
    
    def __init__(self,
                example_id,
                input_ids,
                attention_mask,
                token_type_ids=None,
                labels=None,
                start_positions=None,
                end_positions=None
        ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.start_positions = start_positions
        self.end_positions = end_positions


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

    def get_data_for_evaluation(self):
        self.data_for_evaluation["pred_type"] = self.get_pred_types()
        self.data_for_evaluation["true_type"] = self.get_true_types()
        return self.data_for_evaluation

    def get_pred_types(self):
        pred_types = []
        for example in self.examples:
            pred_types.append(example.pred_type)
        return pred_types 

    def get_true_types(self):
        true_types = []
        for example in self.examples:
            true_types.append(example.true_type)
        return true_types
    
    def _truncate(self, outputs, max_seq_length):
        is_truncation = False 
        if len(outputs["input_ids"]) > max_seq_length:
            logger.warning("An instance exceeds the maximum length.")
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
            input_ids = torch.tensor(features.input_ids, dtype=torch.long),
            attention_mask = torch.tensor(features.attention_mask, dtype=torch.float32)
        )
        if features.token_type_ids is not None and self.config.return_token_type_ids:
            data_dict["token_type_ids"] = torch.tensor(features.token_type_ids, dtype=torch.long)
        if features.labels is not None:
            data_dict["labels"] = torch.tensor(features.labels, dtype=torch.long)
        if features.start_positions is not None: 
            data_dict["start_positions"] = torch.tensor(features.start_positions, dtype=torch.long)
        if features.end_positions is not None:
            data_dict["end_positions"] = torch.tensor(features.end_positions, dtype=torch.long)
        return data_dict
        
    def collate_fn(self, batch):
        output_batch = dict()
        for key in batch[0].keys():
            output_batch[key] = torch.stack([x[key] for x in batch], dim=0)
        if self.config.truncate_in_batch:
            input_length = int(output_batch["attention_mask"].sum(-1).max())
            for key in ["input_ids", "attention_mask", "token_type_ids", "trigger_left_mask", "trigger_right_mask"]:
                if key not in output_batch:
                    continue
                output_batch[key] = output_batch[key][:, :input_length]
            if "labels" in output_batch and len(output_batch["labels"].shape) == 2:
                if self.config.truncate_seq2seq_output:
                    output_length = int((output_batch["labels"]!=-100).sum(-1).max())
                    output_batch["labels"] = output_batch["labels"][:, :output_length]
                else:
                    output_batch["labels"] = output_batch["labels"][:, :input_length] 
        return output_batch


class SCProcessor(DataProcessor):
    """Data processor for sequence classification."""

    def __init__(self, config, tokenizer, input_file, is_testing):
        super().__init__(config, tokenizer, is_testing)
        self.is_overflow = []
        self.read_examples(input_file)
        self.convert_examples_to_features()
    
    @classmethod
    def get_text_and_entities(self, item):
        utterances = []
        entities = []
        triples = []
        ent_ids = set()
        for turn in item["content"]:
            for key in list(turn.keys())[:2]:
                utterances.append(turn[key])
            for ent in turn["info"]["ents"]:
                for position in ent["pos"]:
                    utterance_id = len(utterances) - (3-position[0])
                    offset = position[1:]
                    entities.append({
                        "id": ent["id"],
                        "name": ent["name"],
                        "type": ent["type"],
                        "position": offset,
                        "utterance_id": utterance_id
                    })
                    ent_ids.add(ent["id"])
                    assert utterances[utterance_id][offset[0]:offset[1]] == ent["name"]
            for triple in turn["info"]["triples"]:
                utterance_id = len(utterances) - (3-triple["pos"][0])
                offset = triple["pos"][1:]
                triples.append({
                    "ent-name": triple["ent-name"] if "ent-name" in triple else "None",
                    "ent-id": triple["ent-id"] if "ent-id" in triple else "None",
                    "value": triple["value"],
                    "prop": triple["prop"],
                    "position": offset,
                    "utterance_id": utterance_id
                })
                assert utterances[utterance_id][offset[0]:offset[1]] == triple["value"]
        ent_ids = sorted(list(ent_ids)+["NA"])
        return utterances, entities, triples, ent_ids  # NA for user entity
    
    @classmethod
    def get_instance(self, utterances, entities, ent_id, triple):
        if ent_id != "NA":
            ent_utt_id = -1 
            min_dis = 1000
            for ent in entities:
                if ent["id"] == ent_id:
                    if ent["utterance_id"] <= triple["utterance_id"]:
                        if triple["utterance_id"]-ent["utterance_id"] < min_dis:
                            ent_utt_id = ent["utterance_id"]
                            min_dis = triple["utterance_id"]-ent["utterance_id"]
                    else:
                        continue
            if ent_utt_id == -1: # no entities before triple 
                return None, None, None 
            # entities 
            entities_in_range = []
            for ent in entities:
                if ent["utterance_id"] == ent_utt_id:
                    entities_in_range.append(ent)
            # text range
            all_range = range(ent_utt_id, triple["utterance_id"]+1)
            instance_utts = []
            for idx in all_range:
                instance_utts.append(utterances[idx])
            text = "".join(instance_utts)
            shift = all_range[0]
            for ent in entities_in_range:
                ent["position"][0] += len("".join(instance_utts[:ent["utterance_id"]-shift]))
                ent["position"][1] += len("".join(instance_utts[:ent["utterance_id"]-shift]))
                ent["type"] = "entity"
                assert text[ent["position"][0]:ent["position"][1]] == ent["name"]
            triple["position"][0] += len("".join(instance_utts[:triple["utterance_id"]-shift]))
            triple["position"][1] += len("".join(instance_utts[:triple["utterance_id"]-shift]))
            triple["type"] = "triple"
            assert text[triple["position"][0]:triple["position"][1]] == triple["value"]
            return text, entities_in_range, triple 
        else:
            _text = utterances[triple["utterance_id"]]
            text = _text + "用户"
            entity = {
                "position": [len(_text), len(text)],
                "type": "user"
            }
            triple["type"] = "triple"
            return text, [entity], triple


    def read_examples(self, input_file):
        self.examples = []
        data = json.load(open(input_file))
        for item in tqdm(data, desc="Reading from %s" % input_file):
            utterances, entities, triples, ent_ids = self.get_text_and_entities(item)
            for triple in triples:
                for ent_id in ent_ids:
                    text, entities_in_text, triple_in_text = self.get_instance(utterances, copy.deepcopy(entities), ent_id, copy.deepcopy(triple))
                    if text is None:
                        continue
                    if len(text) > self.config.max_seq_length:
                        continue
                    if self.is_testing and not self.config.test_exists_labels:
                        labels = None 
                    else:
                        labels = 1 if ent_id == triple["ent-id"] else 0
                    example = InputExample(
                        example_id=None,
                        text=text,
                        entities=entities_in_text,
                        triple=triple_in_text,
                        labels=labels
                    )
                    self.examples.append(example)

    @staticmethod
    def insert_marker(text, entities):
        sorted_entities = sorted(entities, key=lambda e: e["position"][0])
        marked_text = []
        curr_pos = 0
        for ent in sorted_entities:
            if ent["type"] == "triple": # triple
                markers = ["<slot>", "</slot>"]
            elif ent["type"] == "entity":
                markers = ["<entity>", "</entity>"]
            elif ent["type"] == "user":
                markers = ["<user>", "</user>"]
            else:
                raise ValueError()
            marked_text.extend(text[curr_pos:ent["position"][0]])
            marked_text.append(markers[0])
            marked_text.extend(text[ent["position"][0]:ent["position"][1]])
            marked_text.append(markers[1])
            curr_pos = ent["position"][1]
        if text[curr_pos:]:
            marked_text.extend(text[curr_pos:])
        return "".join(marked_text)


    def convert_examples_to_features(self):
        self.input_features = []
        self.is_overflow = []

        for example in tqdm(self.examples, desc="Processing features for SL"):
            text = self.insert_marker(example.text, example.entities+[example.triple])
            outputs = self.tokenizer(text,
                                     padding="max_length",
                                     truncation=False,
                                     max_length=self.config.max_seq_length,
                                     is_split_into_words=False)
            # Roberta tokenizer doesn't return token_type_ids
            if "token_type_ids" not in outputs:
                outputs["token_type_ids"] = [0] * len(outputs["input_ids"])
            outputs, is_overflow = self._truncate(outputs, self.config.max_seq_length)
            self.is_overflow.append(is_overflow)

            features = InputFeatures(
                example_id=example.example_id,
                input_ids=outputs["input_ids"],
                attention_mask=outputs["attention_mask"],
                token_type_ids=outputs["token_type_ids"],
                labels=example.labels
            )
            self.input_features.append(features)


