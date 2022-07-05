import os
import pdb 
import json
import torch

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset


def valid_split(point, spans):
    # retain context of at least 3 tokens
    for sp in spans:
        if point > sp[0] - 3 and point <= sp[1] + 3:
            return False
    return True


def split_spans(point, spans):
    part1 = []
    part2 = []
    i = 0
    for sp in spans:
        if sp[1] < point:
            part1.append(sp)
            i += 1
        else:
            break
    part2 = spans[i:]
    return part1, part2


class Document:
    def __init__(self, item, data_args, is_testing):
        self.id = item["id"]
        self.text, self.entities = self.get_text_and_entities(item)
        self.data_args = data_args 
        self.is_testing = is_testing
        self.populate_entity_spans()
    
    def assert_offset(self, text, entities):
        for ent in entities:
            assert text[ent["position"][0]:ent["position"][1]] == ent["name"]
    
    def get_text_and_entities(self, item):
        utterances = []
        entities = []
        for turn in item["content"]:
            for key in list(turn.keys())[:2]:
                utterances.append(turn[key])
            for ent in turn["info"]["ents"]:
                for position in ent["pos"]:
                    utterance_id = len(utterances) - (3-position[0])
                    offset = position[1:]
                    entities.append({
                        "id": ent["id"] if "id" in ent else "None",
                        "name": ent["name"],
                        "type": ent["type"],
                        "position": offset,
                        "utterance_id": utterance_id
                    })
                    assert utterances[utterance_id][offset[0]:offset[1]] == ent["name"]
        return utterances, entities
    

    def entity_id(self, entity):
        return entity["name"] + "_" + str(entity["position"][0])


    def populate_entity_spans(self):
        entities = sorted(self.entities, key=lambda x: (x["utterance_id"], x["position"][0]))
        entity2id = {self.entity_id(e):idx for idx, e in enumerate(entities)} 
        self.sorted_entity_spans = [(entity["utterance_id"], entity["position"], entity["type"]) for entity in entities]
        if not (self.is_testing and self.data_args.test_exists_labels):
            merged_entities = defaultdict(list)
            for entity in self.entities:
                merged_entities[entity["id"]].append(entity)
            merged_entities = list(merged_entities.values())
            self.label_groups = [[entity2id[self.entity_id(e)] for e in entities] for entities in merged_entities] # List[List[int]] each sublist is a group of entity index that co-references each other
        else:
            self.label_groups = None 


class ECProcessor(Dataset):
    def __init__(self, data_args, tokenizer, input_path, is_testing):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.is_testing = is_testing
        self.max_length = data_args.max_seq_length
        self.load_examples(input_path)
        self.examples = self.examples
        self.tokenize()
        self.to_tensor()
    
    def load_examples(self, input_path):
        self.examples = []
        with open(os.path.join(input_path))as f:
            data = json.load(f)
        for item in tqdm(data):
            doc = Document(item, self.data_args, self.is_testing)
            if doc.sorted_entity_spans:
                self.examples.append(doc)
    
    def tokenize(self):
        # TODO: split articless into part of max_length
        self.tokenized_samples = []
        for example in tqdm(self.examples, desc="tokenizing"):
            entity_spans = [] # [[(start, end)], [],...]
            input_ids = [] # [[], [], ...]

            label_groups = example.label_groups
            spans = example.sorted_entity_spans
            text = example.text
            entity_id = 0
            sub_input_ids = [self.tokenizer.cls_token_id]
            sub_entity_spans = []
            for sent_id, word in enumerate(text):
                i = 0
                tmp_entity_spans = []
                tmp_input_ids = []
                # add special tokens for entity
                while entity_id < len(spans) and spans[entity_id][0] == sent_id:
                    sp = spans[entity_id]
                    if i < sp[1][0]:
                        context_ids = self.tokenizer(word[i:sp[1][0]], add_special_tokens=False)["input_ids"]
                        tmp_input_ids += context_ids
                    entity_ids = self.tokenizer(word[sp[1][0]:sp[1][1]], add_special_tokens=False)["input_ids"]
                    start = len(tmp_input_ids)
                    end = len(tmp_input_ids) + len(entity_ids)
                    tmp_entity_spans.append((start, end))
                    assert end != start
                    # special_ids = self.tokenizer(type_tokens(sp[2]), is_split_into_words=True, add_special_tokens=False)["input_ids"]
                    # assert len(special_ids) == 2, print(f"special tokens <{sp[2]}> and <{sp[2]}/> may not be added to tokenizer.")
                    tmp_input_ids += entity_ids

                    i = sp[1][1]
                    entity_id += 1
                if word[i:]:
                    tmp_input_ids += self.tokenizer(word[i:],add_special_tokens=False)["input_ids"]

                # TODO add sep token
                tmp_input_ids += [self.tokenizer.sep_token_id]
                
                if len(sub_input_ids) + len(tmp_input_ids) <= self.max_length:
                    # print(len(sub_input_ids) + len(tmp_input_ids))
                    sub_entity_spans += [(sp[0]+len(sub_input_ids), sp[1]+len(sub_input_ids)) for sp in tmp_entity_spans]
                    sub_input_ids += tmp_input_ids
                else:
                    # print("exceed max length! truncate")
                    assert len(sub_input_ids) <= self.max_length
                    input_ids.append(sub_input_ids)
                    entity_spans.append(sub_entity_spans)

                    # assert len(tmp_input_ids) < self.max_length, print("A sentence too long!\n %s" % " ".join(words[sent_id])) # 3580:
                    while len(tmp_input_ids) >= self.max_length:
                        split_point = self.max_length - 1
                        while not valid_split(split_point, tmp_entity_spans):
                            split_point -= 1
                        tmp_entity_spans_part1, tmp_entity_spans = split_spans(split_point, tmp_entity_spans)
                        tmp_input_ids_part1, tmp_input_ids = tmp_input_ids[:split_point], tmp_input_ids[split_point:]

                        input_ids.append([self.tokenizer.cls_token_id] + tmp_input_ids_part1)
                        entity_spans.append([(sp[0]+1, sp[1]+1) for sp in tmp_entity_spans_part1])

                        tmp_entity_spans = [(sp[0]-len(tmp_input_ids_part1), sp[1]-len(tmp_input_ids_part1)) for sp in tmp_entity_spans]
                        # sub_input_ids = [self.tokenizer.cls_token_id] + tmp_input_ids_part2

                    sub_entity_spans = [(sp[0]+1, sp[1]+1) for sp in tmp_entity_spans]
                    sub_input_ids = [self.tokenizer.cls_token_id] + tmp_input_ids
            if sub_input_ids:
                input_ids.append(sub_input_ids)
                entity_spans.append(sub_entity_spans)
            assert entity_id == len(spans)
                
            tokenized = {"input_ids": input_ids, "attention_mask": None, "entity_spans": entity_spans, "label_groups": label_groups, "doc_id": example.id}
            self.tokenized_samples.append(tokenized)
    
    def to_tensor(self):
        for item in self.tokenized_samples:
            # print(item)
            attention_mask = []
            for ids in item["input_ids"]:
                mask = [1] * len(ids)
                while len(ids) < self.max_length:
                    ids.append(self.tokenizer.pad_token_id)
                    mask.append(0)
                attention_mask.append(mask)
            item["input_ids"] = torch.LongTensor(item["input_ids"])
            item["attention_mask"] = torch.LongTensor(attention_mask)
          
    def __getitem__(self, index):
        return self.tokenized_samples[index]

    def __len__(self):
        return len(self.tokenized_samples)


def collator(data):
    collate_data = {"input_ids": [], "attention_mask": [], "entity_spans": [], "label_groups": [], "splits": [0], "doc_id": []}
    for d in data:
        for k in d:
            collate_data[k].append(d[k])
    lengths = [ids.size(0) for ids in collate_data["input_ids"]]
    for l in lengths:
        collate_data["splits"].append(collate_data["splits"][-1]+l)   
    collate_data["input_ids"] = torch.cat(collate_data["input_ids"])
    collate_data["attention_mask"] = torch.cat(collate_data["attention_mask"])
    return collate_data


def get_dataloader(data_args, training_args, tokenizer, input_path, shuffle=True, is_testing=False):
    dataset = ECProcessor(data_args, tokenizer, input_path, is_testing)
    return DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, 
                        shuffle=shuffle, collate_fn=collator, num_workers=training_args.dataloader_num_workers)


