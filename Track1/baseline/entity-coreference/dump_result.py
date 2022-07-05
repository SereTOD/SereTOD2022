# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
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
    def __init__(self, item):
        self.id = item["id"]
        self.text, self.entities = self.get_text_and_entities(item)
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
        # merged_entities = defaultdict(list)
        # for entity in self.entities:
        #     merged_entities[entity["id"]].append(entity)
        # merged_entities = list(merged_entities.values())
        entities = sorted(self.entities, key=lambda x: (x["utterance_id"], x["position"][0]))
        entity2id = {self.entity_id(e):idx for idx, e in enumerate(entities)} 

        # self.label_groups = [[entity2id[self.entity_id(e)] for e in entities] for entities in merged_entities] # List[List[int]] each sublist is a group of entity index that co-references each other
        self.sorted_entity_spans = [(entity["utterance_id"], entity["position"], entity["type"]) for entity in entities]


def dump_result(input_path, preds):
    # load examples 
    examples = []
    with open(os.path.join(input_path))as f:
        data = json.load(f)
    for item in data:
        doc = Document(item)
        if doc.entities:
            examples.append(doc)
    # each item is the clusters of a document
    final_result = []
    for example, pred_per_doc in zip(examples, preds):
        assert example.id == pred_per_doc["doc_id"]
        entities = sorted(example.entities, key=lambda x: (x["utterance_id"], x["position"][0]))
        utterances = example.text
        clusters = pred_per_doc["clusters"]
        for ent_id, cluster in enumerate(clusters):
            for mid in cluster:
                entities[mid]["id"] = f"ent-{ent_id}"
        final_result.append({
            "id": example.id,
            "utterances": utterances,
            "entities": entities
        })
    final_data = convert_format(data, final_result)
    json.dump(final_data, open(os.path.join("../data", "test_with_entity_coref.json"), "w"), indent=4, ensure_ascii=False)


def convert_format(original_data, result):
    id2result = {}
    for item in result:
        id2result[item["id"]] = item 
    for item in original_data:
        if item["id"] in id2result:
            for entity in id2result[item["id"]]["entities"]:
                turn_id = entity["utterance_id"] // 2
                u_id = entity["utterance_id"] % 2 + 1 
                ent_marker = (u_id, entity["position"][0], entity["position"][1])
                in_original = False 
                for original_ent in item["content"][turn_id]["info"]["ents"]:
                    for position in original_ent["pos"]:
                        position = (position[0], position[1], position[2])
                        if position == ent_marker:
                            original_ent["id"] = entity["id"]
                            in_original = True 
                            break
                    if in_original:
                        break
                assert in_original
    return original_data



# if __name__ == "__main__":
#     preds = json.load(open("output/pred_clusters.json"))
#     dump_result("../data/test.json", preds)
