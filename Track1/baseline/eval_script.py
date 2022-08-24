# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
import os 
import pdb
import copy  
import json 
import numpy as np 
from tqdm import tqdm 

from scipy.optimize import linear_sum_assignment

from post_process import filter_noisy_token_label, filter_noisy_token_pred

def get_text_and_entities(item):
    utterances = []
    entities = []
    triples = []
    for turn_idx, turn in enumerate(item["content"]):
        for key in list(turn.keys())[:2]:
            utterances.append(turn[key])
        for ent in turn["info"]["ents"]:
            for position in ent["pos"]:
                offset = position[1:]
                entities.append({
                    "id": ent["id"],
                    "name": ent["name"],
                    "type": ent["type"],
                    "position": offset,
                    "turn_id": turn_idx
                })
                # assert utterances[utterance_id][offset[0]:offset[1]] == ent["name"]
        for triple in turn["info"]["triples"]:
            # utterance_id = len(utterances) - (3-triple["pos"][0])
            # offset = triple["pos"][1:]
            triples.append({
                "ent-name": triple["ent-name"],
                "ent-id": triple["ent-id"],
                "value": triple["value"],
                "prop": triple["prop"],
                # "position": offset,
                "turn_id": turn_idx
            })
            # assert utterances[utterance_id][offset[0]:offset[1]] == triple["value"]
    return utterances, entities, triples


def get_golden_labels(label_path):
    data = json.load(open(label_path))
    golden_labels = []
    for item in data:
        utterances, entities, triples = get_text_and_entities(item)
        labels_per_doc = {
            "id": item["id"],
            "utterances": utterances,
            "entities": entities,
            "triples": triples
        }
        golden_labels.append(labels_per_doc)
    return golden_labels




def compute_F1(pred_list, golden_list):
    pred_num = len(pred_list)
    golden_num = len(golden_list)
    # delete dup 
    true_positive = 0
    dup_golden_list = copy.deepcopy(golden_list)
    for pred in pred_list:
        if pred in dup_golden_list:
            true_positive += 1
            dup_golden_list.remove(pred)
    # F1
    precision = true_positive / (pred_num+1e-10)
    recall = true_positive / (golden_num+1e-10)
    f1 = 2*precision*recall / (precision+recall+1e-10)
    return f1 


def get_entids(item):
    entids = []
    for ent in item["entities"]:
        if ent["id"] not in entids:
            entids.append(ent["id"])
    entids = sorted(entids)
    return entids


def get_ents_by_id(item, ent_id):
    ents = []
    for ent in item["entities"]:
        if ent["id"] == ent_id:
            ents.append(ent)
    return ents 


def get_triples_by_entid(item, ent_id):
    triples = []
    for triple in item["triples"]:
        if triple["ent-id"] == ent_id:
            triples.append(triple)
    return triples


def get_ent_id(ent, doc_id, type=False):
    if type:
        return "-".join([doc_id, str(ent["turn_id"]), str(ent["position"][0]), str(ent["position"][1]), ent["type"]])
    else:
        return "-".join([doc_id, str(ent["turn_id"]), str(ent["position"][0]), str(ent["position"][1])])


def get_triple_id(triple, doc_id, prop=False, pred=False):
    if prop:
        if pred:
            return  "-".join([doc_id, str(triple["assign-ent-id"]), str(triple["turn_id"]), 
                        triple["value"], triple["prop"]])
        else:
            return  "-".join([doc_id, str(triple["ent-id"]), str(triple["turn_id"]), 
                        triple["value"], triple["prop"]])
    else:
        return "-".join([doc_id, str(triple["turn_id"]), triple["value"]])


def find_best_entity_assignment_per_doc(pred_result, golden_result):
    entids_in_pred = get_entids(pred_result)
    entids_in_label = get_entids(golden_result)
    m = len(entids_in_pred)
    n = len(entids_in_label)
    cost_matrix = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        triples_pred = get_triples_by_entid(pred_result, entids_in_pred[i])
        for j in range(n):
            triples_label = get_triples_by_entid(golden_result, entids_in_label[j])
            pred_triples = []
            for triple in triples_pred:
                pred_triples.append(get_triple_id(triple, ""))
            golden_triples = []
            for triple in triples_label:
                golden_triples.append(get_triple_id(triple, ""))
            final_f1 = compute_F1(pred_triples, golden_triples)
            cost_matrix[i, j] = final_f1
    row_ids, col_ids = linear_sum_assignment(-cost_matrix)
    pred2label = {}
    for row_id, col_id in zip(row_ids, col_ids):
        pred2label[entids_in_pred[row_id]] = entids_in_label[col_id]
    for entid in entids_in_pred:
        if entid not in pred2label:
            pred2label[entid] = "None"
    for ent in pred_result["entities"]:
        ent["assign-id"] = pred2label[ent["id"]]
    for triple in pred_result["triples"]:
        if triple["ent-id"] == "NA":
            triple["assign-ent-id"] = "NA"
        else:
            triple["assign-ent-id"] = pred2label[triple["ent-id"]]
    return pred_result


def compute_result(all_preds, all_labels):
    docid2labels = {}
    for item in all_labels:
        docid2labels[item["id"]] = item
    # loop for entity assignment
    final_preds = []
    for preds in tqdm(all_preds, desc="Linear assignment"):
        if preds["id"] not in docid2labels:
            print("Warning! The id not in label file.", preds["id"])
        labels = docid2labels[preds["id"]]
        final_preds.append(find_best_entity_assignment_per_doc(preds, labels))
    # evaluation
    all_pred_ents = []
    for item in all_preds:
        for ent in item["entities"]:
            all_pred_ents.append(
                get_ent_id(ent, item["id"], True)
            )
    all_label_ents = []
    for item in all_labels:
        for ent in item["entities"]:
            all_label_ents.append(
                get_ent_id(ent, item["id"], True)
            )
    ent_f1 = compute_F1(all_pred_ents, all_label_ents)
    # Triple F1 
    all_pred_triples = []
    for item in all_preds:
        for triple in item["triples"]:
            if "assign-ent-id" in triple:
                all_pred_triples.append(
                    get_triple_id(triple, item["id"], True, True)
                )
    all_label_triples = []
    for item in all_labels:
        for triple in item["triples"]:
            all_label_triples.append(
                get_triple_id(triple, item["id"], True)
            )
    triple_f1 = compute_F1(all_pred_triples, all_label_triples)
    return ent_f1, triple_f1


if __name__ == "__main__":
    all_preds = filter_noisy_token_pred(json.load(open("submissions.json")))
    all_labels = filter_noisy_token_label(get_golden_labels("data/test_with_labels.json"))
    print(compute_result(all_preds, all_labels))