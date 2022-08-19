# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
import os 
import json 
from tqdm import tqdm 


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


def get_pred_per_mention(preds, id2label, text, turn):
    ents = []
    start_pos = None 
    type = None 
    for i, pred in enumerate(preds):
        if id2label[pred][0] == "B": # 
            if start_pos is not None:
                ents.append({
                    "pos": convert_pos(turn, [start_pos, i]),
                    "type": type,
                    "name": "".join(text[start_pos:i])
                })
            start_pos = i 
            type = id2label[pred].split("-")[1]
        elif id2label[pred][0] == "I":
            if start_pos is None: # loose evaluation
                start_pos = i 
                type = id2label[pred].split("-")[1]
        else:
            if start_pos is not None:
                ents.append({
                    "pos": convert_pos(turn, [start_pos, i]),
                    "type": type,
                    "name": "".join(text[start_pos:i])
                })
                start_pos = None 
                type = None 
    if start_pos is not None:
        ents.append({
            "pos": convert_pos(turn, [start_pos, len(preds)]),
            "type": type,
            "name": "".join(text[start_pos:len(preds)])
        })
    speaker1 = list(turn.keys())[0]
    final_ents = []
    for ent in ents:
        if ent["pos"][0][1] < len(turn[speaker1]) and ent["pos"][0][2] > len(turn[speaker1]):
            continue
        final_ents.append(ent)
        assert ent["name"] == turn[list(turn.keys())[ent["pos"][0][0]-1]][ent["pos"][0][1]:ent["pos"][0][2]]
    return final_ents


def convert_pos(turn, position):
    speaker1 = list(turn.keys())[0]
    offset = None 
    if position[0] >= len(turn[speaker1]):
        offset = [
            2, position[0]-len(turn[speaker1]), position[1]-len(turn[speaker1])
        ]
    else:
        offset = [
            1, *position
        ]
    return [offset]


def dump_result_sl(preds, labels, is_overflow, config):
    # get per-word predictions
    preds, _ = select_start_position(preds, labels, False)
    data = json.load(open(config.test_file))
    idx = 0
    for item in tqdm(data, desc="Parsing result for %s" % config.test_file):
        for turn in item["content"]:
            text_in_turn_list = []
            for key in list(turn.keys())[:2]:
                text_in_turn_list.append(turn[key])
            text_in_turn = list("".join(text_in_turn_list))
            # check for alignment 
            if not is_overflow[idx]:
                if len(preds[idx]) != len(text_in_turn):  # remove space/special token
                    print("Warning! An unexpected mis-alignment.", text_in_turn)
            # get predictions
            ents = get_pred_per_mention(preds[idx], config.id2type, text_in_turn, turn)
            # record results
            turn["info"]["ents"] = ents 
            idx += 1
    assert idx == len(preds)
    json.dump(data, open("../data/test_with_entity_mentions.json", "w"), indent=4, ensure_ascii=False)