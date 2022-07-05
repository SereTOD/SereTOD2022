import os 
import json 
import copy 
from tqdm import tqdm 
from data_processor import SCProcessor


def dump_result(data_args, preds, input_file):
    data = json.load(open(input_file))
    idx = 0
    all_data = []
    for item in tqdm(data, desc="Pasring result for %s" % input_file):
        utterances, entities, triples, ent_ids = SCProcessor.get_text_and_entities(item)
        new_item = {}
        new_item["id"] = item["id"]
        new_item["utterances"] = utterances
        new_item["entities"] = entities
        new_item["triples"] = triples
        for triple in triples:
            ent_align_scores = []
            for ent_id in ent_ids:
                text, entities_in_text, triple_in_text = SCProcessor.get_instance(utterances, copy.deepcopy(entities), ent_id, copy.deepcopy(triple))
                if text is None:
                    continue
                if len(text) > data_args.max_seq_length:
                    continue
                ent_align_scores.append({
                    "ent-id": ent_id,
                    "score": preds[idx]
                })
                idx += 1
            ent_align_scores = sorted(ent_align_scores, key=lambda item: item["score"], reverse=True)
            triple["ent-id"] = ent_align_scores[0]["ent-id"]
        all_data.append(new_item)
    assert idx == len(preds)
    json.dump(all_data, open("../data/test_final_results.json", "w"), indent=4, ensure_ascii=False)
                