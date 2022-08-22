# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
import os 
import json 


def get_submissions(result_file):
    results = json.load(open(result_file))
    final_submissions = []
    for item in results:
        new_item = {
            "id": item["id"],
            "entities": item["entities"],
            "triples": item["triples"]
        }
        for ent in new_item["entities"]:
            ent["turn_id"] = ent["utterance_id"] // 2
            ent.pop("utterance_id", None)
        for triple in new_item["triples"]:
            triple["turn_id"] = triple["utterance_id"] // 2
            triple.pop("utterance_id", None)
        final_submissions.append(new_item)
    json.dump(final_submissions, open("submissions.json", "w"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    get_submissions("data/test_final_results.json")