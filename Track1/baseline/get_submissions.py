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
        final_submissions.append(new_item)
    json.dump(final_submissions, open("submissions.json", "w"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    get_submissions("data/test_final_results.json")