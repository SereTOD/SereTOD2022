import json 

noisy_tokens = [" ", "_"]
specific_ents = [
    "积分倍享合约",
    "停开机",
    "手机报",
    "139邮箱",
    "来电提醒",
    "和对讲个人版"
]


def correct_offset(value, offset):
    start_shift = 0
    for token in value:
        if token in noisy_tokens:
            start_shift += 1
        else:
            break 
    for noisy_token in noisy_tokens: value = value.replace(noisy_token, "")
    new_start = offset[0] + start_shift
    new_end = new_start + len(value)
    return value, [new_start, new_end]


def filter_noisy_token_pred(data):
    for item in data:
        for ent in item["entities"]:
            value, offset = correct_offset(ent["name"], ent["position"])
            ent["name"] = value 
            ent["position"] = offset
            if ent["name"] in specific_ents:
                ent["type"] = "数据业务"
        for triple in item["triples"]:
            value, offset = correct_offset(triple["value"], triple["position"])
            triple["value"] = value
            triple["position"] = offset
    return data 


def filter_noisy_token_label(data):
    for item in data:
        for ent in item["entities"]:
            for noisy_token in noisy_tokens: ent["name"] = ent["name"].replace(noisy_token, "")
            if ent["name"] in specific_ents:
                ent["type"] = "数据业务"
        for triple in item["triples"]:
            for noisy_token in noisy_tokens: triple["value"] = triple["value"].replace(noisy_token, "")
    return data 





