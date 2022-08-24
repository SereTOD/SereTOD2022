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
            for spe_ent in specific_ents:
                if spe_ent in ent["name"]:
                    ent["type"] = "数据业务"
                    break 
        for triple in item["triples"]:
            value, offset = correct_offset(triple["value"], triple["position"])
            triple["value"] = value
            triple["position"] = offset
    return data 


def filter_noisy_token_label(data):
    for item in data:
        for ent in item["entities"]:
            value, offset = correct_offset(ent["name"], ent["position"])
            ent["name"] = value 
            ent["position"] = offset
            for spe_ent in specific_ents:
                if spe_ent in ent["name"]:
                    ent["type"] = "数据业务"
                    break
        for triple in item["triples"]:
            for noisy_token in noisy_tokens: triple["value"] = triple["value"].replace(noisy_token, "")
    return data 





