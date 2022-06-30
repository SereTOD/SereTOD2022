def query(KB, ent_id=None, ent_name=None, prop=None, ent_type=None):
    if KB=={}:
        return None
    if ent_id is not None:
        if ent_id not in KB:
            return None
        return KB[ent_id].get(prop, None) if prop else None
    elif ent_name is not None:# use entity name to query local KB
        value=None
        flag=0
        for en in ent_name.split(','):
            for key, ent in KB.items():
                if not key.startswith('ent'):
                    continue
                if en.lower() in ent['name'].split(','):
                    value=ent.get(prop, None)
                    if value is not None:# The corresponding value has been found from the current entity
                        flag=1
                        break
            if flag:
                break
        return value
    elif prop is not None:# query the user information, entity id or name is not needed
        user_info=['用户需求','用户要求','用户状态', '短信', '持有套餐','账户余额','流量余额', "话费余额", '欠费']
        value=None
        if 'NA' in KB:
            value=KB['NA'].get(prop, None)
            if value is None and prop in ['剩余话费', '话费余额']:
                for key in ['剩余话费', '话费余额']:
                    if key in KB['NA']:
                        value=KB['NA'][key]
                        break
        if value is None: # irregular query
            value_list=[]
            for key, ent in KB.items():
                if prop in ent:
                    value_list.append(ent[prop])
            if value_list!=[]:
                value=','.join(value_list)
        return value
    elif ent_type is not None: # Ask which entities are of this type
        names=[]
        for key, ent in KB.items():
            if not key.startswith('ent') or 'type' not in ent:
                continue
            if ent['type']==ent_type.lower():
                names.append(ent['name'])
        return names if len(names)>0 else None

def intent_query(KB, intent):
    pass
