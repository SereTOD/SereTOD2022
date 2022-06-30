import json
import random
import copy
import re
def data_statistics():
    data=json.load(open('data/Raw_data.json', 'r', encoding='utf-8'))
    c1, c2 = 0, 0
    c3=0
    dials1, dials2, dials3 = [], [], []
    dials_std=[]
    print(len(data))
    for dial in data:
        user_first=0
        service_first=0
        other=0
        confusion=0
        speakers=set()
        for turn in dial:
            temp=list(turn.keys())
            service, user=temp[:2]
            speakers.add(service)
            speakers.add(user)
            if '客服意图' in temp and '用户意图' in temp:
                if '意图混乱' in temp or '意图混乱' in turn['用户意图'] or '意图混乱' in turn['客服意图']:
                    confusion=1
                    continue
                if temp.index('客服意图')<temp.index('用户意图'):#客服在前
                    c1+=1
                    service_first=1
                else:
                    c2+=1
                    user_first=1
            else:
                c3+=1
                other=1
        if other or confusion:
            dials3.append(dial)
        elif user_first: # 对话中至少有一轮用户在前
            if len(speakers)==2 and not service_first:
                dials2.append(dial)
        else:
            dials1.append(dial)
            if len(speakers)==2:
                dials_std.append(dial)
    print('轮次——客服在前:', c1, '用户在前:', c2, '其他情况:', c3)
    print('对话——客服在前:', len(dials1), '用户在前:', len(dials2), '其他情况:', len(dials3))
    print('标准对话数:', len(dials_std))
    json.dump(dials1, open('data/part1.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dials2, open('data/part2.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dials3, open('data/part3.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(dials_std, open('data/std_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def clear_data(data):
    dial_count, turn_count=0,0
    new_data=[]
    remaining_data=[]
    for d, dial in enumerate(data):
        dial_key_mission=0
        for t, turn in enumerate(dial):
            turn_key_mission=0
            if "info" in turn:
                if 'ents' not in turn['info']:
                    turn['info']['ents']=[]
                if 'triples' not in turn['info']:
                    turn['info']['triples']=[]
                for ent in turn['info']['ents']:
                    for key in ['name', 'id', 'type', 'pos']:
                        if key not in ent:
                            #print('Entities of dial {} turn {} has no {}'.format(d, t, key))
                            if 'ent-'+key in ent:
                                ent[key]=ent.pop('ent-'+key)
                            else:
                                #print('Entities of dial {} turn {} has no {}'.format(d, t, key))
                                if 'missing' not in ent:
                                    ent['missing']=key
                                else:
                                    ent['missing']+=','+key
                                dial_key_mission=1
                                turn_key_mission=1
                for triple in turn['info']['triples']:
                    for key in ['ent-id', 'ent-name', 'prop', 'value']:
                        if key not in triple:
                            #print('Triples of dial {} turn {} has no {}'.format(d, t, key))
                            if key=='ent-id' and 'id' in triple:
                                triple[key]=triple.pop('id')
                                #print('Triples of dial {} turn {} has {}'.format(d, t, 'id'))
                            elif key=='ent-name' and 'name' in triple:
                                triple[key]=triple.pop('name')
                                #print('Triples of dial {} turn {} has {}'.format(d, t, 'name'))
                            else:
                                #print('Triples of dial {} turn {} has no {}'.format(d, t, key))
                                if 'missing' not in triple:
                                    triple['missing']=key
                                else:
                                    triple['missing']+=','+key
                                dial_key_mission=1
                                turn_key_mission=1
                    if 'ref' in triple:
                        triple.pop('ref')
            turn_count+=turn_key_mission
        dial_count+=dial_key_mission
        if not dial_key_mission:
            new_data.append(dial)
        else:
            remaining_data.append(dial)
    print('Dials with missing keys:{}, turns with missing keys:{}'.format(dial_count, turn_count))
    return new_data, remaining_data

def restructure():
    data=json.load(open('data/std_data.json', 'r', encoding='utf-8'))
    data1=json.load(open('data/part2.json', 'r', encoding='utf-8'))
    data, _=clear_data(data)
    data1,_=clear_data(data1)
    new_data=[]
    missing_pos=0
    for dial in data:
        service, user=list(dial[0].keys())[:2]
        new_dial=[]
        for n in range(len(dial)-1):
            turn1=dial[n]
            turn2=dial[n+1]
            new_turn={}
            new_turn['用户']=turn1[user].replace('[UNK]', '')
            new_turn['客服']=turn2[service].replace('[UNK]', '')
            new_turn['用户意图']=turn1['用户意图']
            new_turn['客服意图']=turn2['客服意图']
            new_turn.update({"info": {
                "ents":[],
			    "triples":[]
                }
            })
            if 'info' in turn1:
                for ent in turn1['info']['ents']:
                    pos=[]
                    for p in ent['pos']:
                        if p==[]:
                            continue
                        if p[0]==2:
                            pt=copy.deepcopy(p)
                            pt[0]=1
                            pos.append(pt)
                    if len(pos)>0:
                        new_ent=copy.deepcopy(ent)
                        new_ent['pos']=pos
                        new_turn['info']['ents'].append(new_ent)

                for triple in turn1['info']['triples']:
                    if triple['value'] in turn1[user]:
                        new_turn['info']['triples'].append(triple)
            if 'info' in turn2:
                for ent in turn2['info']['ents']:
                    pos=[]
                    for p in ent['pos']:
                        if p==[]:
                            missing_pos+=1
                            continue
                        if p[0]==1:
                            pt=copy.deepcopy(p)
                            pt[0]=2
                            pos.append(pt)
                    if len(pos)>0:
                        new_ent=copy.deepcopy(ent)
                        new_ent['pos']=pos
                        new_turn['info']['ents'].append(new_ent)

                for triple in turn2['info']['triples']:
                    if triple['value'] in turn2[service]:
                        new_turn['info']['triples'].append(triple)
            new_dial.append(new_turn)
        new_data.append(new_dial)
    
    for dial in data1:
        user, service=list(dial[0].keys())[:2]
        new_dial=[]
        for turn in dial:
            turn['用户']=turn.pop(user).replace('[UNK]', '')
            turn['客服']=turn.pop(service).replace('[UNK]', '')
            new_dial.append(turn)
        new_data.append(new_dial)
    print('Total restructured data:', len(new_data))
    json.dump(new_data, open('data/restructured_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def extract_local_KB():
    data=json.load(open('data/restructured_data.json', 'r', encoding='utf-8'))
    new_data={}
    count=0
    turn_num=0
    query_num=0
    query_dial=0
    for n, dial in enumerate(data):
        new_data[n]={
            'KB':{},
            'goal':{}
            }
        new_data[n]['log']=dial
        KB={}
        goal={}
        with_query=0
        # extract db
        for turn in dial:
            turn_num+=1
            turn['用户意图']=turn['用户意图'].replace('（', '(').replace('）',')').replace('，',',')
            turn['客服意图']=turn['客服意图'].replace('（', '(').replace('）',')').replace('，',',')
            if '(' in turn['用户意图']:
                query_num+=1
                with_query=1
            if 'info' in turn:
                for ent in turn['info']['ents']:
                    if ent['name'] in turn['用户']:#只有用户提到的才加到goal中
                        if ent['id'] not in goal:
                            goal[ent['id']]={'type':ent['type']}
                        if 'name' not in goal[ent['id']]:
                            goal[ent['id']]['name']=set([ent['name']])
                        else:
                            goal[ent['id']]['name'].add(ent['name'])

                    if ent['id'] not in KB:
                        KB[ent['id']]={
                            'name':set([ent['name'].lower()]),
                            'type':ent['type'].lower()
                        }
                    else:# we accumulate all the names for one entity
                        KB[ent['id']]['name'].add(ent['name'].lower())
                for triple in turn['info']['triples']:
                    if triple['value'] in turn['用户']:
                        if triple['ent-id'].startswith('ent') and triple['ent-id'] not in goal:
                            goal[triple['ent-id']]={'name':set([triple['ent-name']])}
                        if triple['ent-id'] not in goal:   
                            goal[triple['ent-id']]={}
                        if triple['prop'] not in goal[triple['ent-id']]:
                            goal[triple['ent-id']][triple['prop']]=set([triple['value']])
                        else:
                            goal[triple['ent-id']][triple['prop']].add(triple['value'])

                    if triple['ent-id'].startswith('ent') and triple['ent-id'] not in KB:
                        #print('Triple appeare before entity:', triple['ent-id'])
                        KB[triple['ent-id']]={'name':set([triple['ent-name'].lower()])}
                        count+=1
                    if triple['ent-id'] not in KB:   
                        KB[triple['ent-id']]={}
                    if triple['prop'] not in KB[triple['ent-id']]:
                        KB[triple['ent-id']][triple['prop'].lower()]=set([triple['value']])
                    else:
                        KB[triple['ent-id']][triple['prop'].lower()].add(triple['value'])
            
            ui=turn['用户意图'] #user intent
            if '(' in ui:
                for intent in ui.split(','):
                    if '(' in intent:
                        act=intent[:intent.index('(')]
                        info=re.findall(r'\((.*?)\)', intent)
                        for e in info:
                            e=e.strip('(').strip(')')
                            if e in ['业务','数据业务','套餐', '主套餐','附加套餐','国际漫游业务','流量包','长途业务','4G套餐','5G套餐']:
                                item=act+'-'+e
                                if '咨询' not in goal:
                                    goal['咨询']=[item]
                                elif item not in goal['咨询']:
                                    goal['咨询'].append(item)
                            elif '-' in e:
                                ent_id=e[:5]
                                prop=e[5:].strip('-')
                                if ent_id in goal:
                                    if prop in goal[ent_id]:
                                        goal[ent_id][prop].add('?')
                                    else:
                                        goal[ent_id][prop]=set(['?'])
                                    if '用户意图' in goal[ent_id]:
                                        goal[ent_id]['用户意图'].add(act)
                                    else:
                                        goal[ent_id]['用户意图']=set([act])
                                else:
                                    goal[ent_id]={prop:set(['?']), '用户意图':set([act])}
                                
                            else: #查询个人信息
                                if 'NA' in goal:
                                    if e in goal['NA']:
                                        goal['NA'][e].add('?')
                                    else:
                                        goal['NA'][e]=set(['?'])
                                    if '意图' in goal['NA']:
                                        goal['NA']['意图'].add(act)
                                    else:
                                        goal['NA']['意图']=set([act])
                                else:
                                    goal['NA']={e:set(['?']), '意图':set([act])}
        if with_query:
            query_dial+=1
        for id, ent in KB.items():
            for key, value in ent.items():
                if isinstance(value, set):
                    KB[id][key]=','.join(list(value))
        for id, ent in goal.items():
            if isinstance(ent, list):
                goal[id]=','.join(ent)
            else:
                for key, value in ent.items():
                    if isinstance(value, set):
                        goal[id][key]=','.join(list(value))
        new_data[n]['KB']=KB
        new_data[n]['goal']=goal
    print('Triple appeare before entity:', count)
    print('Total turns:', turn_num, 'query turns:', query_num)
    print('Total dials:', len(new_data), 'query dials:', query_dial)
    json.dump(new_data, open('data/processed_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def intent_norm():
    data=json.load(open('data/processed_data.json', 'r', encoding='utf-8'))
    for dial_id, dial in data.items():
        for turn in dial['log']:
            turn['用户意图']=turn['用户意图'].replace(',ent', ')(ent')
            turn['用户意图']=turn['用户意图'].replace('))', ')')
            turn['用户意图']=turn['用户意图'].replace('),(', ')(')
            turn['用户意图']=turn['用户意图'].replace(',(', ')(')
            info=re.findall(r'\((.*?)\)',turn['用户意图'])
            for e in info:
                if ',' in e:
                    e_list=[]
                    ent_id=None
                    for item in e.split(','):
                        if item.startswith('ent'):
                            ent_id=item[:5]
                            e_list.append(item)
                        elif item in ['业务','数据业务','套餐', '主套餐','附加套餐','国际漫游业务','流量包','长途业务','4G套餐','5G套餐']:
                            # type
                            e_list.append(item)
                        elif item in ['用户需求','用户要求','用户状态', '短信', '持有套餐','账户余额','流量余额', "话费余额", '欠费']:
                            # prop
                            e_list.append(item)
                        else:
                            # property of enity
                            if ent_id is not None:
                                e_list.append(ent_id+'-'+item)
                            else:
                                e_list.append(item)
                    ui=turn['用户意图']
                    turn['用户意图']=turn['用户意图'].replace(e, ')('.join(e_list))
                    print(ui, turn['用户意图'])
    json.dump(data, open('data/processed_data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)


def add_constraint():
    data=json.load(open('data/processed_data.json', 'r', encoding='utf-8'))
    query_num, query_wo_cons, query_cons_in_triple=0, 0, 0
    inquiry_num, inquiry_wo_cons, inquiry_cons_in_triple=0, 0, 0
    collected_turns=[]
    add_num=0
    for dial_id, dial in data.items():
        for turn in dial['log']:
            collected=False
            ui=turn['用户意图']
            for query_intent in ['求助-查询', '询问', '主动确认']:
                if query_intent in ui:
                    query_num+=1
                    if query_intent+'(' not in ui:
                        added_cons=''
                        query_wo_cons+=1
                        if 'info' in turn:
                            for tri in turn['info']['triples']:
                                if tri['value'] in turn['用户']:
                                    if tri['ent-id'].startswith('ent'):
                                        cons=tri['ent-id']+'-'+tri['prop']
                                    elif tri['ent-id']=='NA':
                                        cons=tri['prop']
                                    added_cons+='('+cons+')'
                                    #query_cons_in_triple+=1
                                    if not collected:
                                        collected_turns.append(turn)
                                        collected=True
                        if added_cons!='':
                            turn['用户意图']=turn['用户意图'].replace(query_intent, query_intent+added_cons)
                            add_num+=1
                            break
    #print('查询意图:', query_num, '无约束的查询意图', query_wo_cons, '查询约束存在于三元组标注中', query_cons_in_triple)
    #print('询问意图:', inquiry_num, '无约束的询问意图', inquiry_wo_cons, '询问约束存在于三元组标注中', inquiry_cons_in_triple)
    #print('Collected turns:', len(collected_turns))
    json.dump(collected_turns, open('data/temp.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('Add constraints num:', add_num)
    json.dump(data, open('data/data.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

if __name__=='__main__':
    data_statistics()
    intent_norm()
    extract_local_KB()
    add_constraint()