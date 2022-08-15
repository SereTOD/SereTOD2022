"""
Copyright 2022 Tsinghua University
Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)
"""

import json
import os
import torch
import logging
import random
import re
import numpy as np
import copy
from config import global_config as cfg
from KB_query import query

# End of entity-name-list/user/entity-name/user-intent/KB-result/system-intent/system-response
special_tokens=['[EOS_L]', '[EOS_U]', '[EOS_E]', '[EOS_UI]', '[EOS_K]', '[EOS_SI]', '[EOS_S]']


def convert_to_sequences(data, dial_ids=None):
    sequences=[]
    for dial in data:
        dial_id=dial['id']
        if dial_ids is not None and dial_id not in dial_ids:
            continue
        EN_list={}
        KB=dial['KB']
        for turn in dial['content']:
            pv_EN_list=copy.deepcopy(EN_list)
            ui=turn['用户意图']
            EN=set([])
            KB_result=[]
            if '(' in ui:# there's entity id annotation in the user intent
                for intent in ui.split(','):
                    if '('  not in intent:
                        continue
                    #act=intent[:intent.index('(')]
                    info=re.findall(r'\((.*?)\)', intent)
                    for e in info:
                        if e.startswith('ent'):
                            ent_id=e[:5] 
                            prop=e[5:].strip('-')
                            ent_name=set([])
                            # check whether the ent_id appears in the entities and triples of current turn 
                            if 'info' in turn:
                                for ent in turn['info']['ents']:
                                    if ent['id']==ent_id and ent['name'].strip()!='NA':
                                        EN.add(ent['name'])
                                        ent_name.add(ent['name'])
                                        if ent_id not in EN_list:
                                            EN_list[ent_id]=set([ent['name']])
                                        else:
                                            EN_list[ent_id].add(ent['name'])
                                for triple in turn['info']['triples']:
                                    if triple['ent-id']==ent_id and triple['ent-name'].strip()!='NA':
                                        EN.add(triple['ent-name'])
                                        ent_name.add(triple['ent-name'])
                                        if ent_id not in EN_list:
                                            EN_list[ent_id]=set([triple['ent-name']])
                                        else:
                                            EN_list[ent_id].add(triple['ent-name'])
                            if len(EN)==0 and ent_id in EN_list:
                                # no entity info in current turn annotation, then query history information
                                EN=EN.union(EN_list[ent_id])
                                ent_name=ent_name.union(EN_list[ent_id])
                            ent_name=list(ent_name)
                            if ent_name!=[]:
                                ent_name_lens=[len(item) for item in ent_name]
                                max_len_id=ent_name_lens.index(max(ent_name_lens))
                                max_len_ent=ent_name[max_len_id]
                                if max_len_ent.startswith('ent'):
                                    max_len_ent=max_len_ent[5:].strip('-')
                                ui=ui.replace(ent_id, max_len_ent)
                            else:
                                ui=ui.replace(ent_id+'-', '')
                            # query database
                            res=query(KB, ent_id=ent_id, prop=prop)
                        elif e in ['业务','数据业务','套餐', '主套餐','附加套餐','国际漫游业务','流量包','长途业务','4G套餐','5G套餐']:
                            res=query(KB, ent_type=e)
                        else:
                            res=query(KB, prop=e)
                        if res is not None:
                            if isinstance(res, list):
                                KB_result.append(','.join(res))
                            else:
                                KB_result.append(res)
            if 'info' in turn:
                for ent in turn['info']['ents']:
                    if ent['name'] in turn['用户'] and ent['name'].strip()!='NA': # this is an entity name mentioned by the user in current turn
                        EN.add(ent['name'])
                        ent_id=ent['id']
                        if ent_id not in EN_list:
                            EN_list[ent_id]=set([ent['name']])
                        else:
                            EN_list[ent_id].add(ent['name'])
                for triple in turn['info']['triples']: 
                    if triple['value'] in turn['用户'] and triple['ent-name'].strip()!='NA' and triple['ent-id']!='NA':
                        # this is an triple with ent-name and ent-id mentioned by the user in current turn
                        EN.add(triple['ent-name'])
                        ent_id=triple['ent-id']
                        if ent_id not in EN_list:
                            EN_list[ent_id]=set([triple['ent-name']])
                        else:
                            EN_list[ent_id].add(triple['ent-name'])
            pv_EN_seq=','.join([','.join(list(item)) for item in list(pv_EN_list.values())])
            si=turn['客服意图']
            si=re.sub(r'\(.*\)', '', si)
            ui=re.sub(r'ent-[0-9]+-', '', ui)
            ui=re.sub(r'ent--', '', ui)

            sequence=pv_EN_seq.lower()+'[EOS_L]'+turn['用户'].lower()+'[EOS_U]'\
                +','.join(list(EN)).lower()+'[EOS_E]'+ui.lower()+'[EOS_UI]'+','.join(KB_result).lower()+'[EOS_K]'\
                +si.lower()+'[EOS_SI]'+turn['客服'].lower()+'[EOS_S]'
            sequences.append(sequence)
    return sequences

def read_data(tokenizer):
    encoded_path=os.path.join(cfg.data_dir, 'encoded_data.json')
    if not os.path.exists(encoded_path):
        data=json.load(open(cfg.data_path, 'r', encoding='utf-8'))
        dial_ids=[dial['id'] for dial in data]
        random.shuffle(dial_ids)
        piece=len(dial_ids)//10
        train_ids, dev_ids, test_ids=dial_ids[:8*piece], dial_ids[8*piece:9*piece], dial_ids[9*piece:]
        train_seqs=convert_to_sequences(data, train_ids)
        dev_seqs=convert_to_sequences(data, dev_ids)
        test_seqs=convert_to_sequences(data, test_ids)
        logging.info('Dialogs -- Train:{}, dev:{}, test:{}'.format(len(train_ids), len(dev_ids), len(test_ids)))
        logging.info('Sequences -- Train:{}, dev:{}, test:{}'.format(len(train_seqs), len(dev_seqs), len(test_seqs)))
        seq_data={
            'train':train_seqs,
            'dev':dev_seqs,
            'test':test_seqs
        }
        json.dump(seq_data, open(os.path.join(cfg.data_dir, 'sequences.json'), 'w'), ensure_ascii=False)
        dial_id_data={
            'train':train_ids,
            'dev':dev_ids,
            'test':test_ids
        }
        json.dump(dial_id_data, open(os.path.join(cfg.data_dir, 'dial_ids.json'), 'w'))
        logging.info('Encoding data ...')
        encoded_data={}
        for s in ['train', 'dev', 'test']:
            encoded_data[s]=[]
            for seq in seq_data[s]:
                encoded_data[s].append(tokenizer.encode(seq))
        json.dump(encoded_data, open(encoded_path, 'w'))
        logging.info('Data encoded, saved in:{}'.format(encoded_path))
    else:
        logging.info('Reading encoded data from:{}'.format(encoded_path))
        encoded_data=json.load(open(encoded_path, 'r'))
    logging.info('Train:{}, dev:{}, test:{}'.format(len(encoded_data['train']), len(encoded_data['dev']), len(encoded_data['test'])))
    return encoded_data

def extract_test_dial(data='test'):
    if cfg.test_path=='':
        dial_ids=json.load(open(os.path.join(cfg.data_dir, 'dial_ids.json'), 'r', encoding='utf-8'))
        all_data=json.load(open(cfg.data_path, 'r', encoding='utf-8'))
        test_data=[]
        for dial in all_data:
            if dial['id'] in dial_ids[data]:
                test_data.append(dial)
    else:
        test_data=json.load(open(cfg.test_path, 'r', encoding='utf-8'))
    return test_data

def train_collate_fn(batch):
    pad_batch = padSeqs(batch, cfg.pad_id)
    batch_tensor=torch.from_numpy(pad_batch).long()
    return batch_tensor

def test_collate_fn(batch, sep_id):
    # prediction
    # sep_id: the token id that divides input context and target context
    inputs, labels = [], []
    for seq in batch:
        idx=seq.index(sep_id)
        inputs.append(seq[:idx+1])
        labels.append(seq[idx+1:])
    return [inputs, labels]

def padSeqs(sequences, pad_id, maxlen=None):
    lengths = [len(x) for x in sequences]
    maxlen=max(lengths)
    maxlen=min(1024, maxlen)
    
    pad_batch=np.ones((len(sequences), maxlen))*pad_id
    for idx, s in enumerate(sequences):
        trunc = s[-maxlen:]
        pad_batch[idx, :len(trunc)] = trunc
            
    return pad_batch

def integrate_result(inputs, gens, oracles):
    results=[]
    for context, gen, oracle in zip(inputs, gens, oracles):
        EN_list, left=context.split('[EOS_L]')
        EN_list=EN_list.strip('[CLS]')
        user, left = left.split('[EOS_U]')
        EN, left=left.split('[EOS_E]')
        user_intent, KB_result=left.split('[EOS_UI]')
        KB_result=KB_result.strip('[EOS_K]')
        service_intent, service=oracle.split('[EOS_SI]')
        service=service[:service.index('[EOS_S]')]
        if '[EOS_SI]' in gen:
            service_intent_gen, service_gen=gen.split('[EOS_SI]')
            service_gen=service_gen[:service_gen.index('[EOS_S]')]
        else:
            service_intent_gen=''
            service_gen=gen[:gen.index('[EOS_S]')]
        entry={
            '用户':user.replace(' ', ''),
            '用户意图':user_intent.replace(' ', ''),
            '实体列表':EN_list.replace(' ', ''),
            '实体':EN.replace(' ', ''),
            '数据库结果':KB_result.replace(' ', ''),
            '客服':service.replace(' ', ''),
            '客服意图':service_intent.replace(' ', ''),
            '客服-生成':service_gen.replace(' ', ''),
            '客服意图-生成':service_intent_gen.replace(' ', '')
        }

        results.append(entry)
    return results

def batch_align(contexts,left_len,return_attn=False):
    max_len=max([len(context) for context in contexts])
    max_len=min(1024-left_len, max_len)
    new_contexts=[]
    attentions=[]
    for id, context in enumerate(contexts):
        if len(context)<max_len:
            new_context=(max_len-len(context))*[cfg.pad_id]+context
            attention=(max_len-len(context))*[0]+len(context)*[1]
        else:
            new_context=context[-max_len:]
            attention=len(new_context)*[1]
        new_contexts.append(new_context)
        attentions.append(attention)
    if return_attn:
        return new_contexts, attentions
    return new_contexts

if __name__=='__main__':
    data=json.load(open(cfg.data_path, 'r', encoding='utf-8'))
    dial_ids=list(data.keys())
    random.shuffle(dial_ids)
    piece=len(dial_ids)//10
    train_ids, dev_ids, test_ids=dial_ids[:8*piece], dial_ids[8*piece:9*piece], dial_ids[9*piece:]
    train_seqs=convert_to_sequences(data, train_ids)
    dev_seqs=convert_to_sequences(data, dev_ids)
    test_seqs=convert_to_sequences(data, test_ids)
    print('Train:{}, dev:{}, test:{}'.format(len(train_seqs), len(dev_seqs), len(test_seqs)))
    seq_data={
        'train':train_seqs,
        'dev':dev_seqs,
        'test':test_seqs
    }
    json.dump(seq_data, open(os.path.join(cfg.data_dir, 'sequences.json'), 'w'), indent=2, ensure_ascii=False)