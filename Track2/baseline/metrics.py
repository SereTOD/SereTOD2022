"""
Copyright 2022 Tsinghua University
Author: Hong Liu (liuhong21@mails.tsinghua.edu.cn)
"""

import math, logging, copy, json, re
from collections import Counter, OrderedDict
from nltk.util import ngrams
from KB_query import query

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):
        '''

        :param parallel_corpus: zip(list of str 1,list of str 2)
        :return: bleu4 score
        '''
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]
        empty_num = 0
        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            if hyps == ['']:
                empty_num += 1
                continue

            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]

            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        # print('empty turns:',empty_num)
        return bleu * 100


def eval_end_to_end(test_data):
    bleu_scorer=BLEUScorer()
    gen_ss, true_ss = [], []
    tp_u, fp_u, fn_u = 0, 0, 0
    tp_s, fp_s, fn_s = 0, 0, 0
    dials_with_requested=0
    success=0
    for dial in test_data:
        KB, goal, log=dial['KB'], dial['goal'], dial['content']
        requested_info=extract_request_info(goal, KB)
        completed=[0 for _ in range(len(requested_info))]
        dial['requested_info']=requested_info
        for turn in log:
            if '用户意图-生成' not in turn:
                continue
            gen_ui, _=get_intent_dict(turn['用户意图-生成'].replace('求助-查询', '询问'), KB)
            true_ui, true_ui2=get_intent_dict(turn['用户意图'].replace('求助-查询', '询问'), KB)
            gen_si=re.sub(r'(\(.*?\))', '', turn['客服意图-生成'])
            true_si=re.sub(r'(\(.*?\))', '', turn['客服意图'])
            gen_s, true_s=turn['客服-生成'], turn['客服']
            gen_ss.append([' '.join(list(gen_s))])
            true_ss.append([' '.join(list(true_s))])
            # calculate the prediction performance of intent
            # compare system intents
            for si in gen_si.split(','):
                if si in true_si:
                    tp_s+=1
                else:
                    fp_s+=1
            for si in true_si:
                if si not in gen_si:
                    fn_s+=1
            # compare user intents
            for ui, info in gen_ui.items():
                if info is None:
                    if ui in true_ui:
                        tp_u+=1
                    else:
                        fp_u+=1
                elif isinstance(info, list):
                    if true_ui.get(ui, None) is None:
                        fp_u+=1
                    else:
                        flag=1
                        for e in info:
                            if e not in true_ui2[ui]:
                                flag=0
                                break
                        if flag:
                            tp_u+=1
                        else:
                            fp_u+=1
                else:
                    logging.info('Unknown format of user intent')

            for ui, info in true_ui.items():
                if info is None:
                    if ui not in gen_ui:
                        fn_u+=1
                elif isinstance(info, list):
                    if gen_ui.get(ui, None) is None:
                        fn_u+=1
                    else:
                        flag=0
                        for e in info:
                            if isinstance(e, list):
                                if not any([t in gen_ui[ui] for t in e]):
                                    flag=1
                                    break
                            else:
                                if e not in gen_ui[ui]:
                                    flag=1
                                    break
                        if flag:
                            fn_u+=1
            # evaluate success
            if len(requested_info)>0:
                for n, info in enumerate(requested_info):
                    for p in info.split(','):
                        if p in gen_s:
                            completed[n]=1
        if len(requested_info)>0:
            dials_with_requested+=1
            if all(completed):
                success+=1

    bleu = bleu_scorer.score(zip(gen_ss, true_ss))
    P_u, P_s=tp_u/(tp_u+fp_u), tp_s/(tp_s+fp_s)
    R_u, R_s=tp_u/(tp_u+fn_u), tp_s/(tp_s+fn_s)
    F1_u, F1_s=2*P_u*R_u/(P_u+R_u), 2*P_s*R_s/(P_s+R_s)
    success_rate=success/dials_with_requested
    eval_result={
        'P/R/F1 for user intent':(P_u, R_u, F1_u),
        'P/R/F1 for system intent':(P_s, R_s, F1_s),
        'BLEU':bleu,
        'Success':success_rate
    }
    return eval_result

def get_intent_dict(intent, KB):
    intent_dict={}
    for act in intent.split(','):
        if '(' not in act:
            intent_dict[act]=None
        else:
            info=re.findall(r'\((.*?)\)', act)
            key=act[:act.index('(')]
            intent_dict[key]=[]
            for e in info:
                if 'ent' in e:
                    ent_id=e[:5]
                    if ent_id not in KB:
                        logging.info('Wrong entity id:{}'.format(ent_id))
                        continue
                    prop=e.split('-')[-1]
                    ent_name=KB[ent_id]['name'].split(',')
                    entry=[en+'-'+prop for en in ent_name]
                    if entry not in intent_dict[key]:
                        intent_dict[key].append(entry)
                else:
                    intent_dict[key].append(e)
    intent_dict2=copy.deepcopy(intent_dict)
    for key, entry in intent_dict2.items():
        if isinstance(entry, list):
            new_entry=[]
            for item in entry:
                if isinstance(item, list):
                    new_entry.extend(item)
                else:
                    new_entry.append(item)
            intent_dict2[key]=new_entry
    return intent_dict, intent_dict2


def extract_request_info(goal, KB):
    requested_info=[]
    for key, ent in goal.items():
        if key=='咨询':
            for item in ent.split(','):
                ent_type=item.split('-')[-1]
                res=query(KB, ent_type=ent_type)
                if res is not None:
                    requested_info.append(','.join(res))
        elif key in KB:
            for prop in goal[key]:
                if '?' in goal[key][prop] and prop in KB[key]:
                    info=[]
                    for value in KB[key][prop].split(','):
                        if value not in goal[key][prop]:
                            info.append(value)
                    if info!=[]:
                        requested_info.append(','.join(info))
    return requested_info

def eval_context_to_response(gens, oracles):
    bleu_scorer=BLEUScorer()
    gen_ss, oracle_ss = [], []
    tp, fp, fn = 0, 0, 0
    for gen, oracle in zip(gens, oracles):
        if '[EOS_SI]' not in oracle:
            logging.info('Test data error: no customer intent')
            continue
        oracle_si, oracle_s = oracle.split('[EOS_SI]')
        oracle_si=oracle_si.split(',')
        oracle_s = oracle_s[:oracle_s.index('[EOS_S]')]
        if '[EOS_SI]' in gen:
            gen_si, gen_s = gen.split('[EOS_SI]')
            gen_si=gen_si.split(',')
            gen_s = gen_s[:gen_s.index('[EOS_S]')]
            for ci in gen_si:
                if ci in oracle_si:
                    tp+=1
                else:
                    fp+=1
            for ci in oracle_si:
                if ci not in gen_si:
                    fn+=1
        else:
            gen_s = gen[:gen.index('[EOS_S]')]
            fn+=len(oracle_si)
        gen_ss.append(gen_s)
        oracle_ss.append(oracle_s)
    wrap_generated = [[_] for _ in gen_ss]
    wrap_truth = [[_] for _ in oracle_ss]
    bleu = bleu_scorer.score(zip(wrap_generated, wrap_truth))
    P=tp/(tp+fp)
    R=tp/(tp+fn)
    F1=2*P*R/(P+R)
    return (R, R, F1), bleu

if __name__=='__main__':
    KB={
      "ent-1": {
        "name": "套餐",
        "type": "主套餐",
        "办理渠道": "得自己发短信办",
        "业务规则": "得今天办明天就明天办不了了"
      },
      "ent-2": {
        "name": "套卡",
        "type": "主套餐",
        "业务费用": "三十八块钱",
        "流量总量": "一个G",
        "通话时长": "两百七十分钟"
      },
      "ent-3": {
        "name": "二十八,二十八套餐",
        "type": "主套餐",
        "业务费用": "二十八",
        "流量总量": "六百兆",
        "通话时长": "二百七,两百七十分钟",
        "业务规则": "存六十每个月返五块",
        "办理渠道": "营业厅"
      }
    }
    intent1="问候, 求助-查询(ent-3-业务费用)(业务)"
    intent2="问候, 求助-查询(二十八套餐-业务费用)(业务)"
    print(get_intent_dict(intent1, KB))
    print(get_intent_dict(intent2, KB))