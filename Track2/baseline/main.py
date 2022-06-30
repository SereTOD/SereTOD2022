from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel
from transformers import BertTokenizer
from reader import *
from metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os, shutil
import random
import argparse
import time
import logging
import json
from tqdm import tqdm
import numpy as np
import copy, re
from torch.utils.tensorboard import SummaryWriter
from config import global_config as cfg

class Model(object):
    def __init__(self, device='cuda:0'):
        self.device=device
        self.tokenizer = BertTokenizer.from_pretrained(cfg.gpt_path)
        self.model=GPT2LMHeadModel.from_pretrained(cfg.gpt_path)
        self.model.to(self.device)
        if cfg.mode=='train':
            json.dump(cfg.__dict__,open(os.path.join(cfg.exp_path,'cfg_all.json'),'w'),indent=2)
        
            # Add special tokens
            init_vocab_size=len(self.tokenizer)
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            logging.info('Added special tokens:{}'.format(special_tokens))
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logging.info('Special token added, vocab size:{}-->{}'.format(init_vocab_size, len(self.tokenizer)))

        # log
        log_path='./log/log_{}'.format(cfg.exp_name)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
            os.mkdir(log_path)
        else:
            os.mkdir(log_path)
        self.tb_writer = SummaryWriter(log_dir=log_path)
    
    def train(self):
        encoded_data=read_data(self.tokenizer)
        train_dataloader=DataLoader(encoded_data['train'], batch_size=cfg.batch_size, shuffle=True, collate_fn=train_collate_fn)
        dev_dataloader=DataLoader(encoded_data['dev'], batch_size=cfg.eval_batch_size, collate_fn=train_collate_fn)
        optimizer, scheduler = self.get_optimizers(len(encoded_data['train']), self.model)
        log_inputs = 2
        global_step = 0
        min_loss=10000

        for epoch in range(cfg.epoch_num):
            tr_loss = 0.0
            step_loss=0
            btm = time.time()
            oom_time = 0
            self.model.zero_grad()

            for batch_idx, batch in enumerate(train_dataloader):
                try:  # avoid OOM
                    self.model.train()
                    if log_inputs > 0:  # log inputs for the very first two turns
                        logging.info('Training Sequences:')
                        logging.info(self.tokenizer.decode(batch[0,:]))
                        log_inputs-=1
                    inputs=batch.to(self.device) #B, T
                    labels=inputs
                    outputs = self.model(inputs)
                    loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                    loss=loss/cfg.gradient_accumulation_steps
                    loss.backward()
                    tr_loss += loss.item()
                    step_loss+=loss.item()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    if (batch_idx+1) % cfg.gradient_accumulation_steps == 0 or batch_idx+1==len(train_dataloader):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        self.tb_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step)
                        self.tb_writer.add_scalar('loss', step_loss, global_step)
                        step_loss=0

                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        logging.info("WARNING: ran out of memory,times: {}, batch size: {}".format(oom_time, cfg.batch_size))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logging.info(str(exception))
                        raise exception
            eval_loss=self.eval(dev_dataloader)
            logging.info('Epoch:{}, Train epoch time:{:.2f} min, epoch loss:{:.3f}, eval loss:{:.3f}'.format(epoch, (time.time()-btm)/60, tr_loss, eval_loss))
            self.tb_writer.add_scalar('eval_loss', eval_loss, epoch)
            if eval_loss<min_loss:
                min_loss=eval_loss
                self.save_model()
 
    def get_optimizers(self, num_samples, model):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
        #print(num_samples, cfg.epoch_num, cfg.gradient_accumulation_steps, cfg.batch_size)
        num_training_steps = num_samples*cfg.epoch_num // (cfg.gradient_accumulation_steps*cfg.batch_size)
        num_warmup_steps = int(num_training_steps*cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,\
            num_training_steps=num_training_steps)
        return optimizer, scheduler
    
    def calculate_loss_and_accuracy(self, outputs, labels):
        # GPT2-chicahat/train.py
        lm_logits = outputs[0]

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=cfg.pad_id, reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # avg loss
        not_ignore = shift_labels.ne(cfg.pad_id)
        num_targets = not_ignore.long().sum().item()

        loss /= num_targets
        return loss
    
    def eval(self, data):
        self.model.eval()
        total_loss=0
        with torch.no_grad():
            for batch in data:
                inputs=batch.to(self.device) #B, T
                labels=inputs
                outputs = self.model(inputs)
                loss = self.calculate_loss_and_accuracy(outputs, labels=labels)
                total_loss+=loss.item()
        return total_loss/len(data)

    def save_model(self, path=None, model=None):
        save_path = os.path.join(cfg.exp_path, path) if path else os.path.join(cfg.exp_path, 'best_model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.info('Saving model checkpoint to %s', save_path)
        if not model:
            self.model.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def test_end_to_end(self, data='test'):
        self.model.eval()
        result_path=os.path.join(cfg.gpt_path, 'e2e_result.json')
        if os.path.exists(result_path):
            test_data=json.load(open(result_path, 'r', encoding='utf-8'))
        else:#generate results
            test_data=extract_test_dial(data=data)
            st=time.time()
            dial_num=0
            turn_num=0
            with torch.no_grad():
                for key, dial in tqdm(test_data.items()):
                    dial_num+=1
                    #if dial_num==5:
                    #   break
                    KB, goal=dial['KB'], dial['goal']
                    EN_list=set([])
                    for turn in dial['log']:
                        turn_num+=1
                        pv_EN_list=copy.deepcopy(EN_list)
                        EN_list_seq=','.join(list(pv_EN_list))
                        context=EN_list_seq+'[EOS_L]'+turn['用户']+'[EOS_U]'
                        context_ids=self.tokenizer.encode(context)[:-1]
                        # predict entity names mentioned in this turn
                        #logging.info(self.tokenizer.decode(context_ids).replace(' ', ''))
                        max_len=len(context_ids)+15
                        eos_id=self.tokenizer.convert_tokens_to_ids('[EOS_E]')
                        outputs = self.model.generate(input_ids=torch.tensor([context_ids]).to(self.model.device),
                                                pad_token_id=cfg.pad_id, max_length=max_len, eos_token_id=eos_id)
                        generated = outputs[0].cpu().numpy().tolist()
                        if eos_id not in generated:
                            generated[-1]=eos_id
                        EN=self.tokenizer.decode(generated[len(context_ids):-1]).replace(' ', '')
                        # delete repetition
                        current_EN_set=set(EN.split(','))
                        EN=','.join(list(current_EN_set))
                        if EN!='':
                            EN_list=EN_list.union(current_EN_set)
                        # predict user intent
                        context_ids=generated
                        #logging.info(self.tokenizer.decode(context_ids).replace(' ', ''))
                        max_len=len(context_ids)+25
                        eos_id=self.tokenizer.convert_tokens_to_ids('[EOS_UI]')
                        outputs = self.model.generate(input_ids=torch.tensor([context_ids]).to(self.model.device),
                                                pad_token_id=cfg.pad_id, max_length=max_len, eos_token_id=eos_id)
                        generated = outputs[0].cpu().numpy().tolist()
                        if eos_id not in generated:
                            generated[-1]=eos_id
                        UI=self.tokenizer.decode(generated[len(context_ids):-1]).replace(' ', '')
                        # query local database
                        KB_result=[]
                        if '(' in UI:
                            for intent in UI.split(','):
                                if '('  not in intent:
                                    continue
                                act=intent[:intent.index('(')]
                                info=re.findall(r'\((.*?)\)', intent)
                                for e in info:
                                    e=e.strip('-')
                                    if '-' in e:
                                        if e.split('-')!=2:
                                            continue
                                        ent_name, prop=e.split('-')
                                        res=query(KB, ent_name=ent_name, prop=prop)
                                    elif e in ['业务','数据业务','套餐', '主套餐','附加套餐','国际漫游业务','流量包','长途业务','4G套餐','5G套餐']:
                                        res=query(KB, ent_type=e)
                                    else:
                                        res=query(KB, prop=e)
                                    if res is not None:
                                        if isinstance(res, list):
                                            KB_result.append(','.join(res))
                                        else:
                                            KB_result.append(res)
                        KB_seq=','.join(KB_result)
                        # generate system intent
                        context_ids=generated+self.tokenizer.encode(KB_seq+'[EOS_K]')[1:-1]
                        #logging.info(self.tokenizer.decode(context_ids).replace(' ', ''))
                        max_len=len(context_ids)+10
                        eos_id=self.tokenizer.convert_tokens_to_ids('[EOS_SI]')
                        outputs = self.model.generate(input_ids=torch.tensor([context_ids]).to(self.model.device),
                                                pad_token_id=cfg.pad_id, max_length=max_len, eos_token_id=eos_id)
                        generated = outputs[0].cpu().numpy().tolist()
                        if eos_id not in generated:
                            generated[-1]=eos_id
                        SI=self.tokenizer.decode(generated[len(context_ids):-1]).replace(' ', '')
                        # generate system response
                        context_ids=generated
                        #logging.info(self.tokenizer.decode(context_ids).replace(' ', ''))
                        max_len=len(context_ids)+60
                        eos_id=self.tokenizer.convert_tokens_to_ids('[EOS_S]')
                        outputs = self.model.generate(input_ids=torch.tensor([context_ids]).to(self.model.device),
                                                pad_token_id=cfg.pad_id, max_length=max_len, eos_token_id=eos_id)
                        generated = outputs[0].cpu().numpy().tolist()
                        if eos_id not in generated:
                            generated[-1]=eos_id
                        resp=self.tokenizer.decode(generated[len(context_ids):-1]).replace(' ', '')
                        # delete repetition
                        repeated=re.findall(r'(.{3,12})\1+', resp)
                        for p in repeated:
                            if p in resp:
                                idx=resp.index(p)+len(p)
                                resp=resp[:idx]
                        turn['history_ents']=EN_list_seq
                        turn['current_ents']=EN
                        turn['用户意图-生成']=UI
                        turn['查询结果']=KB_seq
                        turn['客服意图-生成']=SI
                        turn['客服-生成']=resp
                        if 'info' in turn:
                            turn.pop('info')
            logging.info('Dial num:{}, turn num:{}, testing time:{:.3f} min'.format(dial_num, turn_num, (time.time()-st)/60))
            json.dump(test_data, open(result_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        eval_result=eval_end_to_end(test_data)
        logging.info(eval_result)

    def test_context_to_resp(self, data='test'):
        encoded_data=read_data(self.tokenizer)
        test_data=encoded_data['test'] if data=='test' else encoded_data['dev']
        sep_id=self.tokenizer.convert_tokens_to_ids('[EOS_K]')
        eos_id=self.tokenizer.convert_tokens_to_ids('[EOS_S]')
        test_dataloader=DataLoader(test_data, batch_size=cfg.eval_batch_size, collate_fn=lambda x:test_collate_fn(x, sep_id), shuffle=False)
        self.model.eval()
        max_len=50
        gens, oracles, contexts = [], [], []
        st=time.time()
        with torch.no_grad():
            for batch in test_dataloader:
                # first predict the user intent
                # the generate 
                inputs, labels = batch[0], batch[1]
                gen_batch=self.generate_batch(self.model, inputs, max_len, eos_id)
                gens+=self.convert_batch_ids_to_tokens(self.tokenizer, gen_batch, eos_id)
                oracles+=self.convert_batch_ids_to_tokens(self.tokenizer, labels, eos_id)
                contexts+=self.convert_batch_ids_to_tokens(self.tokenizer, inputs, sep_id)
        logging.info('Generation time:{:.2f} min'.format((time.time()-st)/60))
        (P, R, F1), bleu = eval_context_to_response(gens, oracles)
        logging.info('Intent P/R/F1:{:.3f},{:.3f},{:.3f}, BLEU of response:{:.2f}'.format(P, R, F1, bleu))
        results=integrate_result(contexts, gens, oracles)
        json.dump(results, open(os.path.join(cfg.gpt_path, 'result.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    

    def generate_batch(self, model, contexts, max_len, eos_id, beam=1):
        # generate by batch
        # contexts: a list of ids
        # max_len: the max generated length
        # eos_id: the end id
        # return: a batch of ids with pre pad 
        batch_size=len(contexts)
        end_flag=np.zeros(batch_size)
        if beam>1:
            beam_box=[beam]*batch_size
            beam_result=[[] for _ in range(batch_size)]
            max_prob=[-float('inf')]*batch_size
        past_key_values=None
        inputs,attentions=batch_align(contexts,left_len=max_len,return_attn=True)
        inputs=torch.tensor(inputs).to(model.device)
        attentions=torch.tensor(attentions).to(model.device)
        model.eval()
        with torch.no_grad():
            for i in range(max_len):
                if beam==1:
                    position_ids = attentions.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attentions == 0, 1)
                    if past_key_values is not None:
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                    if inputs.size(0)==0:
                        raise ValueError(contexts, inputs.cpu().list(), attentions)
                    outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                            return_dict=True,use_cache=True,past_key_values=past_key_values)

                    past_key_values=outputs.past_key_values

                    preds=outputs.logits[:,-1,:].argmax(-1)#B
                    if i==0:
                        gen_tensor=preds.unsqueeze(1)
                    else:
                        gen_tensor=torch.cat([gen_tensor,preds.unsqueeze(1)],dim=1)
                    attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                    inputs=preds.unsqueeze(1)
                    end_flag+=(preds.cpu().numpy()==eos_id).astype(float)
                    if sum(end_flag==0)==0:
                        break
                else:
                    if i==0:
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values)
                        past_key_values=[outputs.past_key_values]*beam
                        log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                        beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                        gen_tensor=beam_idx.unsqueeze(-1)# B, beam, 1
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx#B, beam
                    else:
                        for j in range(beam):
                            inputs=pv_beam_idx[:,j].unsqueeze(-1) # B, 1
                            outputs=model(inputs,attention_mask=attentions,position_ids=position_ids,\
                                return_dict=True,use_cache=True,past_key_values=past_key_values[j])
                            past_key_values[j]=outputs.past_key_values
                            log_prob=F.log_softmax(outputs.logits[:, -1, :], -1) # B, V
                            beam_prob, beam_idx=torch.topk(log_prob, beam, -1) # B, beam
                            if j==0:
                                prob_pool= beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam) # B, beam
                                id_pool=beam_idx
                            else:
                                prob_pool=torch.cat([prob_pool, beam_prob+pv_beam_prob[:, j].unsqueeze(-1).expand(-1, beam)],-1) # B, beam*beam
                                id_pool=torch.cat([id_pool, beam_idx], -1)# B, beam*beam
                        beam_prob, temp_id=torch.topk(prob_pool, beam, -1) #B, beam
                        beam_idx=torch.gather(id_pool, -1, temp_id)
                        temp_id=temp_id//beam
                        new_past_key_values=copy.deepcopy(past_key_values)
                        for b in range(batch_size):
                            gen_tensor[b, :, :]=gen_tensor[b, :, :].index_select(0, temp_id[b, :])
                            for t in range(beam):
                                for l in range(6):
                                    new_past_key_values[t][l][:, b, :,:,:]=past_key_values[temp_id[b, t]][l][:, b, :, :, :]
                        past_key_values=new_past_key_values
                        #past_key_values=[past_key_values[t] for t in temp_id.cpu().list()]
                        gen_tensor=torch.cat([gen_tensor, beam_idx.unsqueeze(-1)],-1) #B, beam, T
                        attentions=torch.cat((attentions,torch.ones(batch_size,1).long().to(model.device)),dim=1)
                        position_ids = attentions.long().cumsum(-1) - 1
                        position_ids.masked_fill_(attentions == 0, 1)
                        position_ids=position_ids[:, -1].unsqueeze(-1)
                        pv_beam_prob=beam_prob #B, beam
                        pv_beam_idx=beam_idx
                    for m in range(batch_size):
                        for n, gen in enumerate(gen_tensor.cpu().tolist()[m]):
                            if eos_id in gen:
                                beam_box[m]-=1
                                avg_prob=pv_beam_prob[m][n]/len(gen)
                                beam_result[m].append((gen, avg_prob))
                                pv_beam_prob[m][n]=-float('inf')
                    # we do not break during beam search
                    #if not any(beam_box):
                     #   break
        if beam==1:
            return gen_tensor.cpu().tolist()
        else:
            for i, tup in enumerate(beam_result):
                beam_list=sorted(tup, key=lambda item:item[1], reverse=True)
                beam_result[i]=[item[0] for item in beam_list[:beam]]
            return beam_result     

    def convert_batch_ids_to_tokens(self, tokenizer, input_ids, eos_id, return_ids=False):
        # input_ids: B*T
        # output: B*string
        outputs=[]
        outputs_ids=[]
        for sent_ids in input_ids:
            if eos_id in sent_ids:
                sent_ids=sent_ids[:sent_ids.index(eos_id)+1]
            else:
                sent_ids[-1]=eos_id
            outputs_ids.append(sent_ids)
            outputs.append(tokenizer.decode(sent_ids))
        if return_ids:
            return outputs, outputs_ids
        return outputs


def parse_arg_cfg(args):
    # add args to cfg
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k == 'cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')
    if not os.path.exists('./log'):
        os.mkdir('./log')

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()
    cfg.mode = args.mode
    parse_arg_cfg(args)
    parse_arg_cfg(args)
    if cfg.exp_path=='':
        experiments_path = './experiments'
        cfg.exp_path = os.path.join(experiments_path, cfg.exp_name)
        if not os.path.exists(cfg.exp_path):
            os.mkdir(cfg.exp_path)

    cfg._init_logging_handler()

    # fix random seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    # initialize model
    m = Model(cfg.device)
    # train
    if cfg.mode=='train':
        m.train()
    else:
        m.test_end_to_end()

if __name__ == "__main__":
    main()
