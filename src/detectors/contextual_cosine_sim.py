from typing import Any
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import jenkspy
from statistics import mean


class contextualSim():
    def __init__(self,
                 model_name:str='sentence-transformers/distiluse-base-multilingual-cased-v2',
                 layer_strat='last_hidden',
                 alt=False) -> None:
        # model_name = 'bert-base-multilingual-cased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval()
        self.layer_strat = layer_strat
        self.alt = alt

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _word_tensor_processing(self, model_output, attention_mask, token_id):
        if self.layer_strat == 'last_hidden':
            token_embeddings = model_output[0]
        elif self.layer_strat == 'second_to_last':
            hidden_states = model_output[1]
            token_embeddings = torch.stack(hidden_states[2:]).sum(0)
        elif self.layer_strat == 'conc_last4':
            hidden_states = model_output[1]
            token_embeddings = torch.cat([hidden_states[i]
                                          for i in [-1, -2, -3, -4]],
                                          dim=-1)

        token_embeddings = token_embeddings.squeeze()[token_id[0][0]]
        return token_embeddings

    def context_emb(self, src_sent:str, tgt_sent:str, aligns:str):
        pd_dict = {
            'src_sent': src_sent,
            'tgt_sent': tgt_sent,
            'aligns': aligns,
            'src': [],
            'tgt': [],
            'cosine_w': [],
            'cosine_sent': 0,
        }

        aligns = [[int(i) for i in pair.split('-')] for pair in aligns.split(' ')]
        tok_sys = lambda sent : self.tokenizer(sent,
                                          padding=True,
                                          truncation=True,
                                          max_length=128,
                                          return_tensors='pt')

        src_tok = tok_sys(src_sent)
        tgt_tok = tok_sys(tgt_sent)

        with torch.no_grad():
            src_out = self.model(**src_tok)
            tgt_out = self.model(**tgt_tok)

        src_embs_s = self._mean_pooling(src_out, src_tok['attention_mask'])
        tgt_embs_s = self._mean_pooling(tgt_out, tgt_tok['attention_mask'])
        dist_sent = float(1 - torch.cosine_similarity(src_embs_s.reshape(1,-1), tgt_embs_s.reshape(1,-1)))
        pd_dict['cosine_sent'] = dist_sent

        # print('src <-> tgt', '\t', 'cosine distance', '\n')
        for src, tgt in aligns:
            pd_dict['src'].append(src_sent.split(' ')[src])
            pd_dict['tgt'].append(tgt_sent.split(' ')[tgt])

            src_tok_id = np.where(np.array(src_tok.word_ids()) == src)
            tgt_tok_id = np.where(np.array(tgt_tok.word_ids()) == tgt)


            # src_embs = mean_pooling(src_out, src_tok['attention_mask'])
            # break
            src_embs = self._word_tensor_processing(src_out, src_tok['attention_mask'], src_tok_id)
            tgt_embs = self._word_tensor_processing(tgt_out, tgt_tok['attention_mask'], tgt_tok_id)
            # src_states = src_out.hidden_states[-1].squeeze()
            # tgt_states = tgt_out.hidden_states[-1].squeeze()

            # src_embs = src_states[src_tok_id[0][0]]
            # tgt_embs = tgt_states[src_tok_id[0][0]]

            dist_w = round(float(1 - torch.cosine_similarity(src_embs.reshape(1,-1), tgt_embs.reshape(1,-1))), 17)

            src_tgt = '{} <-> {}'.format(pd_dict['src'][-1], pd_dict['tgt'][-1])
            # print(src_tgt, '\t\t', dist_w)

            pd_dict['cosine_w'].append(dist_w)

        # print(dist_sent)
        return pd.DataFrame.from_dict(pd_dict)

    def find_shift(self, w_cosine_list, n_classes=2, thres=0.2):
        if self.alt:
           return ['repro' if score >= 0.485 else 'shift' for score in w_cosine_list]
        else:
            if abs(min(w_cosine_list) - max(w_cosine_list)) >= thres:
                breaks = jenkspy.jenks_breaks(w_cosine_list, n_classes=n_classes)

                groups = {1: [], 2: []}
                for score in w_cosine_list:
                    if breaks[0] <= score <= breaks[1]:
                        groups[1].append(score)
                    else:
                        groups[2].append(score)

                if len(groups[1]) == len(groups[2]):
                    if mean(groups[1]) > mean(groups[2]):
                        group_1 = 'shift'
                        group_2 = 'repro'
                    else:
                        group_1 = 'repro'
                        group_2 = 'shift'

                    return [group_1 if score in groups[1] else group_2 for score in w_cosine_list]

                else:
                    group_min = groups[min(groups.items(), key=lambda x : len(x[-1]))[0]]
                    group_max = groups[max(groups.items(), key=lambda x : len(x[-1]))[0]]
                    if 1 - mean(group_min) > mean(group_min):
                        group_min_class = 'repro'
                    else:
                        group_min_class = 'shift'

                    if 1 - mean(group_max) > mean(group_max):
                        group_max_class = 'repro'
                    else:
                        group_max_class = 'shift'

                # return [group_max_class if breaks[0] <= score <= breaks[1] else group_min_class for score in w_cosine_list]
                return [group_max_class if score in group_max else group_min_class for score in w_cosine_list]

            else:
                return ['repro' if 1 - mean(w_cosine_list) > mean(w_cosine_list) else 'shift' for score in w_cosine_list]


class SimRunner():
    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def __call__(self, text_list, aw_list) -> Any:
        return self.iterator(text_list, aw_list)

    def file_loader(self, text_path, aw_path):
        with open(aw_path, 'r', encoding='utf-8') as f:
            aw_list = f.read().splitlines()

        with open(text_path, 'r', encoding='utf-8') as f:
            text_list = f.read().splitlines()

        aw_list = [aw for aw in aw_list if aw != '']
        text_list = [[sent for sent in sent_pair.split(' ||| ')]
                     for sent_pair in text_list
                     if not '' in sent_pair.split(' ||| ')]

        return text_list, aw_list

    def iterator(self, text_list, aw_list):
        for idx, sent_pair in enumerate(text_list):
            src_sent = sent_pair[0]
            tgt_sent = sent_pair[1]
            aligns = aw_list[idx]
            if idx == 0:
                df = self.embedder.context_emb(src_sent, tgt_sent, aligns)
                df['class'] = self.embedder.find_shift(df.cosine_w.to_list())
            else:
                df_c = self.embedder.context_emb(src_sent, tgt_sent, aligns)
                df_c['class'] = self.embedder.find_shift(df_c.cosine_w.to_list())
                df = pd.concat([df, df_c], ignore_index=True)

        return df