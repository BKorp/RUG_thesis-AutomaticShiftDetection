from typing import Any
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import jenkspy
from statistics import mean
from collections import Counter


class contextualSim():
    '''A class for running contextual similarity.'''
    def __init__(self,
                 model_name:str='sentence-transformers/distiluse-base-multilingual-cased-v2',
                 layer_strat='last_hidden',
                 alt=False) -> None:
        self.dev = self.gpu_checker()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.dev).eval()
        self.layer_strat = layer_strat
        self.alt = alt

    def gpu_checker(self, cpu_override=False):
        '''Checking for the GPU availability'''
        if torch.cuda.is_available() and not cpu_override:
            dev = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            dev = torch.device("cpu")
            print("Running on the CPU")

        return dev

    def _mean_pooling(self, model_output, attention_mask):
        '''Mean pooling is used with the model output and attention
        mask to return the overall sentence embeddings.
        '''
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _word_tensor_processing(self, model_output, token_id):
        '''The word embeddings are retrieved by using a layer strategy
        (such as last hidden and second to last hidden layer) after
        which token id is used to return the specific word embedding that
        is required.
        '''
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
        '''A language model is used with pytorch to return a dataframe
        with a sentence cosine distance score and word level cosine
        distance scores for a given source sentence, target sentence
        and word alignments.
        '''
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
                                          return_tensors='pt').to(self.dev)

        src_tok = tok_sys(src_sent)
        tgt_tok = tok_sys(tgt_sent)

        with torch.no_grad():
            src_out = self.model(**src_tok)
            tgt_out = self.model(**tgt_tok)

        src_embs_s = self._mean_pooling(src_out, src_tok['attention_mask'])
        tgt_embs_s = self._mean_pooling(tgt_out, tgt_tok['attention_mask'])
        dist_sent = float(1 - torch.cosine_similarity(src_embs_s.reshape(1,-1), tgt_embs_s.reshape(1,-1)))
        pd_dict['cosine_sent'] = dist_sent

        for src, tgt in aligns:
            pd_dict['src'].append(src_sent.split(' ')[src])
            pd_dict['tgt'].append(tgt_sent.split(' ')[tgt])

            src_tok_id = np.where(np.array(src_tok.word_ids()) == src)
            tgt_tok_id = np.where(np.array(tgt_tok.word_ids()) == tgt)


            src_embs = self._word_tensor_processing(src_out, src_tok['attention_mask'], src_tok_id)
            tgt_embs = self._word_tensor_processing(tgt_out, tgt_tok['attention_mask'], tgt_tok_id)

            dist_w = round(float(1 - torch.cosine_similarity(src_embs.reshape(1,-1), tgt_embs.reshape(1,-1))), 17)

            src_tgt = '{} <-> {}'.format(pd_dict['src'][-1], pd_dict['tgt'][-1])

            pd_dict['cosine_w'].append(dist_w)

        return pd.DataFrame.from_dict(pd_dict)

    def find_label(self, num):
        '''Returns a label based on a given score number
        and a threshold that has been set in the object.
        '''
        if num > self.threshold:
            return 'creative shift'
        else:
            return 'reproduction'

    def find_groups_num(self, grp, n_classes=2):
        breaks = jenkspy.jenks_breaks(grp, n_classes=n_classes)
        groups = [breaks[i:i+2] for i in range(len(breaks))
                  if not i + 2 > len(breaks)]
        return [
            [i for i in grp if mima_i[0] <= i <= mima_i[1]]
            if idx == 0
            else [i for i in grp if mima_i[0] < i <= mima_i[1]]
            for idx, mima_i in enumerate(groups)
        ]

    def find_groups(self, df_sent, threshold, sent_sim=False, simple_thres=False):
        '''Return a dataframe for a given sentence and threshold with
        multiple options.

        Labels the words in a given sentence based on a cosine score.
        If simple thres is true, uses a simple threshold workflow to
        see if a score is above or below the threshold. Otherwise,
        uses a complex jenksbreaks setup to split the data into a
        majority and minority group.

        Additionally, uses the sentence distance if sentence sim is true,
        otherwise uses the word level cosine distance for labeling
        the majority group.
        '''
        self.threshold = threshold
        grp = df_sent.cosine_w.to_list()
        unique_scores = set(grp)

        if simple_thres:
            df_sent['labels'] = [self.find_label(i.cosine_w) for idx, i in df_sent.iterrows()]

        else:
            if len(unique_scores) == 1:
                df_sent['labels'] = self.find_label(list(unique_scores)[0])
            else:
                try:
                    breaks = jenkspy.jenks_breaks(grp, n_classes=2)
                except Exception:
                    print(f'JENKSPY_ERROR: {grp}')
                groups = [breaks[i:i+2] for i in range(len(breaks)) if not i + 2 > len(breaks)]

                lst = []
                for idx, i in df_sent.iterrows():
                    for j_idx, mima_i in enumerate(groups):
                        if j_idx == 0 and mima_i[0] <= i.cosine_w <= mima_i[1]:
                            lst.append('lower')
                        elif mima_i[0] < i.cosine_w <= mima_i[1]:
                            lst.append('higher')
                df_sent['groups'] = lst

                major_group = Counter(lst).most_common(1)[0][0]

                if sent_sim:
                    major_label = self.find_label(df_sent.cosine_sent.unique()[0])
                else:
                    major_label = self.find_label(mean(df_sent[df_sent.groups == major_group].cosine_w.to_list()))

                if major_label == 'creative shift':
                    if major_group == 'higher':
                        minor_label = 'reproduction'
                    else:
                        minor_label = 'creative shift'
                else:
                    if major_group == 'higher':
                        minor_label = 'reproduction'
                    else:
                        minor_label = 'creative shift'

                group_labels = []
                for idx, i in df_sent.iterrows():
                    if i.groups == major_group:
                        group_labels.append(major_label)
                    else:
                        group_labels.append(minor_label)
                df_sent['labels'] = group_labels

                df_sent.drop(['groups'], axis=1, inplace=True)

        return df_sent


class SimRunner():
    '''A class used to run the cosine similarity system.'''
    def __init__(self, embedder) -> None:
        '''Initializes a runner object for a given embedder.'''
        self.embedder = embedder

    def __call__(self, text_list, aw_list, film) -> Any:
        '''Starts the iterator of the object to generate dataframes
        with labels for sentences given in a text list with a list
        of word alignments. Additionally stores the name of the given
        film.
        '''
        self.film = film
        return self.iterator(text_list, aw_list)

    def file_loader(self, text_path, aw_path):
        '''Returns a list of sentences and a list of word alignments
        for a given text file path and word alignment path.
        '''
        with open(aw_path, 'r', encoding='utf-8') as f:
            aw_list = f.read().splitlines()

        with open(text_path, 'r', encoding='utf-8') as f:
            text_list = f.read().splitlines()

        aw_list = [aw for aw in aw_list if aw != '']
        text_list = [[sent for sent in sent_pair.split(' ||| ')]
                     for sent_pair in text_list
                     if not '' in sent_pair.split(' ||| ')]

        return text_list, aw_list

    def iterator(self, text_list, aw_list, v1=False, threshold=0.6, simple_thres=False):
        '''Returns a dataframe for a given list of sentences and word alignments.
        Additionally, three options exist to make changes to the processing.

        v1 runs the old v1 setup, which is left for legacy reasons.
        threshold can be used to set the threshold of the inference
        for label creation.
        simple_thres dictates whether to use a majority minority
        grouping (false) or a simple above or below threshold labeling
        (true).
        '''
        for idx, sent_pair in enumerate(text_list):
            try:
                src_sent = sent_pair[0]
                tgt_sent = sent_pair[1]
                aligns = aw_list[idx]
                if idx == 0:
                    df = self.embedder.context_emb(src_sent, tgt_sent, aligns)
                    df['film'] = self.film
                    df['sent_idx'] = idx
                    if v1:
                        df['class'] = self.embedder.find_shift(df.cosine_w.to_list())
                    else:
                        df = self.embedder.find_groups(df, threshold, simple_thres=simple_thres)
                else:
                    df_c = self.embedder.context_emb(src_sent, tgt_sent, aligns)
                    # df['film'] = self.film
                    df_c['film'] = self.film
                    df_c['sent_idx'] = idx
                    if v1:
                        df_c['class'] = self.embedder.find_shift(df_c.cosine_w.to_list())
                    else:
                        df_c = self.embedder.find_groups(df_c, threshold, simple_thres=simple_thres)
                    df = pd.concat([df, df_c], ignore_index=True)
            except Exception:
                print(self.film, '\n', idx, sent_pair)
                continue

        if v1:
            for i in df.sent_idx.unique():
                df_c = self.embedder.find_groups(df[df.sent_idx == i])
                df = pd.concat([df, df_c], ignore_index=True)
        return df