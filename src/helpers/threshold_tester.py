import pandas as pd
import jenkspy
from statistics import mean
from collections import Counter
from helpers.timestamp import name_timestamp
from sklearn.metrics import classification_report as report


class ThresholdFinder:
    def __init__(self, thresholds=[.4, .45, .5, .55, .6],
                 gold_data='../../data/new_gold_v2_annotated.tsv', gold_sep='\t',
                 basic_thres=True,
                 sent_sim=True,
                 human_label='human_label') -> None:
        self.thresholds = thresholds
        self.gold = pd.read_csv(gold_data, sep=gold_sep)
        self.run_basic_thres = basic_thres
        self.run_sent_sim = sent_sim
        self.human_label = human_label

    def __call__(self, df:pd.DataFrame):
        self.thres_dfs = {}
        df['file'] = df.film
        df['genre'] = df.film.apply(lambda x : x.split('/')[4])
        df['film'] = df.film.apply(lambda x : x.split('/')[5].split('_')[0])
        self.orig_df = df

        for t in self.thresholds:
            self.cur_thres = t

            for idx, row in self.gold.iterrows():
                if idx == 0:
                    out_df = self.find_groups(df[(df.film == row.film) & (df.sent_idx == row.sent_idx)].copy())
                    out_df['thres_metric'] = 'word-major-minor'
                    self.thres_dfs[t] = out_df
                else:
                    out_df = self.find_groups(df[(df.film == row.film) & (df.sent_idx == row.sent_idx)].copy())
                    out_df['thres_metric'] = 'word-major-minor'
                    self.thres_dfs[t] = pd.concat([self.thres_dfs[t], out_df])

                if self.run_sent_sim:
                    out_df = self.find_groups(df[(df.film == row.film) & (df.sent_idx == row.sent_idx)].copy(),
                                              sent_sim=self.run_sent_sim)
                    out_df['thres_metric'] = 'sent-major-minor'
                    self.thres_dfs[t] = pd.concat([self.thres_dfs[t], out_df])

                if self.run_basic_thres:
                    out_df = self.find_groups(df[(df.film == row.film) & (df.sent_idx == row.sent_idx)].copy(),
                                              basic_thres=self.run_basic_thres)
                    out_df['thres_metric'] = 'basic'
                    self.thres_dfs[t] = pd.concat([self.thres_dfs[t], out_df])

    def score(self):
        self.thres_reports = {}
        for threshold, df in self.thres_dfs.items():
            t_metric_reports = {}
            for t_metric in df['thres_metric'].unique():
                pred = []
                gold = []
                for idx, row in self.gold.iterrows():
                    try:
                        cur_gold = row[self.human_label].lower()
                        gold.append(cur_gold)

                        cur_pred = df[
                            (df.sent_idx == row.sent_idx)
                            & (df.src == row.src)
                            & (df['thres_metric'] == t_metric)
                        ]['labels'].to_list()[0]
                        pred.append(cur_pred)
                    except Exception:
                        print('PROBLEM FOUND:\n')
                        print(idx, row)
                        print(df[(df.sent_idx == idx) & (df.src == row.src)])
                        return

                t_metric_reports[t_metric] = report(gold, pred, output_dict=True)

            self.thres_reports[threshold] = t_metric_reports

        for rep_cnt, (threshold, t_metrics) in enumerate(self.thres_reports.items()):
            for metric_cnt, (t_metric, rep) in enumerate(t_metrics.items()):
                out_df = pd.DataFrame(rep['macro avg'], index=[threshold])
                out_df['t_metric'] = t_metric
                if rep_cnt == 0 and metric_cnt == 0:
                    self.comp_df = out_df
                else:
                    self.comp_df = pd.concat([self.comp_df, out_df])

    def get_best(self):
        return self.comp_df[self.comp_df['f1-score'] == self.comp_df['f1-score'].max()]

    def store(self, overwrite: bool|str=False):
        if not overwrite:
            thres_choice = self.best_thres
        else:
            thres_choice = overwrite

        try:
            self.thres_dfs[thres_choice].to_csv(
                f'{name_timestamp()}-{thres_choice}-context.tsv',
                sep='\t',
            )
        except Exception:
            print(f'The threshold \'{thres_choice}\' does not exist')

    def closer_to(self, num):
        if num > self.cur_thres:
            return 1
        else:
            return 0

    def find_label(self, num):
        close = self.closer_to(num)
        if close == 1:
            return 'creative shift'
        else:
            return 'reproduction'

    def find_groups_num(self, grp, n_classes=2):
        breaks = jenkspy.jenks_breaks(grp, n_classes=n_classes)
        groups = [breaks[i:i+2] for i in range(len(breaks)) if not i + 2 > len(breaks)]
        return [[i for i in grp if mima_i[0] <= i <= mima_i[1]] if idx == 0
               else [i for i in grp if mima_i[0] < i <= mima_i[1]]
               for idx, mima_i in enumerate(groups)]

    def find_groups(self, df_sent, sent_sim=False, basic_thres=False):
        grp = df_sent.cosine_w.to_list()
        unique_scores = set(grp)

        if basic_thres:
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


class StaticThres(ThresholdFinder):
    def __init__(self, gold_data, thresholds=[.4, .45, .5, .55, .6]) -> None:
        self.thresholds = thresholds
        self.gold = gold_data

    def __call__(self, df:pd.DataFrame):
        self.gold = pd.concat([self.gold, df.reset_index()[['static_cosine', 'genre']]], axis=1)
        self.thres_dfs = {}
        self.gold.rename(columns={'label_static': 'human_label'}, inplace=True)

        for t in self.thresholds:
            self.cur_thres = t

            out_df = self.gold.copy()
            out_df['labels'] = [self.static_thres(i.static_cosine) for idx, i in self.gold.iterrows()]
            out_df['thres_metric'] = 'basic'
            self.thres_dfs[t] = out_df

    def static_thres(self, cosine_score):
        if cosine_score > self.cur_thres:
            return 'Creative Shift'
        elif cosine_score < self.cur_thres:
            return 'Reproduction'

    def score(self):
        self.thres_reports = {}
        for threshold, df in self.thres_dfs.items():

            self.thres_reports[threshold] = report(
                y_true=df['human_label'].to_list(),
                y_pred=df['labels'].to_list(),
                output_dict=True
            )

        for rep_cnt, (threshold, rep) in enumerate(self.thres_reports.items()):
            if rep_cnt == 0:
                self.comp_df = pd.DataFrame(rep['macro avg'], index=[threshold])
                self.comp_df['t_metric'] = 'basic'
            else:
                out_df = pd.DataFrame(rep['macro avg'], index=[threshold])
                out_df['t_metric'] = 'basic'
                self.comp_df = pd.concat([self.comp_df, out_df])


class ContextualThres(ThresholdFinder):
    def best_df(self):
        self.cur_thres = self.get_best().index[0]
        thres_metric = self.get_best().iloc[0].t_metric

        if thres_metric == 'word-major-minor':
            return self.find_groups(self.orig_df.copy())

        elif thres_metric == 'sent-major-minor':
            return self.find_groups(self.orig_df.copy(),
                                      sent_sim=True)

        elif thres_metric == 'basic':
            return self.find_groups(self.orig_df.copy(),
                                      basic_thres=True)

    def retrieve_best_gold(self):
        df = self.thres_dfs[self.get_best().index[0]]
        thres_metric = self.get_best().iloc[0].t_metric

        for idx, row in self.gold.iterrows():
            cur_df = df[
                (df.sent_idx == row.sent_idx)
                & (df.src == row.src)
                & (df['thres_metric'] == thres_metric)
            ].head(1).copy()

            cur_df['human_label'] = row.label_context

            if idx == 0:
                best_gold = cur_df

            else:
                best_gold = pd.concat([best_gold, cur_df])

        return best_gold

class SyntaxThres(ThresholdFinder):
    def __init__(self, thresholds=[.4, .45, .5, .55, .6],
                 human_label='human_label') -> None:
        self.thresholds = thresholds
        self.human_label = human_label

    def __call__(self, df:pd.DataFrame):
        self.thres_dfs = {}
        for t in self.thresholds:
            self.cur_thres = t
            cur_df = df.copy()
            for i in range(len(df)):
                cur_df.at[i, 'sacr_cross_label'] = self.find_label(cur_df.iloc[i].sacr_cross_score)
                cur_df.at[i, 'label_label'] = self.find_label(cur_df.iloc[i].label_changes)
                cur_df.at[i, 'astred_label'] = self.find_label(cur_df.iloc[i].astred_score)
                cur_df.at[i, 'thres_metric'] = 'basic'

            self.thres_dfs[t] = cur_df

    def score(self):
        self.thres_reports = {}
        for threshold, df in self.thres_dfs.items():
            t_metric_reports = {}
            gold = df[self.human_label].to_list()
            for t_metric in ['sacr_cross_label', 'label_label', 'astred_label']:
                pred = []
                for idx, row in df.iterrows():
                    pred.append(row[t_metric])

                t_metric_reports[t_metric] = report(gold, pred, output_dict=True)

            self.thres_reports[threshold] = t_metric_reports

        for rep_cnt, (threshold, t_metrics) in enumerate(self.thres_reports.items()):
            for metric_cnt, (t_metric, rep) in enumerate(t_metrics.items()):
                out_df = pd.DataFrame(rep['macro avg'], index=[threshold])
                out_df['t_metric'] = t_metric
                if rep_cnt == 0 and metric_cnt == 0:
                    self.comp_df = out_df
                else:
                    self.comp_df = pd.concat([self.comp_df, out_df])

    def get_best(self):
        return self.comp_df[self.comp_df['f1-score'] == self.comp_df['f1-score'].max()]

    def store(self, overwrite: bool|str=False):
        if not overwrite:
            thres_choice = self.best_thres
        else:
            thres_choice = overwrite

        try:
            self.thres_dfs[thres_choice].to_csv(
                f'{name_timestamp()}-{thres_choice}-context.tsv',
                sep='\t',
            )
        except Exception:
            print(f'The threshold \'{thres_choice}\' does not exist')

    def closer_to(self, num):
        if num > self.cur_thres:
            return 1
        else:
            return 0

    def find_label(self, num):
        close = self.closer_to(num)
        if close == 1:
            return 'creative shift'
        else:
            return 'reproduction'

    def best_df(self):
        return self.thres_dfs[self.get_best().index[0]]
