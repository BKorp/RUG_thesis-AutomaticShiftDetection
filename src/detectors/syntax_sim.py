from astred import AlignedSentences, Sentence
from statistics import mean
import pandas as pd


class astredRunner:
    def __init__(self, en, nl, aligns) -> None:
        self.sent_en = Sentence.from_text(en, 'en')
        self.sent_nl = Sentence.from_text(nl, 'nl')
        self.aligns = aligns

        self.aligned = AlignedSentences(self.sent_en,
                                        self.sent_nl,
                                        word_aligns=self.aligns)

    def sacr_cross_score(self, round_bool=False, round_num=2):
        '''The final SACr value is
        the number of crossing alignment links between
        the source and target SACr groups, normalised by
        the number of these alignments. ~Vanroy et al.
        '''
        astred_obj = self.aligned
        score = len(astred_obj.no_null_word_pairs) / astred_obj.src.sacr_cross
        if round_bool:
            return round(score, round_num)
        else:
            return score

    def label_changes_score(self, round_bool=False, round_num=2, verbose=False):
        '''We look at each source word and compare its label
        to the labels of the words that it is aligned to.
        These label changes are then normalised by
        the total number of alignments ~Vanroy et al.
        '''
        astred_obj = self.aligned

        change_list = []
        for src, tgt in astred_obj.no_null_word_pairs:
            if src.deprel == tgt.deprel:
                change = 0
            else:
                change = 1
            change_list.append(change)
            if verbose:
                print(f'\'{src.text}\'({src.deprel}) | \'{tgt.text}\'({tgt.deprel}) | {change}')

        score = sum(change_list) / len(astred_obj.no_null_word_pairs)

        if round_bool:
            score = round(score, round_num)

        if verbose:
            print(f'Total: {sum(change_list)} (normalised: {sum(change_list)} out of {len(astred_obj.no_null_word_pairs)} = {score})')

        return score

    def astred_score(self, round_bool=False, round_num=2, verbose=False):
        '''Use dependency trees with UD labels on
        grouped source-target tokens to retrieve
        the amount of steps necessary for both
        source and target trees to become the same as the other.
        Normalised by taking the total scores of
        all trees by the average of source and target words.
        '''
        astred_obj = self.aligned
        en_sent = self.sent_en
        nl_sent = self.sent_nl

        src_astred_score = []
        tgt_astred_score = []
        for src, tgt in astred_obj.no_null_word_pairs:
            src_astred_score.append(src.tree.astred_cost)
            tgt_astred_score.append(tgt.tree.astred_cost)
            if verbose:
                print(src.text, src.tree.astred_op, tgt.text, tgt.tree.astred_op)

        score = (
            (sum(src_astred_score) + sum(tgt_astred_score))
             / mean([len(en_sent.no_null_words), len(nl_sent.no_null_words)])
        )

        if round_bool:
            score = round(score, round_num)

        if verbose:
            print('')

        return score


class astredVanroy(astredRunner):
    def simple_analysis(self):
        print('\n', self.simple_analysis.__name__, '\n')
        for word in self.sent_nl.no_null_words:
            for aligned_word in word.aligned:
                print(word.text, aligned_word.text, word.deprel, aligned_word.deprel)

    def is_changed(self):
        print('\n', self.is_changed.__name__, '\n')
        verb_is = self.sent_nl[2]
        print("Dutch:", verb_is.text, verb_is.upos)
        for aligned_id, change in self.sent_nl[2].changes("upos").items():
            print("Aligned:", self.sent_en[aligned_id].text, self.sent_en[aligned_id].upos, change)

    def span_root(self):
        print('\n', self.span_root.__name__, '\n')
        for span in self.sent_en.no_null_sacr_spans:
            print(span.text, span.root.text)

    def data_frame(self):
        print('\n', self.data_frame.__name__, '\n')
        df_src = pd.DataFrame.from_dict({w.text: [w.deprel, w.cross, w.sacr_group.cross, w.num_changes(), w.tree.astred_op]
                                    for w in self.aligned.src.no_null_words},
                orient="index",
                columns=["deprel", "cross", "sacr_cross", "dep_changes", "astred_op"])
        df_tgt = pd.DataFrame.from_dict({w.text: [w.deprel, w.cross, w.sacr_group.cross, w.num_changes(), w.tree.astred_op]
                                    for w in self.aligned.tgt.no_null_words},
                orient="index",
                columns=["deprel", "cross", "sacr_cross", "dep_changes", "astred_op"])
        return df_src, df_tgt


class astredAndre(astredRunner):
    def __init__(self, en, nl, aligns, name) -> None:
        super().__init__(en, nl, aligns)
        self.name = name

    def data_frame(self):
        return pd.DataFrame.from_dict({src.text: [self.name, tgt.text, src.deprel, src.cross, src.sacr_group.cross, src.num_changes(), src.tree.astred_op]
                                       for src, tgt in self.aligned.no_null_word_pairs},
                orient="index",
                columns=["sent_name", "aligned_tgt", "deprel", "cross", "sacr_cross", "dep_changes", "astred_op"])

    def testing(self):
        return self.aligned.sacr_cross