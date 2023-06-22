from astred import AlignedSentences, Sentence
import pandas as pd


class astredRunner:
    def __init__(self, en, nl, aligns) -> None:
        self.sent_en = Sentence.from_text(en, 'en')
        self.sent_nl = Sentence.from_text(nl, 'nl')
        self.aligns = aligns

        self.aligned = AlignedSentences(self.sent_en,
                                        self.sent_nl,
                                        word_aligns=self.aligns)


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
