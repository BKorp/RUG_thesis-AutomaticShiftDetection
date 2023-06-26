from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
import re


@dataclass
class baseFunctionsData:
    def read_writer(self, fname, mode='r', data=None):
        with open(fname, mode, encoding='utf-8') as f:
            if mode == 'r':
                inp = f.read().splitlines()
            elif mode == 'w':
                f.write('\n'.join(data))

        if mode == 'r':
            return inp

    def sort_globber(self, path, pattern):
        return sorted(path.glob(pattern))[0]

    def wa_converter(self, wa, sa, fname):
        new_wa = [
            sorted([list(map(int, y.split('-'))) for y in x])
            for x in [i.split(' ') for i in wa]
        ]

        wa_conv_sents = []
        wa_conv_words = []
        for sent_idx, wa_sents in enumerate(new_wa):
            st, tt = sa[sent_idx].split(' ||| ')
            for wa_words in wa_sents:
                wa_st_idx, wa_tt_idx = wa_words
                try:
                    wa_conv_words.append([
                        st.split(' ')[wa_st_idx],
                        tt.split(' ')[wa_tt_idx]
                        ])
                except Exception:
                    print(fname, sent_idx, wa_words, st, tt)
            wa_conv_sents.append(wa_conv_words)
            wa_conv_words = []

        return wa_conv_sents


@dataclass
class origData(baseFunctionsData):
    ''''''
    path: Path
    name: str

    en_nl: list = field(init=False)
    en_mt: list = field(init=False)
    wa_en_nl: list = field(init=False)
    wa_en_mt: list = field(init=False)

    en: list = field(init=False)
    nl: list = field(init=False)
    mt: list = field(init=False)

    w_pair_en_nl: list = field(init=False)
    w_pair_en_mt: list = field(init=False)

    def __post_init__(self):
        ''''''
        self.en_nl = self.read_writer(self.sort_globber(self.path, f'txt/*_en_nl.txt'))
        self.en_mt = self.read_writer(self.sort_globber(self.path, f'txt/*_ht_mt.txt'))
        self.wa_en_nl = self.read_writer(self.sort_globber(self.path, f'wa/*_en_nl.txt'))
        self.wa_en_mt = self.read_writer(self.sort_globber(self.path, f'wa/*_ht_mt.txt'))

        self.en = [i.split(' ||| ')[0] for i in self.en_nl]
        self.nl = [i.split(' ||| ')[-1] for i in self.en_nl]
        self.mt = [i.split(' ||| ')[-1] for i in self.en_mt]

        self.w_pair_en_nl = self.wa_converter(self.wa_en_nl, self.en_nl, self.name)
        self.w_pair_en_mt = self.wa_converter(self.wa_en_mt, self.en_mt, self.name)


@dataclass
class newrunData(baseFunctionsData):
    ''''''
    path: Path
    name: str

    en_nl: list = field(init=False)
    wa_en_nl: list = field(init=False)

    en: list = field(init=False)
    nl: list = field(init=False)

    w_pair_en_nl: list = field(init=False)

    def __post_init__(self):
        ''''''
        self.en_nl = self.read_writer(self.sort_globber(self.path, f'txt/*_en-nl.lfa'))
        self.wa_en_nl = self.read_writer(self.sort_globber(self.path, f'wa/*_en-nl.wa'))

        self.en_nl = [i for i in self.en_nl if '' not in re.sub(' +', ' ', i).split(' ||| ')]
        self.wa_en_nl = [i for i in self.wa_en_nl if i != '']

        self.en = [i.split(' ||| ')[0] for i in self.en_nl]
        self.nl = [i.split(' ||| ')[-1] for i in self.en_nl]

        self.w_pair_en_nl = self.wa_converter(self.wa_en_nl, self.en_nl, self.name)


class dataPrepper:
    def setup(self, path_orig=Path('../data/0_original/'), path_newrun=('../data/1_rerun/')):
        self.path_orig = path_orig
        self.path_newrun = path_newrun

        return self.prep('orig'), self.prep('newrun')

    def help_info(self):
        return (
            'These object structures contain all the data that is used in the thesis. '
            '\n\nIt contains the following data structure: '
            '\ndata root(root object name, for example, orig).'
            'movie_id(e.g. ac01).data_type(e.g. en). '
            '\n\nThe datatypes that can be found are: '
            '\n\nen_nl: Aligned sentences en to nl'
            '\nen_mt: Aligned sentences en to machine translation nl'
            '\n\nwa_en_nl: Word alignments for en to nl'
            '\nwa_en_mt: Word alignments for en to mt'
            '\n\nen: The English (en) sentences'
            '\nnl: The Dutch (nl) sentences'
            '\nmt: The Machine translation Dutch (mt) sentences'
            '\n\nw_pair_en_nl: The word pairs for en to nl based on the word alignment'
            '\nw_pair_en_mt: The word pairs for en to mt based on the word alignment'
        )

    def prep(self, variant):
        if variant == 'orig':
            cur_path = self.path_orig
        elif variant == 'newrun':
            cur_path = self.path_newrun

        file_ids = {'help': self.help_info()}
        for p in sorted(cur_path.glob('*/*')):
            if p.is_dir():
                file_name = p.as_posix().split('/')[-1]
                idx_code = file_name.split('_')[0].lower()
                if variant == 'orig':
                    file_ids[idx_code] = origData(p, file_name)
                elif variant == 'newrun':
                    file_ids[idx_code] = newrunData(p, file_name)

        return SimpleNamespace(**file_ids)