from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
from helpers.timestamp import name_timestamp
import pandas as pd
import scipy


class staticSim:
    def __init__(self, lang='en-nl',
                 en_model='./data/en_mapped.vec',
                 nl_model='./data/nl_mapped.vec') -> None:
        self.lang = lang
        self.model_loader(en_model, nl_model)

    def model_loader(self, en_model, nl_model):
        # Load the English model
        # self.en_model = KeyedVectors.load_word2vec_format(en_model)
        self.en_model = load_facebook_vectors(en_model)

        # Load the Dutch model
        # self.nl_model = KeyedVectors.load_word2vec_format(nl_model)
        self.nl_model = load_facebook_vectors(nl_model)

    def get_sl_vector(self, word):
        vector = self.en_model.get_vector(word)
        return vector

    def get_tl_vector(self, word):
        vector_nl = self.nl_model.get_vector(word)
        return vector_nl

    def word_emb_cos(self, src_w, tgt_w):
        src_w_in_emb = ''
        tgt_w_in_emb = ''
        if src_w in self.en_model and tgt_w in self.nl_model:
            src_w_in_emb = 'yes'
            tgt_w_in_emb = 'yes'
            w_sl_vector = self.get_sl_vector(src_w)
            w_tl_vector = self.get_tl_vector(tgt_w)
            distance = (scipy.spatial.distance.cosine(w_sl_vector, w_tl_vector))

        if src_w not in self.en_model:
            src_w_in_emb = 'no'
            distance = (float('nan'))
        if tgt_w not in self.nl_model:
            tgt_w_in_emb = 'no'
            distance = (float('nan'))

        return distance, src_w_in_emb, tgt_w_in_emb


class shift_finder:
    def __init__(self, src_tgt_d_df: pd.DataFrame) -> None:
        self.df = src_tgt_d_df

    def prep_csv(self, filename='shift_finder'):
        ''''''
        filetime = name_timestamp()
        self.df.to_csv(f'{filetime}-{filename}')


    def find_csv_threshold(self):
        ''''''

    def find_shifts(self):
        ''''''
