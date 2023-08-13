from gensim.models import KeyedVectors
import scipy


class staticSim:
    '''A class used to load embeddings into Gensim for inference.'''
    def __init__(self, lang='en-nl',
                 en_model='../../data/_embeddings/fasttext/cc.en.300.mapped.vec',
                 nl_model='../../data/_embeddings/fasttext/cc.nl.300.mapped.vec') -> None:
        '''Initializes a staticSim object based on a given lang attribute
        (input-output) and loads the models for faster inference.
        '''
        self.lang = lang
        self.model_loader(en_model, nl_model)

    def model_loader(self, en_model, nl_model):
        '''Loads the given English and Dutch models into Gensim for
        inference.
        '''
        self.en_model = KeyedVectors.load_word2vec_format(en_model)
        self.nl_model = KeyedVectors.load_word2vec_format(nl_model)

    def get_sl_vector(self, word):
        '''Returns the source language vector for a given word.'''
        vector = self.en_model.get_vector(word)
        return vector

    def get_tl_vector(self, word):
        '''Returns the target language vector for a given word.'''
        vector_nl = self.nl_model.get_vector(word)
        return vector_nl

    def word_emb_cos(self, src_w, tgt_w):
        '''Returns the cosine distance, source word, and target word
        for a given source word and target word.

        Returns NaN if a word is not found.
        '''
        if src_w in self.en_model and tgt_w in self.nl_model:
            w_sl_vector = self.get_sl_vector(src_w)
            w_tl_vector = self.get_tl_vector(tgt_w)
            distance = (scipy.spatial.distance.cosine(w_sl_vector, w_tl_vector))

        if src_w not in self.en_model:
            distance = (float('nan'))
        if tgt_w not in self.nl_model:
            distance = (float('nan'))

        return distance, src_w, tgt_w
