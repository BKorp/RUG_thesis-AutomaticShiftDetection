from pysubparser.parsers.srt import parse_timestamps
from pysubparser.classes.subtitle import Subtitle
from itertools import count
from pysubparser.cleaners import brackets, formatting
import re
import unicodedata
from sacremoses import MosesTokenizer
import html
import stanza


class subtitler:
    '''A class for subtitle (srt) conversion, cleanup, and
    tokenization
    '''
    def __init__(self, tokenizer) -> None:
        '''Initialize the object while loading the requested
        tokenizer (Moses or Stanza).
        '''
        self.tokenizer = tokenizer
        if self.tokenizer == 'moses':
            self.tokenizer_en = MosesTokenizer('en')
            self.tokenizer_nl = MosesTokenizer('nl')
        else:
            self.tokenizer_en = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, verbose=False)
            self.tokenizer_nl = stanza.Pipeline(lang='nl', processors='tokenize', tokenize_no_ssplit=True, verbose=False)

    def subtitle_prep(self, data):
        '''Return a basic conversion of srt to txt without
        full cleanup.
        '''
        subtitles = self.subtitle_transformer(data)
        for cleaner in [formatting, brackets]:
            subtitles = cleaner.clean(subtitles)
        subtitles = [subtitle.text for subtitle in subtitles]
        return subtitles

    def subtitle_transformer(self, data):
        '''Return a list of text that is converted from SRT format.'''
        timestamp_separator = " --> "
        index = count(0)
        subtitle = None

        sub_list = []
        for line in data.split('\n'):
            line = line.rstrip()

            if not subtitle:
                if timestamp_separator in line:
                    start, end = parse_timestamps(line)

                    subtitle = Subtitle(next(index), start, end)
            else:
                if line:
                    subtitle.add_line(line)
                else:
                    sub_list.append(subtitle)
                    subtitle = None

        return sub_list

    def regex_steps(self, sent):
        '''Prepare regex and shared regex steps'''
        sent = unicodedata.normalize('NFKD', sent)

    def tokenize(self, sent, lang, unescape=True):
        '''Return a tokenized sentence for the given sent for the
        given language. If using moses, it is possible to remove
        html (xml) escapes from the output.
        '''
        tokenizer = {'en': self.tokenizer_en, 'nl': self.tokenizer_nl}

        if self.tokenizer == 'moses':
            tokenized = tokenizer[lang].tokenize(sent, return_str=True)

            # This differs from the original code, but I unescape the sequences
            # because we have no use of these escapes in the final output
            if unescape:
                return html.unescape(tokenized)
            else:
                return tokenized

        else:
            tokenized = ' '.join([token.text for token
                                  in tokenizer[lang](sent).sentences[0].tokens])
            return tokenized


class subtitlerRepro(subtitler):
    '''A reproduction class for subtitle (srt) conversion, cleanup, and
    tokenization
    '''
    def regex_steps(self, sent):
        '''Return cleaned sentences using regex (primarily).'''
        super(subtitlerRepro, self).regex_steps(sent)

        punctuation = '<>0123456789'
        sent = sent.lower()
        for marker in punctuation:
            sent = sent.replace(marker, '')

        re_groups = (
            r'({.*?})'
            r'|(♪.*?♪)'
            r'|([\(\[].*?[\)\]])'
        )
        sent = re.sub(re_groups, '', sent)

        return sent.lstrip().rstrip()


class subtitlerNew(subtitler):
    '''A new class for subtitle (srt) conversion, cleanup, and
    tokenization
    '''
    def regex_steps(self, sent):
        '''Return cleaned sentences using regex (primarily).'''
        super(subtitlerNew, self).regex_steps(sent)

        # Remove xml leftovers, music, ellipsis, square bracket information
        re_groups = (
            r'({.*?})'
            r'|(♪+.*♪+)'
            r'|(\.\.\.)|(…)'
            r'|([\(\[].*?[\)\]])'
            # r'|(^[ A-Z]*: *)',  # Remove speaker, this introduces issues with the inverted uppercase in The Bourne Ultimatum
                                  # and sometimes speakers can occur midline.
        )
        sent = re.sub(re_groups, '', sent)

        # Replace dashes with spaces
        re_groups = (
            r'(--+)'
            r'|(^-+)|(-+$)'
            r'|( +-+)'
            r'|(-+ +)'
        )
        sent = re.sub(re_groups, ' ', sent)

        # Remove all excess spaces
        sent = re.sub(r' +', ' ', sent)

        return sent.lstrip().rstrip()
