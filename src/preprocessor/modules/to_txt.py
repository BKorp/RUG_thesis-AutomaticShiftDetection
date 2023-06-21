from pysubparser.parsers.srt import parse_timestamps
from pysubparser.classes.subtitle import Subtitle
from itertools import count
from pysubparser.cleaners import ascii, brackets, formatting
import re
import unicodedata
from pathlib import Path
# from mosestokenizer import MosesTokenizer # from https://pypi.org/project/fast-mosestokenizer/#usage-python
from sacremoses import MosesTokenizer
import html


class subtitler:
    def __init__(self) -> None:
        self.tokenizer_en = MosesTokenizer('en')
        self.tokenizer_nl = MosesTokenizer('nl')

    def subtitle_prep(self, data):
        ''''''
        subtitles = self.subtitle_transformer(data)
        for cleaner in [formatting, brackets, ascii]:
            subtitles = cleaner.clean(subtitles)
        return [subtitle.text for subtitle in subtitles]

    def subtitle_transformer(self, data):
        ''''''
        timestamp_separator = " --> "
        index = count(0)
        subtitle = None

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
                    yield subtitle
                    subtitle = None

        # # subtitles = parser.parse(file_name)
        # subtitles = brackets.clean(formatting.clean(subtitles))
        # return [subtitle.text for subtitle in subtitles]

    def regex_steps(self, sent):
        '''...'''
        sent = unicodedata.normalize('NFKD', sent)

    def tokenize(self, sent, lang, unescape=True):
        ''''''
        tokenizer = {'en': self.tokenizer_en, 'nl': self.tokenizer_nl}
        tokenized = tokenizer[lang].tokenize(sent, return_str=True)

        # This differs from the original code, but I unescape the sequences
        # because we have no use of these escapes in the final output
        if unescape:
            return html.unescape(tokenized)
        else:
            return tokenized


class subtitlerRepro(subtitler):
    def regex_steps(self, sent):
        super(subtitlerRepro, self).regex_steps(sent)

        punctuation = '<>0123456789'
        sent = sent.lower()
        for marker in punctuation:
            sent = sent.replace(marker, '')

        regex_steps = [
            r'{.*?}',
            r'♪.*?♪',
            r'[\(\[].*?[\)\]]',
        ]
        for step in regex_steps:
            sent = re.sub(step, '', sent)

        return sent.lstrip().rstrip()


class subtitlerNew(subtitler):
    def regex_steps(self, sent):
        super(subtitlerNew, self).regex_steps(sent)

        regex_steps = [
            r'<.+>',  # Remove all <...> sequences
            r'{.*?}',  # Remove all occurences of {\an8}
            r'♪.*?♪',
            r'\- *',  # Remove Dashes (-) from the sentences
            r'\[.*\]',  # remove background music and other sounds
                        # Example from assassin's creed:
                        # [Massive Attack's "He Says He Needs Me" playing]
            r'\([ A-Z]*\)',  # Remove background sounds
                             # Example taken from saving private ryan
                             # (GUNFIRE), (BODY FALLS), etc.
            r'^[ A-Z]*: *',  # Remove speaker
                             # Example taken from saving private ryan
                             # MAN: , GENERAL MARSHALL: , etc.
        ]
        for step in regex_steps:
            sent = re.sub(step, '', sent)

        return sent.lstrip().rstrip()
