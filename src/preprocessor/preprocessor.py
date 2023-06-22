# Basic file conversion
    # Convert XML to SRT
    # Convert SRT to txt -> .txt files with lines
    # Pre-cleanup (weird tokens and errors that should be removed before machine translation)

# Alignment file creation
    # (Machine translation occurs here)
    # Convert txt to sentence aligned -> sentence pairs (preferably stored as indices, will come to this shortly)
    # (Further cleanup if needed)
    # (Tokenisation occurs here)
    # Convert sentence aligned to word aligned -> word pairs (preferably stored as indices, will come to this shortly)

### Alignment files are stored as indices so sentence cleanup can be portable
### cleanup of non dialogue (e.g., sounds, emojis, etc.)

## Machine translation
    # Files are sent through DEEPL for translation
    # A script is made to move the mouse and automate the setup directly

## Tokenisation
    # Tokenisation of data (is required to be done before word pair alignment and cleanup; sent aligner has its own tokenisation,
        # but this does not have to be used beyond the sentence alignment)
    # NLTK, Spacy, moses, or STANZA are all potential options

from modules.to_srt import to_srt as external_srt_sys
from modules.to_txt import subtitlerNew, subtitlerRepro
from modules import align_sent
from modules import align_word
import codecs
from pathlib import Path


class readWriter():
    def __init__(self) -> None:
        pass

    def normal(self, fname, readwrite='r', out_data=None, list_out=True):
        ''''''
        with open(fname, readwrite, encoding='utf-8') as f:
            if readwrite == 'r':
                return f.read().splitlines()
            else:
                if list_out:
                    f.write('\n'.join(out_data))
                else:
                    f.write(out_data)

    def codec(self, fname, readwrite='r', out_data=None):
        if readwrite == 'r':
            with codecs.open(fname, 'rb', encoding="utf-8") as f:
                return f.read()
        else:
            with codecs.open(fname, 'wb', encoding="utf-8") as f:
                f.write(out_data)


class fileConverter:
    def __init__(self, data) -> None:
        self.data_orig = data
        self.data = data
        self.init_ext = '.xml'
        self.txt_convert = None

    def to_srt(self):
        ''''''
        self.data_srt = external_srt_sys(self.data, self.init_ext)
        self.data = self.data_srt

    def to_txt(self, lang='en'):
        ''''''
        convert = self.txt_convert
        self.data_txt = [convert.regex_steps(i) for i in convert.subtitle_prep(self.data) if i != '']
        self.data_tok = [convert.tokenize(i, lang) for i in self.data_txt]
        self.data = self.data_tok


class reproc(fileConverter):
    def __init__(self, data, aligner, fname) -> None:
        super().__init__(data)
        self.to_srt()
        self.txt_convert = subtitlerRepro()
        self.to_txt()
        # self.sent_align(aligner, fname)

    def sent_align(self, repro_aligner, fname):
        repro_aligner(fname, self.data)


class new(fileConverter):
    def __init__(self, data) -> None:
        super().__init__(data)
        self.to_srt()
        self.txt_convert = subtitlerNew()
        self.to_txt()


class main:
    def __init__(self) -> None:
        self.main()

    def main(self):
        repro_aligner = align_sent.reprocAligner()
        new_aligner = align_sent.newAligner()

        data_paths = [f for f in sorted(Path('../../data').iterdir()) if f.is_dir()]
        # for i in data_paths[1].glob('*/**/*.xml'):
        #     i_list = i.as_posix().split('/')
        #     cur_dir = i_list[:-2]
        #     new_file = Path('/'.join(cur_dir)) / i_list[-2:][0].replace('xml', 'txt') / i_list[-2:][1].replace('.xml', '.txt')

        #     proc = reproc(readWriter().codec(i), repro_aligner, i.as_posix())
        #     readWriter().normal(new_file, 'w', proc.data)


        # for i in data_paths[1].glob('*/**/*.lfa'):
        #     with open(i, 'r', encoding='utf-8') as f:
        #         data = f.read().splitlines()

        #     data = [i for i in data
        #             if i.split(' ||| ')[0] != ''
        #             and i.split(' ||| ')[1] != '']
        #     try:
        #         wa_data = align_word.reprocAligner().awesome_align(data=data)
        #     except Exception:
        #         print(f'error found for {i}')
        #         continue

        #     with open(
        #         i.as_posix().replace('/txt/', '/wa/').replace('.lfa', '.wa'),
        #         'w',
        #         encoding='utf-8'
        #     ) as f:
        #         f.write('\n'.join(wa_data))

        sent_aligner = align_sent.newAligner()
        word_aligner = align_word.newAligner()
        for i in data_paths[2].glob('*/**/*en.xml'):
            i_list = i.as_posix().split('/')
            cur_dir = i_list[:-2]
            new_file = Path('/'.join(cur_dir)) / i_list[-2:][0].replace('xml', 'txt') / i_list[-2:][1].replace('.xml', '.txt')
            print(i_list, cur_dir, new_file)

            proc_en = new(readWriter().codec(i))
            proc_nl = new(readWriter().codec(i.as_posix().replace('en.xml', 'nl.xml')))
            readWriter().normal(new_file, 'w', proc_en.data)
            readWriter().normal(new_file.as_posix().replace('en.txt', 'nl.txt'), 'w', proc_nl.data)
            readWriter().normal(new_file.as_posix().replace('en.txt', 'en_notok.txt'), 'w', proc_en.data_txt)
            readWriter().normal(new_file.as_posix().replace('en.txt', 'nl_notok.txt'), 'w', proc_nl.data_txt)

            sent_aligned = sent_aligner(proc_en.data, proc_nl.data)
            new_file = Path('/'.join(cur_dir)) / i_list[-2:][0].replace('xml', 'txt') / i_list[-2:][1].replace('_en.xml', '_en-nl_v.lfai')
            readWriter().normal(new_file, 'w', sent_aligned, list_out=False)
            non_verbose_aligned = [i for i in sent_aligned.split('\n')[::3] if i != '']
            new_file = Path('/'.join(cur_dir)) / i_list[-2:][0].replace('xml', 'txt') / i_list[-2:][1].replace('_en.xml', '_en-nl.lfai')
            readWriter().normal(new_file, 'w', non_verbose_aligned)

            aligned_sents = word_aligner.aligned_sent_list(proc_en.data, proc_nl.data, non_verbose_aligned)
            new_file = Path('/'.join(cur_dir)) / i_list[-2:][0].replace('xml', 'txt') / i_list[-2:][1].replace('_en.xml', '_en-nl.lfa')
            readWriter().normal(new_file, 'w', [f'{x} ||| {y}' for x, y in aligned_sents])

            wa = [word_aligner.awesome_astred_align(w_en, w_nl) for w_en, w_nl in aligned_sents]
            new_file = Path('/'.join(cur_dir)) / i_list[-2:][0].replace('xml', 'wa') / i_list[-2:][1].replace('_en.xml', '_en-nl.wa')
            readWriter().normal(new_file, 'w', wa)

            break




if __name__ == '__main__':
    main()
