from modules.to_srt import to_srt as external_srt_sys
from modules.to_txt import subtitlerNew, subtitlerRepro
from modules import align_sent
from modules import align_word
import codecs
import re
from pathlib import Path
from alive_progress import alive_it
from argparse import ArgumentParser


class readWriter():
    '''A reader and writer class.'''
    def __init__(self) -> None:
        pass

    def normal(self, fname, readwrite='r', out_data=None, list_out=True):
        '''A reader and writer that returns normal formatted data.'''
        with open(fname, readwrite, encoding='utf-8') as f:
            if readwrite == 'r':
                return f.read().splitlines()
            else:
                if list_out:
                    f.write('\n'.join(out_data))
                else:
                    f.write(out_data)

    def codec(self, fname, readwrite='r', out_data=None):
        '''A reader and writer that returns codec formatted data.'''
        if readwrite == 'r':
            with codecs.open(fname, 'rb', encoding="utf-8") as f:
                return f.read()
        else:
            with codecs.open(fname, 'wb', encoding="utf-8") as f:
                f.write(out_data)


class fileConverter:
    '''A file converter class for xml to srt to txt and tokenization'''
    def __init__(self, data) -> None:
        '''Initializes the object with a starting data attribute.
        Additionally sets the attributes for the extension and txt
        conversion.
        '''
        self.data_orig = data
        self.data = data
        self.init_ext = '.xml'
        self.txt_convert = None

    def to_srt(self):
        '''Generates the srt conversion using an external srt system and
        upates the data attribute.
        '''
        self.data_srt = external_srt_sys(self.data, self.init_ext)
        self.data = self.data_srt

    def to_txt(self):
        '''Generates the txt conversion using an external txt system and
        updates the data attribute.
        '''
        convert = self.txt_convert
        self.data_txt = [convert.regex_steps(i) for i in convert.subtitle_prep(self.data)]
        self.data_txt = [re.sub(r' +', ' ', i) for i in self.data_txt if i != '']
        self.data = self.data_txt

    def tokenize(self, lang='en'):
        '''Generates the tokenized text using an external tokenizer and
        updates the data attribute.
        '''
        convert = self.txt_convert
        self.data_tok = [convert.tokenize(i, lang) for i in self.data_txt]
        self.data = self.data_tok


class reproc(fileConverter):
    '''A conversion class for reproducing van der Heden's
    original system.
    '''
    def __init__(self, data, aligner, fname) -> None:
        '''Initializes the object while settings the data attributes form
        the parent fileConverter object. Subsequently sets the attribute
        for the text converter and runs the srt and txt conversion.
        '''
        super().__init__(data)
        self.to_srt()
        self.txt_convert = subtitlerRepro('moses')
        self.to_txt()
        # self.sent_align(aligner, fname)

    def sent_align(self, repro_aligner, fname):
        '''Aligns sentences based on a given aligner and file name.'''
        repro_aligner(fname, self.data)


class new(fileConverter):
    '''A conversion class for the updated conversion system.'''
    def __init__(self, data, lang) -> None:
        '''Initializes the object while settings the data attributes form
        the parent fileConverter object. Subsequently sets the attribute
        for the text converter and runs the srt and txt conversion.
        '''
        super().__init__(data)
        self.to_srt()
        self.txt_convert = subtitlerNew('stanza')
        self.to_txt()
        self.tokenize(lang=lang)


class main:
    '''The main class for running the preprocessor.'''
    def __init__(self) -> None:
        '''Initializes the object while running the main function.'''
        self.main()

    def arg_parser(self):
        '''Returns the arguments given to the program at runtime.'''
        parser = ArgumentParser()

        parser.add_argument('-t',
                            '--type',
                            type=str,
                            choices=['all', 'reproc', 'new'],
                            default='all',
                            help='Choose what type of data '
                                 '(all, reproc, or new) '
                                 'you want to process (default: all)')

        parser.add_argument(
            '-r',
            '--root',
            type=str,
            default='../../data',
            help='Give the root folder for all data folders '
                 '(default: ../../data)'
        )

        return parser.parse_args()

    def data_xml_to_txt(self, f, f_template):
        '''Returns the new processing data objects for English
        and Dutch which are processed, converted, and stored
        in files for srt and txt data.
        '''
        proc_en = new(readWriter().codec(f), 'en')
        proc_nl = new(readWriter().codec(f.as_posix().replace('en.xml', 'nl.xml')), 'nl')

        readWriter().normal(f_template.as_posix().replace('en.txt', 'en.srt').replace('/txt/', '/srt/'), 'w', proc_en.data_srt, list_out=False)
        readWriter().normal(f_template.as_posix().replace('en.txt', 'nl.srt').replace('/txt/', '/srt/'), 'w', proc_nl.data_srt, list_out=False)

        readWriter().normal(f_template.as_posix().replace('en.txt', 'en_t.txt'), 'w', proc_en.data)
        readWriter().normal(f_template.as_posix().replace('en.txt', 'nl_t.txt'), 'w', proc_nl.data)
        readWriter().normal(f_template, 'w', proc_en.data_txt)
        readWriter().normal(f_template.as_posix().replace('en.txt', 'nl.txt'), 'w', proc_nl.data_txt)

        return proc_en, proc_nl

    def data_vecalign(self, proc_en, proc_nl, cur_dir, f_list):
        '''Returns aligned sentences using given English and Dutch
        processing objects. Stores the data using a given directory
        and file_list. The alignments are broken up into three versions:
        alignment index (i), alignment non-tokenized, and tokenized (t).
        '''
        sent_aligned = self.sent_aligner(proc_en.data_txt, proc_nl.data_txt)

        non_verbose_aligned = [i for i in sent_aligned.split('\n')[::3] if i != '']
        new_file = (
            Path('/'.join(cur_dir))
            / f_list[-2:][0].replace('xml', 'txt')
            / f_list[-2:][1].replace('_en.xml', '_en-nl_i.lfa')
        )
        readWriter().normal(new_file, 'w', non_verbose_aligned)

        aligned_sents = self.word_aligner.aligned_sent_list(proc_en.data_txt, proc_nl.data_txt, non_verbose_aligned)
        new_file = (
            Path('/'.join(cur_dir))
            / f_list[-2:][0].replace('xml', 'txt')
            / f_list[-2:][1].replace('_en.xml', '_en-nl.lfa')
        )
        readWriter().normal(new_file, 'w', [f'{x} ||| {y}' for x, y in aligned_sents])

        aligned_sents = self.word_aligner.aligned_sent_list(proc_en.data_tok, proc_nl.data_tok, non_verbose_aligned)
        new_file = (
            Path('/'.join(cur_dir))
            / f_list[-2:][0].replace('xml', 'txt')
            / f_list[-2:][1].replace('_en.xml', '_en-nl_t.lfa')
        )
        readWriter().normal(new_file, 'w', [f'{x} ||| {y}' for x, y in aligned_sents])

        return aligned_sents

    def data_awesome(self, aligned_sents, cur_dir, f_list):
        '''Stores word alignments in a wa file for aligned sentences
        in a given dir using a given file list.
        '''
        wa = [self.word_aligner.awesome_astred_align(w_en, w_nl) for w_en, w_nl in aligned_sents]

        new_file = (
            Path('/'.join(cur_dir))
            / f_list[-2:][0].replace('xml', 'wa')
            / f_list[-2:][1].replace('_en.xml', '_en-nl.wa')
        )
        readWriter().normal(new_file, 'w', wa)

    def repro_flow(self):
        '''Runs the reproduction work-flow for all xml files in
        the dataset.
        '''
        repro_aligner = align_sent.reprocAligner()

        bar = alive_it(sorted(self.data_paths[1].glob('*/**/*en.xml')))
        for f in bar:
            f_list = f.as_posix().split('/')
            cur_dir = f_list[:-2]
            file_template = (
                Path('/'.join(cur_dir))
                / f_list[-2:][0].replace('xml', 'txt')
                / f_list[-2:][1].replace('.xml', '.txt')
                )
        for f in bar:
            f_list = f.as_posix().split('/')
            cur_dir = f_list[:-2]
            new_file = (
                Path('/'.join(cur_dir))
                / f_list[-2:][0].replace('xml', 'txt')
                / f_list[-2:][1].replace('.xml', '.txt')
            )

            proc = reproc(readWriter().codec(i), repro_aligner, i.as_posix())
            readWriter().normal(new_file, 'w', proc.data)


        bar = alive_it(sorted(self.data_paths[1].glob('*/**/*.lfa')))
        for f in bar:
            with open(f, 'r', encoding='utf-8') as fi:
                data = fi.read().splitlines()

            data = [i for i in data
                    if i.split(' ||| ')[0] != ''
                    and i.split(' ||| ')[1] != '']
            try:
                wa_data = align_word.reprocAligner().awesome_align(data=data)
            except Exception:
                print(f'error found for {f}')
                continue

            with open(
                f.as_posix().replace('/txt/', '/wa/').replace('.lfa', '.wa'),
                'w',
                encoding='utf-8'
            ) as fi:
                fi.write('\n'.join(wa_data))

    def new_flow(self):
        '''Runs the new work-flow for all xml files in
        the dataset.
        '''
        self.sent_aligner = align_sent.newAligner()
        self.word_aligner = align_word.newAligner()

        bar = alive_it(sorted(self.data_paths[2].glob('*/**/*en.xml')))
        for f in bar:
            f_list = f.as_posix().split('/')
            cur_dir = f_list[:-2]
            f_template = (
                Path('/'.join(cur_dir))
                / f_list[-2:][0].replace('xml', 'txt')
                / f_list[-2:][1].replace('.xml', '.txt')
                )

            proc_en, proc_nl = self.data_xml_to_txt(f, f_template)
            aligned_sents = self.data_vecalign(proc_en, proc_nl, cur_dir, f_list)
            self.data_awesome(aligned_sents, cur_dir, f_list)

    def main(self):
        args = self.arg_parser()
        self.root_path = args.root

        self.data_paths = [
            f for f in sorted(Path(self.root_path).iterdir())
            if f.is_dir()
        ]

        # The original reproduction workflow has been disabled, as
        # it requires manual annotation and clutters the folders.
        # It was not used beyond early setup and testing, and may
        # therefore not work perfectly

        # if args.type == 'all':
        #     self.repro_flow()
        #     self.new_flow()
        # elif args.type == 'reproc':
        #     self.repro_flow()
        # elif args.type == 'new':
        self.new_flow()


if __name__ == '__main__':
    main()
