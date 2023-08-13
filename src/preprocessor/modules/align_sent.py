import subprocess
import tempfile
from pathlib import Path
import regex as re
import shutil


class sentAligner:
    '''A sentence aligner class to store a shared function for
    its children classes.
    '''
    def __init__(self) -> None:
        pass

    def term_run(self, command, env=None):
        '''Returns the output of a subprocess using Bash, using a given
        command and environment variables.
        '''
        return subprocess.run(command,
                              shell=True,
                              capture_output=True,
                              text=True,
                              env=env)


class reprocAligner(sentAligner):
    '''A class for reproduction (LF Aligner) sentence alignment'''
    def __init__(self, prepare=False) -> None:
        '''Initializes an object and if prepare is set to True, prepares
        folders and indices. Sets attributes for the purposes of creating
        and finding input and output files for all alignments.
        '''
        self.prepare = prepare
        self.align_root = Path('./reproc_sent_align_files')
        self.align_input = self.align_root / 'input'
        self.align_output = self.align_root / 'output'
        self.index_file = self.align_root / 'index.tsv'

        if prepare:
            self.make_folder()
            self.build_index()
        else:
            self.read_index()

    def __call__(self, inp_f=None, inp_data=None) -> None:
        '''Run the reproduction system, creating an index for manual annotation,
        or reading manual annotation if it exists.
        '''
        if self.prepare:
            if inp_f == None or inp_data == None:
                print('Make sure to input data for manual sentence alignment!')
                return
            self.append_index(inp_f)
            self.fill_input(inp_data, inp_f)
        else:
            self.rename_file()
            self.to_data()

    def make_folder(self, from_scratch=True):
        '''Make a folder for the input and output of the aligner.
        If done from scratch, any previous folder is rebuilt from
        scratch.
        '''
        if from_scratch:
            shutil.rmtree(self.align_root)

        for p in (self.align_input, self.align_output):
            p.mkdir(parents=True, exist_ok=True)

    def build_index(self):
        '''Create an index file and write the headers of the file
        to it.
        '''
        self.index_file.touch()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            f.write('index\toriginal_file\tnew_file\n')

    def append_index(self, inp_f):
        '''Append the input file to the index file.'''
        idx = inp_f.split('/')[-1].split('.')[0]
        new_f = '/'.join(inp_f.split('/')[:-2]) + '/txt/' + re.sub('en|nl', 'en-nl', idx) + '.txt'
        with open(self.index_file, 'a', encoding='utf-8') as f:
            f.write(f'{idx}\t{inp_f}\t{new_f}\n')

    def fill_input(self, inp_data, inp_f):
        '''Write the input data to the input file.'''
        inp_f = inp_f.split('/')[-1].replace('.xml', '.txt').replace('_en', '.en').replace('_nl', '.nl')
        with open(self.align_input / inp_f, 'w', encoding='utf-8') as f:
            f.write('\n'.join(inp_data))

    def read_index(self):
        '''Read the index file and put it into the idx attribute.'''
        with open(self.index_file, 'r', encoding='utf-8') as f:
            self.idx = f.read().splitlines()[1:]
        self.idx = [row.split('\t') for row in self.idx]

    def rename_file(self, inp_f):
        '''Rename a given input file.'''
        for i in self.idx:
            if inp_f == i[1]:
                for f in Path('../reproc_sent_align_files/output').iterdir():
                    re.sub('\.', '_', f, 1)



class newAligner(sentAligner):
    '''A class for the embeddings from Using the English and Dutch
    input, or the new (Vecalign) sentence alignment
    '''
    def __init__(self) -> None:
        '''Initialize the object and run the exports function to
        prepare bash variables.
        '''
        self.exports()

    def exports(self):
        '''Prepare the environment variables for Bash.'''
        self.env = {
            'LASER': Path('../../external_tools/LASER').resolve().as_posix(),
            'DATA': Path('../../data').resolve().as_posix(),
            'VECALIGN': Path('../../external_tools/vecalign').resolve().as_posix(),
            'ENVPY': Path('../../env/bin/activate').resolve().as_posix(),
        }

    def overlaps_embeddings(self, en_inp, nl_inp):
        '''Return the output of vecalign using vecalign sentence
        overlaps and LASER embeddings for a given English input and
        Dutch input.
        '''
        with (
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as en,
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as nl,
        ):
            en.write('\n'.join(en_inp))
            en.seek(0)
            nl.write('\n'.join(nl_inp))
            nl.seek(0)
            overlaps_en = './overlaps_en'
            overlaps_nl = './overlaps_nl'
            overlaps_en_emb = './overlaps_en_emb'
            overlaps_nl_emb = './overlaps_nl_emb'

            out = self.term_run(f'source $ENVPY; python $VECALIGN/overlap.py -i "{en.name}" -o "{overlaps_en}" -n 10', self.env)
            if out.stderr != '':
                print(out.stderr)

            out = self.term_run(f'source $ENVPY; python $VECALIGN/overlap.py -i "{nl.name}" -o "{overlaps_nl}" -n 10', self.env)
            if out.stderr != '':
                print(out.stderr)

            out = self.term_run(f'source $ENVPY; $LASER/tasks/embed/embed.sh "{overlaps_en}" "{overlaps_en_emb}"', self.env)
            if out.stderr != '':
                print(out.stderr)

            out = self.term_run(f'source $ENVPY; $LASER/tasks/embed/embed.sh "{overlaps_nl}" "{overlaps_nl_emb}"', self.env)
            if out.stderr != '':
                print(out.stderr)

            return self.vecalign_runner(en, nl,
                                        overlaps_en, overlaps_nl,
                                        overlaps_en_emb, overlaps_nl_emb)

    def vecalign_runner(
        self,
        en,
        nl,
        overlaps_en,
        overlaps_nl,
        overlaps_en_emb,
        overlaps_nl_emb,
        alignment_max_size=8,
    ):
        '''Return the alignment output of Vecalign for given overlaps
        and embeddings. Additionally, the alignment size can be set.
        '''
        vecalign = self.term_run(
            'source $ENVPY;'
            '$VECALIGN/vecalign.py '
            f'--alignment_max_size {alignment_max_size} '
            f'--src "{en.name}" '
            f'--tgt "{nl.name}" '
            f'--src_embed "{overlaps_en}" "{overlaps_en_emb}" '
            f'--print_aligned_text '
            f'--tgt_embed "{overlaps_nl}" "{overlaps_nl_emb}"',
            self.env
        )
        self.term_run(f'rm {overlaps_en} {overlaps_en_emb} {overlaps_nl} {overlaps_nl_emb}')
        return vecalign.stdout

    def __call__(
        self,
        en_inp,
        nl_inp,
    ):
        '''Return the sentence alignments from vecalign by way of
        sentence overlaps and LASER embeddings.
        '''
        return self.overlaps_embeddings(en_inp, nl_inp)
