import subprocess
import tempfile
from pathlib import Path
import regex as re
import shutil


class sentAligner:
    def __init__(self) -> None:
        pass

    def term_run(self, command, env=None):
        return subprocess.run(command,
                              shell=True,
                              capture_output=True,
                              text=True,
                              env=env)



class reprocAligner(sentAligner):
    def __init__(self, prepare=False) -> None:
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
        if from_scratch:
            shutil.rmtree(self.align_root)

        for p in (self.align_input, self.align_output):
            p.mkdir(parents=True, exist_ok=True)

    def build_index(self):
        self.index_file.touch()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            f.write('index\toriginal_file\tnew_file\n')

    def append_index(self, inp_f):
        idx = inp_f.split('/')[-1].split('.')[0]
        new_f = '/'.join(inp_f.split('/')[:-2]) + '/txt/' + re.sub('en|nl', 'en-nl', idx) + '.txt'
        with open(self.index_file, 'a', encoding='utf-8') as f:
            f.write(f'{idx}\t{inp_f}\t{new_f}\n')

    def fill_input(self, inp_data, inp_f):
        inp_f = inp_f.split('/')[-1].replace('.xml', '.txt').replace('_en', '.en').replace('_nl', '.nl')
        with open(self.align_input / inp_f, 'w', encoding='utf-8') as f:
            f.write('\n'.join(inp_data))

    def read_index(self):
        with open(self.index_file, 'r', encoding='utf-8') as f:
            self.idx = f.read().splitlines()[1:]
        self.idx = [row.split('\t') for row in self.idx]

    def rename_file(self, inp_f):
        for i in self.idx:
            if inp_f == i[1]:
                for f in Path('../reproc_sent_align_files/output').iterdir():
                    re.sub('\.', '_', f, 1)



class newAligner(sentAligner):
    def __init__(self) -> None:
        self.exports()

    def exports(self):
        ''''''
        self.env = {
            'LASER': Path('../../external_tools/LASER').resolve().as_posix(),
            'DATA': Path('../../data').resolve().as_posix(),
            'VECALIGN': Path('../../external_tools/vecalign').resolve().as_posix(),
            'ENVPY': Path('../../env/bin/activate').resolve().as_posix(),
        }

    def retrieve_embedding(self, en_inp, nl_inp):
        ''''''
        with (
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as en,
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as nl,
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as overlaps_en,
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as overlaps_nl,
            tempfile.NamedTemporaryFile('w+b') as overlaps_en_emb,
            tempfile.NamedTemporaryFile('w+b') as overlaps_nl_emb
        ):
            en.write('\n'.join(en_inp))
            en.seek(0)
            nl.write('\n'.join(nl_inp))
            nl.seek(0)

            out = self.term_run(f'source $ENVPY; python $VECALIGN/overlap.py -i "{en.name}" -o "{overlaps_en.name}" -n 10', self.env)
            out = self.term_run(f'source $ENVPY; python $VECALIGN/overlap.py -i "{nl.name}" -o "{overlaps_nl.name}" -n 10', self.env)

            # with open('overlaps_en', 'w', encoding='utf-8') as f_en, open('overlaps_nl', 'w', encoding='utf-8') as f_nl:
            #     f_en.write(overlaps_en.read())
            #     f_nl.write(overlaps_nl.read())

            out = self.term_run(f'source $ENVPY; $LASER/tasks/embed/embed.sh "{overlaps_en.name}" "{overlaps_en_emb.name}"', self.env)
            # print(out.stdout)
            out = self.term_run(f'source $ENVPY; $LASER/tasks/embed/embed.sh "{overlaps_nl.name}" "{overlaps_nl_emb.name}"', self.env)
            # print(out.stdout)

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
        ''''''
        vecalign = self.term_run(
            'source $ENVPY;'
            '$VECALIGN/vecalign.py '
            f'--alignment_max_size {alignment_max_size} '
            f'--src "{en.name}" '
            f'--tgt "{nl.name}" '
            f'--src_embed "{overlaps_en.name}" "{overlaps_en_emb.name}" '
            f'--print_aligned_text '
            f'--tgt_embed "{overlaps_nl.name}" "{overlaps_nl_emb.name}"',
            self.env
        )
        # ).splitlines()

        # print(vecalign.stderr)
        # print(vecalign.stdout)

        return vecalign.stdout

    def __call__(
        self,
        en_inp,
        nl_inp,
    ):
        ''''''
        return self.retrieve_embedding(en_inp, nl_inp)
