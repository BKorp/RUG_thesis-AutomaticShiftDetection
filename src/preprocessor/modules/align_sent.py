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
                              env=env).stdout



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
            'LASER': self.term_run('realpath "../../external_tools/LASER"'),
            'DATA': self.term_run('realpath "../../data"'),
            'VECALIGN': self.term_run('realpath "../../external_tools/vecalign"'),
        }

    def retrieve_embedding(self, en_inp, nl_inp):
        ''''''
        with (
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as en,
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as nl,
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as overlaps_en,
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as overlaps_nl,
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as overlaps_en_emb,
            tempfile.NamedTemporaryFile('w+t', encoding='utf-8') as overlaps_nl_emb
        ):
            en.write('\n'.join(en_inp))
            en.seek(0)
            nl.write('\n'.join(nl_inp))
            nl.seek(0)

            self.term_run(f'$VECALIGN/overlap.py -i "{en}" -o "{overlaps_en}" -n 10', self.env)
            self.term_run(f'$VECALIGN/overlap.py -i "{nl}" -o "{overlaps_nl}" -n 10', self.env)

            self.term_run(f'$LASER/tasks/embed/embed.sh "{overlaps_en}" "{overlaps_en_emb}"', self.env)
            self.term_run(f'$LASER/tasks/embed/embed.sh "{overlaps_nl}" "{overlaps_nl_emb}"', self.env)

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
            '$VECALIGN/vecalign.py '
            f'--alignment_max_size {alignment_max_size} '
            f'--src "{en}" '
            f'--tgt "{nl}" '
            f'--src_embed "{overlaps_en}" "{overlaps_en_emb}" '
            f'--print_aligned_text '
            f'--tgt_embed "{overlaps_nl}" "{overlaps_nl_emb}"',
            self.env
        ).splitlines()

        return vecalign

    def aligner(
        self,
        en_inp,
        nl_inp,
    ):
        ''''''
        return self.retrieve_embedding(en_inp, nl_inp)
