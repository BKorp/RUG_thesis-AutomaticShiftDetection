from modules import align_word
from pathlib import Path
from modules.to_txt import subtitler
from alive_progress import alive_it


def read_writer(fname, readwrite='r', out_data=None, list_out=True):
        '''Returns a file as a list of lines or writes to a given
        file name. Alternatively, can be specified to not be a
        list output for a basic write command.
        '''
        with open(fname, readwrite, encoding='utf-8') as f:
            if readwrite == 'r':
                return f.read().splitlines()
            else:
                if list_out:
                    f.write('\n'.join(out_data))
                else:
                    f.write(out_data)


def main():
    tokenizer = subtitler('stanza')
    aligner = align_word.newAligner()

    data_path = Path('../../data/2_new_run')
    aligned_mt_bar = alive_it(sorted(data_path.glob('**/*_en-mt.lfa')))
    for f in aligned_mt_bar:
        tokenized_f = f.as_posix().replace('.lfa', '_t.lfa')
        word_aligned_f = f.as_posix().replace('/txt/', '/wa/').replace('.lfa', '.wa')

        lfa = read_writer(f.as_posix())
        en, nl = zip(*[line.split(' ||| ') for line in lfa])

        en_t = [tokenizer.tokenize(i, 'en') for i in en]
        nl_t = [tokenizer.tokenize(i, 'nl') for i in nl]

        lfa_t_list = list(zip(en_t, nl_t))
        lfa_t = [' ||| '.join(en_nl) for en_nl in lfa_t_list]
        read_writer(tokenized_f, 'w', out_data=lfa_t)

        wa = [aligner.awesome_astred_align(src, tgt) for src, tgt in lfa_t_list]
        read_writer(word_aligned_f, 'w', out_data=wa)

if __name__ == '__main__':
    main()