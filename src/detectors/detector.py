import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from contextual_cosine_sim import SimRunner, contextualSim
from static_cosine_sim import staticSim
from helpers.timestamp import name_timestamp
from alive_progress import alive_it
from collections import defaultdict


def arg_parser():
    parser = ArgumentParser()

    parser.add_argument('-t',
                        '--type',
                        type=str,
                        choices=['all', 'contextual', 'static'],
                        default='all',
                        help='Choose what type of score '
                             '(all, contextual, or static) '
                             'you want to generate for the detection '
                             'of creative shifts (default: all)')

    parser.add_argument(
        '-se',
        '--source_emb',
        type=str,
        default='../../data/_embeddings/fasttext/cc.en.300.mapped.emb',
        help='Give a source embedding if -t=static '
             '(default: ../../data/_embeddings/fasttext/cc.en.300.mapped.emb)'
    )

    parser.add_argument(
        '-te',
        '--target_emb',
        type=str,
        default='../../data/_embeddings/fasttext/cc.nl.300.mapped.emb',
        help='Give a target embedding if -t=static '
             '(default: ../../data/_embeddings/fasttext/cc.nl.300.mapped.emb)'
    )

    parser.add_argument(
        '-di',
        '--data_input',
        type=str,
        default='../../data/2_new_run',
        help='Give a data input, which is the root file for genre output '
             '(default: ../../data/2_new_run)'
    )

    parser.add_argument(
        '-mt',
        '--machine_translation',
        action='store_true',
        help='Tell the system you are using machine translation data (default: False)'

    )

    return parser.parse_args()


def wa_to_pairs(st_tt:str, wa:str):
    '''Returns a tuple consisting of a list of all word pairs per line
    and a list of all sentence pairs using a given sourceText_targetText file
    and a given word alignment file.
    '''
    with open(st_tt, 'r', encoding='utf-8') as f:
        sent_pairs = f.read().splitlines()

    with open(wa, 'r', encoding='utf-8') as f:
        wa_idx = f.read().splitlines()

    sent_pairs = [[sent.split(' ') for sent in i.split(' ||| ')]
                  for i in sent_pairs if '' not in i.split(' ||| ')]

    wa_idx_list = [
        [[int(pair.split('-')[0]), int(pair.split('-')[1])] for pair in line]
        for line in wa_idx if line != ''
    ]

    all_pairs_list = []
    pair_list = []
    for idx, sent_pair in enumerate(sent_pairs):
        for wa_pair in wa_idx_list[idx]:
            pair_list.append([sent_pair[0][wa_pair[0]], sent_pair[1][wa_pair[1]]])
        all_pairs_list.append(pair_list)
        pair_list = []

    return all_pairs_list, sent_pairs


def file_finder(args:arg_parser):
    '''Returns the sentence pair files and word alignment files
    as lists with Path file paths using arguments from the argparser.
    '''
    sent_path = Path(args.data_input)
    machine_translation = args.machine_translation

    if machine_translation:
        sent_pair_files = sorted(sent_path.glob('*/**/*en-mt_t.lfa'))
        sent_wa_files = sorted(sent_path.glob('*/**/*en-mt.wa'))
    else:
        sent_pair_files = sorted(sent_path.glob('*/**/*en-nl_t.lfa'))
        sent_wa_files = sorted(sent_path.glob('*/**/*en-nl.wa'))

    return sent_pair_files, sent_wa_files


def static_sys(args, sent_pair_files, sent_wa_files):
    '''Stores the static MT or static human data with static cosine
    distance scores in a tsv using pandas and gensim for a list of
    sentence pair files and sentence word alignment files.
    '''
    machine_translation = args.machine_translation

    static = staticSim(en_model=args.source_emb, nl_model=args.target_emb)

    static_dict = defaultdict(list)
    for idx, film in enumerate(sent_pair_files):
        wa_sent_pairs, sent_pairs = wa_to_pairs(film, sent_wa_files[idx])
        for sent_idx, sent in enumerate(wa_sent_pairs):
            for pair in sent:
                static_distance, sw, tw, = static.word_emb_cos(*pair)

                static_dict['film'].append(film.as_posix().split('/')[5].split('_')[0])
                static_dict['genre'].append(film.as_posix().split('/')[4].lower())
                static_dict['sent_idx'].append(sent_idx)
                static_dict['src_sent'].append(' '.join(sent_pairs[sent_idx][0]))
                static_dict['tgt_sent'].append(' '.join(sent_pairs[sent_idx][1]))
                static_dict['src'].append(sw)
                static_dict['tgt'].append(tw)
                static_dict['static_cosine'].append(static_distance)
                static_dict['type'].append('human')

    df_static = pd.DataFrame.from_dict(static_dict)

    source_emb = args.source_emb.split('/')[-1].split('.')[-2]
    if machine_translation:
        df_static.to_csv(f'{args.data_input}/{name_timestamp()}-{source_emb}-static_mt.tsv', sep='\t')
    else:
        df_static.to_csv(f'{args.data_input}/{name_timestamp()}-{source_emb}-static.tsv', sep='\t')


def context_sys(args, sent_pair_files, sent_wa_files):
    '''Stores the contextual MT or contextual human data with
    contextual cosine distance scores in a tsv using pandas and
    a sentence-transformer model for a list of sentence pair files
    and sentence word alignment files.
    '''
    machine_translation = args.machine_translation

    cont_embedder = contextualSim()
    context = SimRunner(cont_embedder)

    bar = alive_it(sent_pair_files)
    frames = [context(*context.file_loader(film, sent_wa_files[idx]), film) for idx, film in enumerate(bar)]
    df_context = pd.concat(frames)
    if machine_translation:
        df_context.to_csv(f'{args.data_input}/{name_timestamp()}-context_mt.tsv', sep='\t')
    else:
        df_context.to_csv(f'{args.data_input}/{name_timestamp()}-context.tsv', sep='\t')


def main():
    args = arg_parser()
    sent_pair_files, sent_wa_files = file_finder(args)

    if args.type == 'contextual' or args.type == 'all':
        context_sys(args, sent_pair_files, sent_wa_files)

    if args.type == 'static' or args.type == 'all':
        static_sys(args, sent_pair_files, sent_wa_files)


if __name__ == '__main__':
    main()
