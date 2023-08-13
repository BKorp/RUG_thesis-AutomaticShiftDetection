import tempfile
import subprocess
from astred import Aligner
import json


class wordAligner:
    '''A class for word alignment'''
    def __init__(self) -> None:
        pass

    def term_run(self, command, env=None):
        '''Prepare the environment variables for Bash.'''
        return subprocess.run(command,
                              shell=True,
                              capture_output=True,
                              text=True,
                              env=env)



class reprocAligner(wordAligner):
    '''A class for reproduction (awesome_align) word alignment'''
    def awesome_align(
        self,
        data,
        extraction='softmax',
        batch_size=32,
        model_name_or_path='bert-base-multilingual-cased',
    ):
        '''Return the output of awesome align using bash with the data
        and default variables for extraction, batch size and model
        '''
        with (
            tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8')
            as output_file,
            tempfile.NamedTemporaryFile(mode='w+t', encoding='utf-8')
            as data_file
        ):
            data_file.write('\n'.join(data))
            data_file.seek(0)
            term_info = self.term_run(
                'CUDA_VISIBLE_DEVICES=0 awesome-align '
                f'--output_file={output_file.name} '
                f'--model_name_or_path={model_name_or_path} '
                f'--data_file="{data_file.name}" '
                f'--extraction "{extraction}" '
                f'--batch_size {str(batch_size)}'
            )
            # print(term_info)
            output = output_file.read().splitlines()

        return output


class newAligner(wordAligner):
    '''A class for (awesome_align) word alignment'''
    def __init__(self) -> None:
        '''Initialize an object while storing an Aligner object to the
        aligner variable
        '''
        self.aligner = Aligner()

    def aligned_sent_list(self, en, nl, lfa):
        '''Return a list of aligned sentences for a given English,
        Dutch, and alignment data.
        '''
        aligned_indices = []
        for align in lfa:
            align = [json.loads(i)
                     if i != [] else '' for i in align.split(':')[:2]]
            if align != '':
                aligned_indices.append(align)

        aligned_sents = [[' '.join([en[i] for i in w_en_list]), ' '.join([nl[i] for i in w_nl_list])] for w_en_list, w_nl_list in aligned_indices]

        return aligned_sents

    def awesome_astred_align(self, src, tgt):
        '''Return the pharaoh format word alignments for a given source
        and target sentence.
        '''
        aligns = self.aligner.align(src, tgt)
        pharaoh = ' '.join([f'{x}-{y}' for x, y in aligns])
        return pharaoh
