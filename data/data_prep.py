import gdown
import re
import shutil
import string
from pathlib import Path as path
from zipfile import ZipFile


def prepare_data(url, dir_names, download_overwrite, quiet=False):
    '''Download and extract the zip file containing the data from
    Google Drive. If download overwrite, remove earlier downloads, and
    if quiet is True, do not print information about download.
    '''
    data_zip = path('./data.zip')

    if download_overwrite:
        data_zip.unlink(missing_ok=True)
        for d in dir_names:
            try:
                shutil.rmtree(path(d))
            except Exception:
                continue

    if not data_zip.exists():
        gdown.download(url=url, use_cookies=False, quiet=quiet, fuzzy=True)

    with ZipFile('./data.zip') as zfile:
        zfile.extractall('.')
    path('./data').rename(dir_names[0])


def remove_char(fname):
    '''Return filnames with undesirable characters removed.'''
    relevant_punct = re.sub(r'[_/-]', '', string.punctuation)
    for i in relevant_punct:
        fname = fname.replace(i, '')
    fname = fname.replace(' ', '_')

    return fname


def unwanted_file(cur_path):
    '''Return True if an unwanted file is found in current path else
    return False.
    '''
    if re.findall(r'fr\.|.DS_Store', cur_path):
        return True
    else:
        return False


def create_folder(cur_dir, lst=[]):
    '''Create a file in current directory for file in list.'''
    for f in lst:
        f_new = cur_dir.replace('0_original/', f)
        path(f_new).mkdir(parents=True, exist_ok=False)


def copy_file(f_old, lst=[]):
    '''Copy old files and replace with file for file in list.'''
    for f in lst:
        f_new = path(f_old.as_posix().replace('0_original/', f))
        f_new.write_bytes(f_old.read_bytes())


def main():
    url = ('https://drive.google.com/file/d/'
           '1Z2gMpJHgY08I4jTB3DHuclpxgLlrlWsK/view?usp=sharing')
    main_dir = path('./0_original')
    new_dirs = ['1_rerun/', '2_new_run/']

    download_overwrite = False
    if not main_dir.exists() or download_overwrite:
        prepare_data(url, [main_dir.as_posix(),]
                     + new_dirs, download_overwrite)

    for d in main_dir.glob('*/*'):
        if d.is_dir():
            d = d.replace(remove_char(d.as_posix()))
            for f in d.glob('**/*'):
                if unwanted_file(f.as_posix()):
                    f.unlink()
                else:
                    # Remove (trailing) spaces in filenames
                    f.replace(re.sub(r' *', '', f.as_posix()))

    new_dirs = ['1_rerun/', '2_new_run/']
    for d in main_dir.glob('*/*/*'):
        if d.is_dir():
            create_folder(d.as_posix(), new_dirs)
            if '/xml' in d.as_posix():
                for f in d.iterdir():
                    copy_file(f, new_dirs)


if __name__ == '__main__':
    main()
