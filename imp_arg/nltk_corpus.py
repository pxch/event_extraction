import platform
import re
from os.path import join

from nltk.corpus import BracketParseCorpusReader
from nltk.corpus import NombankCorpusReader
from nltk.corpus import PropbankCorpusReader
from nltk.data import FileSystemPathPointer

system_name = platform.system()
dist_name = platform.linux_distribution()[0]

if system_name == 'Darwin':
    corpus_root = '/Users/pengxiang/corpora/'
elif system_name == 'Linux':
    if dist_name == 'Ubuntu':
        corpus_root = '/scratch/cluster/pxcheng/corpora/'
    elif dist_name == 'CentOS':
        corpus_root = '/work/03155/pxcheng/maverick/corpora/'
else:
    raise RuntimeError('Unrecognized system: {}'.format(system_name))

treebank_root = join(corpus_root, 'penn-treebank-rel3/parsed/mrg/wsj')
treebank_file_pattern = '\d\d/wsj_.*\.mrg'

propbank_root = join(corpus_root, 'propbank-LDC2004T14/data')
propbank_file = 'prop.txt'
propbank_verbs_file = 'verbs.txt'

nombank_root = join(corpus_root, 'nombank.1.0')
nombank_file = 'nombank.1.0_sorted'
nombank_nouns_file = 'nombank.1.0.words'

frame_file_pattern = 'frames/.*\.xml'


def fileid_xform_function(filename):
    result = re.sub(r'^wsj/', '', filename)
    # result = re.sub(r'^wsj/\d\d/', '', filename)
    # result = re.sub(r'\.mrg$', '', result)
    return result


treebank = BracketParseCorpusReader(
    root=treebank_root,
    fileids=treebank_file_pattern,
    tagset='wsj',
    encoding='ascii'
)


propbank = PropbankCorpusReader(
    root=FileSystemPathPointer(propbank_root),
    propfile=propbank_file,
    framefiles=frame_file_pattern,
    verbsfile=propbank_verbs_file,
    parse_fileid_xform=fileid_xform_function,
    parse_corpus=treebank
)


nombank = NombankCorpusReader(
    root=FileSystemPathPointer(nombank_root),
    nomfile=nombank_file,
    framefiles=frame_file_pattern,
    nounsfile=nombank_nouns_file,
    parse_fileid_xform=fileid_xform_function,
    parse_corpus=treebank
)
