import re
from collections import defaultdict

from nltk.corpus import BracketParseCorpusReader
from nltk.corpus import NombankCorpusReader
from nltk.corpus import PropbankCorpusReader
from nltk.data import FileSystemPathPointer

treebank_root = '/Users/pengxiang/corpora/penn-treebank-rel3/parsed/mrg/wsj'
treebank_file_pattern = '\d\d/wsj_.*\.mrg'

propbank_root = '/Users/pengxiang/corpora/propbank-LDC2004T14/data'
propbank_file = 'prop.txt'
propbank_verbs_file = 'verbs.txt'

nombank_root = '/Users/pengxiang/corpora/nombank.1.0'
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
