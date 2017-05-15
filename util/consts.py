# path to Ontonotes corpus, only valid on my laptop
ONTONOTES_SOURCE = \
    '/Users/pengxiang/corpora/ontonotes-release-5.0/data/files/data/'
ONTONOTES_ANNOTATIONS_SOURCE = ONTONOTES_SOURCE + 'english/annotations/'

# Identifiers for Ontonotes sub-corpora with *.name and *.coref files
# nw-wsj has *.name and *.coref with part of its files, so not included for now
VALID_ONTONOTES_CORPORA = [
    'english-bn-cnn',
    'english-bn-voa',
    'english-nw-xinhua',
    # 'english-nw-wsj',
    # 'english-nw-wsj_filter'
]

# dependency type from CoreNLP parsed corpus to extract dependencies
# now use enhanced-plus-plus dependencies from universal dependency
# deprecate collapsed-ccprocessed dependencies from Standford dependency
CORENLP_DEPENDENCY_TYPE = 'enhanced-plus-plus-dependencies'
# CORENLP_DEPENDENCY_TYPE = 'collapsed-ccprocessed-dependencies'

# 10 most frequent prepositions (predefined from web)
# deprecated now, use vocab_list/preposition instead
FREQ_PREPS = ['of', 'in', 'to', 'for', 'with',
              'on', 'at', 'from','by', 'about']

# top frequent verbs to be excluded from script learning

# 10 most frequent verbs (predefined from web)
# deprecated now, use STOP_PREDS defined below
# STOP_VERBS = ['have', 'take', 'get', 'become', 'like',
#               'want', 'seem', 'think', 'make', 'need']

# 10 most frequent predicate from training corpus (English Wikipedia 20160901)
STOP_PREDS = ['have', 'include', 'use', 'make', 'play',
              'take', 'win', 'give', 'serve', 'receive']

# NER related constants, used in constructing document.Document

# valid NER tags (combined from CoreNLP and Ontonotes)
VALID_NER_TAGS = ['PER', 'ORG', 'LOC', 'TEMP', 'NUM']

# all possible NER tags from Ontonotes
ONTONOTES_NER_TAGS = [
    'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'DATE', 'TIME',
    'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'EVENT',
    'WORK_OF_ART', 'LAW', 'LANGUAGE'
]

# all possible NER tags from CoreNLP
CORENLP_NER_TAGS = [
    'PERSON', 'ORGANIZATION', 'LOCATION', 'MISC', 'MONEY', 'NUMBER', 'ORDINAL',
    'PERCENT', 'DATE', 'TIME', 'DURATION', 'SET'
]

# mappings from Ontonotes NER tags to CoreNLP NER tags (unused for now)
ONTONOTES_TO_CORENLP_MAPPING = {
    'PERSON': 'PERSON',
    'NORP': 'MISC',
    'FAC': 'LOCATION',
    'ORG': 'ORGANIZATION',
    'GPE': 'LOCATION',
    'LOC': 'LOCATION',
    'PRODUCT': 'MISC',
    'DATE': 'DATE',
    'TIME': 'TIME',
    'PERCENT': 'PERCENT',
    'MONEY': 'MONEY',
    'QUANTITY': '',
    'ORDINAL': 'ORDINAL',
    'CARDINAL': 'NUMBER',
    'EVENT': 'MISC',
    'WORK_OF_ART': '',
    'LAW': '',
    'LANGUAGE': 'MISC'
}

# mappings from Ontonotes NER tags to valid NER tags
ONTONOTES_TO_VALID_MAPPING = {
    'PERSON': 'PER',
    'NORP': 'MISC',
    'FAC': 'LOC',
    'ORG': 'ORG',
    'GPE': 'LOC',
    'LOC': 'LOC',
    'PRODUCT': 'MISC',
    'DATE': 'TEMP',
    'TIME': 'TEMP',
    'PERCENT': 'NUM',
    'MONEY': 'NUM',
    'QUANTITY': '',
    'ORDINAL': 'NUM',
    'CARDINAL': 'NUM',
    'EVENT': 'MISC',
    'WORK_OF_ART': '',
    'LAW': '',
    'LANGUAGE': 'MISC'
}

# mappings from CoreNLP NER tags to valid NER tags
CORENLP_TO_VALID_MAPPING = {
    'PERSON': 'PER',
    'ORGANIZATION': 'ORG',
    'LOCATION': 'LOC',
    'MISC': 'MISC',
    'MONEY': 'NUM',
    'NUMBER': 'NUM',
    'ORDINAL': 'NUM',
    'PERCENT': 'NUM',
    'DATE': 'TEMP',
    'TIME': 'TEMP',
    'DURATION': 'TEMP',
    'SET': 'TEMP'
}

# set of escape characters in constructing the word/lemma/pos of a token
ESCAPE_CHAR_SET = [' // ', '/', ';', ',', ':', '-']

# mappings from escape characters to their representations
ESCAPE_CHAR_MAP = {
    ' // ': '@slashes@',
    '/': '@slash@',
    ';': '@semicolon@',
    ',': '@comma@',
    ':': '@colon@',
    '-': '@dash@',
    '_': '@underscore@'}

# relative path to vocabulary list files
PRED_VOCAB_LIST_FILE = './vocab_list/predicate_min_100'
ARG_VOCAB_LIST_FILE = './vocab_list/argument_min_500'
NER_VOCAB_LIST_FILE = './vocab_list/name_entity_min_500'
PREP_VOCAB_LIST_FILE = './vocab_list/preposition'

PRED_VOCAB_COUNT_FILE = './vocab_list/predicate_min_100_count'
PRED_COUNT_THRES = 100000
