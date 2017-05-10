ONTONOTES_SOURCE = \
    '/Users/pengxiang/corpora/ontonotes-release-5.0/data/files/data/'
ONTONOTES_ANNOTATIONS_SOURCE = ONTONOTES_SOURCE + 'english/annotations/'

VALID_ONTONOTES_CORPORA = [
    'english-bn-cnn',
    'english-bn-voa',
    'english-nw-xinhua',
    # 'english-nw-wsj',
    # 'english-nw-wsj_filter'
]

CORENLP_DEPENDENCY_TYPE = 'enhanced-plus-plus-dependencies'
# CORENLP_DEPENDENCY_TYPE = 'collapsed-ccprocessed-dependencies'

FREQ_PREPS = ['of', 'in', 'to', 'for', 'with',
              'on', 'at', 'from','by', 'about']

STOP_VERBS = ['have', 'take', 'get', 'become', 'like',
              'want', 'seem', 'think', 'make', 'need']

VALID_NER_TAGS = ['PER', 'ORG', 'LOC', 'TEMP', 'NUM']

ONTONOTES_NER_TAGS = [
    'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'DATE', 'TIME',
    'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'EVENT',
    'WORK_OF_ART', 'LAW', 'LANGUAGE'
]

CORENLP_NER_TAGS = [
    'PERSON', 'ORGANIZATION', 'LOCATION', 'MISC', 'MONEY', 'NUMBER', 'ORDINAL',
    'PERCENT', 'DATE', 'TIME', 'DURATION', 'SET'
]

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


ESCAPE_CHAR_SET = [' // ', '/', ';', ',', ':', '-']

ESCAPE_CHAR_MAP = {
    ' // ': '@slashes@',
    '/': '@slash@',
    ';': '@semicolon@',
    ',': '@comma@',
    ':': '@colon@',
    '-': '@dash@',
    '_': '@underscore@'}

PRED_VOCAB_LIST_FILE = './vocab_list/predicate_min_100'
ARG_VOCAB_LIST_FILE = './vocab_list/argument_min_500'
NER_VOCAB_LIST_FILE = './vocab_list/name_entity_min_500'
PREP_VOCAB_LIST_FILE = './vocab_list/preposition'
