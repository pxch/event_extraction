from warnings import warn

VALID_NER_TAGS = ['PER', 'ORG', 'LOC', 'TEMP', 'NUM', 'MISC']

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


def convert_ontonotes_ner_tag(tag, to_corenlp=False):
    if tag == '':
        return ''
    if tag not in ONTONOTES_NER_TAGS:
        warn('{} is not a valid NER tag from ontonotes'.format(tag))
        return ''
    if to_corenlp:
        return ONTONOTES_TO_CORENLP_MAPPING[tag]
    else:
        return ONTONOTES_TO_VALID_MAPPING[tag]


def convert_corenlp_ner_tag(tag):
    if tag == '':
        return ''
    if tag not in CORENLP_NER_TAGS:
        warn('{} is not a valid NER tag from Stanford CoreNLP'.format(tag))
        return ''
    return CORENLP_TO_VALID_MAPPING[tag]
