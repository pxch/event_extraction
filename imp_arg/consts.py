nominal_predicate_mapping = {
    'bid': 'bid',
    'sale': 'sell',
    'loan': 'loan',
    'cost': 'cost',
    'plan': 'plan',
    'investor': 'invest',
    'price': 'price',
    'loss': 'lose',
    'investment': 'invest',
    'fund': 'fund',
}


nombank_function_tag_mapping = {
    'TMP': 'temporal',
    'LOC': 'location',
    'MNR': 'manner',
    'PNC': 'purpose',
    'NEG': 'negation',
    'EXT': 'extent',
    'ADV': 'adverbial',
    # tags below do not appear in implicit argument dataset
    'DIR': 'directional',
    'PRD': 'predicative',
    'CAU': 'cause',
    'DIS': 'discourse',
    'REF': 'reference'
}


def convert_nombank_label(label):
    if label[:3] == 'ARG':
        if label[3].isdigit():
            return label[:4].lower()
        elif label[3] == 'M':
            function_tag = label.split('-')[1]
            return nombank_function_tag_mapping.get(function_tag, '')
    return ''


core_arg_list = ['arg0', 'arg1', 'arg2', 'arg3', 'arg4']


def is_core_arg(label):
    return label in core_arg_list


def convert_fileid(fileid):
    return fileid[3:11]


corenlp_root = '/Users/pengxiang/corpora/wsj_corenlp/20170613'

prep_vocab_list_file = '../vocab_list/preposition'

corenlp_dict_path = './corenlp_dict.pkl'

predicate_dict_path = './predicate_dict.pkl'
