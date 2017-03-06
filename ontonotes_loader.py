import on


def load_ontonotes(cfg_file):
    all_subcorps = []
    cfg = on.common.util.load_config(cfg_file)
    a_ontonotes = on.ontonotes(cfg)
    for subcorp in a_ontonotes:
        all_subcorps.append(subcorp)
    return all_subcorps


def get_all_docs(ontonotes, bank_name):
    all_docs = []
    for subcorp in ontonotes:
        assert bank_name in subcorp, '{} bank not exists in {}'.format(
            bank_name, subcorp)
        for doc in subcorp[bank_name]:
            all_docs.append(doc)
    return all_docs


def get_all_coref_docs(ontonotes):
    return get_all_docs(ontonotes, 'coref')


def get_all_name_docs(ontonotes):
    return get_all_docs(ontonotes, 'name')
