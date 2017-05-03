import on

from util import consts


def get_default_ontonotes_cfg(data_in=consts.ONTONOTES_SOURCE):
    cfg = on.common.util.FancyConfigParser()
    cfg.add_section('corpus')
    cfg.set('corpus', '__name__', 'corpus')
    cfg.set('corpus', 'data_in', data_in)
    cfg.set('corpus', 'granularity', 'section')
    cfg.set('corpus', 'banks', 'parse coref name')
    cfg.set('corpus', 'wsd-indexing', 'word')
    cfg.set('corpus', 'name-indexing', 'word')
    return cfg


def load_ontonotes(corpora):
    assert corpora in consts.VALID_ONTONOTES_CORPORA, \
        'ontonotes corpora can only be one of {}'.format(
            consts.VALID_ONTONOTES_CORPORA)
    cfg = get_default_ontonotes_cfg()
    cfg.set('corpus', 'load', corpora)
    a_ontonotes = on.ontonotes(cfg)
    all_subcorps = []
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
