from copy import deepcopy
from document import *
from sentence import *
from coreference import *
from os.path import join
from warnings import warn
import consts

ONTONOTES_SOURCE = '/Users/pengxiang/corpora/ontonotes-release-5.0/' \
                   'data/files/data/english/annotations/'

def read_coref_link(coref_link):
    mention = Mention(
        coref_link.sentence_index,
        coref_link.start_word_index,
        coref_link.end_word_index + 1)
    if coref_link.subtree is not None:
        mention.set_attrib(
            'text', coref_link.subtree.get_trace_adjusted_word_string())
    return mention


def read_coref_chain(coref_idx, coref_chain):
    coref = Coref(coref_idx)
    for coref_link in coref_chain:
        coref.add_mention(read_coref_link(coref_link))
    return coref


def read_coref_doc(coref_doc):
    all_corefs = []
    coref_idx = 0
    for coref_chain in coref_doc:
        if coref_chain.type == 'IDENT':
            coref = read_coref_chain(coref_idx, coref_chain)
            all_corefs.append(deepcopy(coref))
            coref_idx += 1
    return all_corefs


def read_name_doc(name_doc):
    all_name_entities = []
    for name_entity_set in name_doc:
        for name_entity_hash in name_entity_set:
            for name_entity in name_entity_hash:
                all_name_entities.append(name_entity)
    return all_name_entities


def add_name_entity_to_doc(doc, name_entity):
    assert name_entity.__class__.__name__ == 'name_entity', \
        'name_entity must be a name_entity instance'
    sent = doc.get_sent(name_entity.sentence_index)
    for token_idx in range(
            name_entity.start_word_index, name_entity.end_word_index + 1):
        token = sent.get_token(token_idx)
        # map ontonotes ner tags to coerse grained ner tags
        token.set_attrib(
            'ner', consts.convert_ontonotes_ner_tag(name_entity.type))


def read_conll_depparse(input_path):
    fin = open(input_path, 'r')

    all_sents = []
    sent_idx = 0
    sent = Sent(sent_idx)

    for line_idx, line in enumerate(fin.readlines()):
        if line == '\n':
            all_sents.append(deepcopy(sent))
            sent_idx += 1
            sent = Sent(sent_idx)
        else:
            items = line.strip().split('\t')
            try:
                token_idx = int(items[0])
            except ValueError:
                continue
            if token_idx == len(sent.tokens):
                warn('Line #{}\n\t{}\nhas duplicated token index, '
                     'neglected.'.format(line_idx, line.strip()))
                continue
            word = items[1]
            lemma = items[2]
            pos = items[4]
            sent.add_token(Token(word, lemma, pos))
            try:
                head_idx = int(items[6])
            except ValueError:
                continue
            dep_label = items[7]
            if dep_label != 'root':
                sent.add_dep(Dep(dep_label, head_idx - 1,
                                 token_idx - 1,extra=False))
            if items[8] != '_':
                for e_dep in items[8].strip().split('|'):
                    try:
                        e_dep_head_idx = int(e_dep.split(':')[0])
                    except ValueError:
                        continue
                    e_dep_label = ':'.join(e_dep.split(':')[1:])
                    sent.add_dep(Dep(e_dep_label, e_dep_head_idx - 1,
                                     token_idx - 1, extra=True))

    return all_sents


def read_doc_from_ontonotes(coref_doc, name_doc):
    assert coref_doc.__class__.__name__ == 'coreference_document', \
        'coref_bank must be a coreference_document instance'
    assert name_doc.__class__.__name__ == 'name_tagged_document', \
        'name_doc must be a name_tagged_document instance'

    doc_id = coref_doc.document_id.split('@')[0]
    assert doc_id == name_doc.document_id.split('@')[0], \
        '{} and {} do not have the same document_id'.format(coref_doc, name_doc)

    print 'Reading document from {}'.format(doc_id)

    conll_file_path = join(ONTONOTES_SOURCE, doc_id + '.depparse')

    all_sents = read_conll_depparse(conll_file_path)

    all_corefs = read_coref_doc(coref_doc)

    doc = Document.construct(all_sents, all_corefs)

    for name_entity in read_name_doc(name_doc):
        add_name_entity_to_doc(doc, name_entity)

    return doc
