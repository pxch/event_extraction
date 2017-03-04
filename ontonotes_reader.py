from copy import deepcopy
from coref import *
from dep import *
from sent import *
from tok import *

from os.path import join
import on


def load_ontonotes(cfg_file):
    all_subcorps = []
    cfg = on.common.util.load_config(cfg_file)
    a_ontonotes = on.ontonotes(cfg)
    for subcorp in a_ontonotes:
        all_subcorps.append(subcorp)
    return all_subcorps


def read_coref_link(coref_link):
    sent_idx = coref_link.sentence_index
    start_token_idx = coref_link.start_word_index
    end_token_idx = coref_link.end_word_index
    # find head token index in Doc.fix_ontonotes_coref_info and Sent.get_head_token_idx
    head_token_idx = -1
    # find rep mention in Doc.fix_ontonotes_coref_info
    rep = False
    if coref_link.subtree is not None:
        text = coref_link.subtree.get_trace_adjusted_word_string()
    else:
        print 'Warning: coref mention\n\t{}\nhas no subtree, use coreference_link.string for text.'.format(coref_link)
        text = coref_link.string
    return Mention(sent_idx, start_token_idx, end_token_idx, head_token_idx, rep, text)


def read_coref_chain(coref_idx, coref_chain):
    coref = Coref(coref_idx)
    for coref_link in coref_chain:
        coref.add_mention(read_coref_link(coref_link))
    return coref


def read_coref_bank(coref_bank):
    all_corefs = []
    coref_idx = 0
    for coref_chain in coref_bank:
        coref = read_coref_chain(coref_idx, coref_chain)
        all_corefs.append(deepcopy(coref))
        coref_idx += 1
    return all_corefs


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
                print 'Warning: Line #{}\n\t{}\nhas duplicated token index, neglected.'.format(line_idx, line.strip())
                continue
            word = items[1]
            lemma = items[2]
            pos = items[4]
            # TODO: add ner info from *.name files
            ner = ''
            sent.add_token(Token(word, lemma, pos, ner))
            try:
                head_idx = int(items[6])
            except ValueError:
                continue
            dep_label = items[7]
            if dep_label != 'root':
                sent.add_dep(Dep(dep_label, head_idx - 1, token_idx - 1, extra=False))
            if items[8] != '_':
                for e_dep in items[8].strip().split('|'):
                    try:
                        e_dep_head_idx = int(e_dep.split(':')[0])
                    except ValueError:
                        continue
                    e_dep_label = ':'.join(e_dep.split(':')[1:])
                    sent.add_dep(Dep(e_dep_label, e_dep_head_idx - 1, token_idx - 1, extra=True))

    return all_sents
