from warnings import warn

import numpy as np

import consts


def escape(text, char_set=consts.ESCAPE_CHAR_SET):
    for char in char_set:
        if char in consts.ESCAPE_CHAR_MAP:
            text = text.replace(char, consts.ESCAPE_CHAR_MAP[char])
        else:
            warn('escape rule for {} undefined'.format(char))
    return text


def unescape(text, char_set=consts.ESCAPE_CHAR_SET):
    for char in char_set:
        if char in consts.ESCAPE_CHAR_MAP:
            text = text.replace(consts.ESCAPE_CHAR_MAP[char], char)
        else:
            warn('unescape rule for {} undefined'.format(char))
    return text


def get_class_name(class_type):
    return '{}.{}'.format(class_type.__module__, class_type.__name__)


def cos_sim(vec1, vec2):
    if np.count_nonzero(vec1) == 0 or np.count_nonzero(vec2) == 0:
        return 0.0
    return vec1.dot(vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
