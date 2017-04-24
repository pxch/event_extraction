from warnings import warn

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

