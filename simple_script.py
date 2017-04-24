import re
from collections import Counter
from itertools import product
from warnings import warn

import consts
import document
import event_script
import utils

ESCAPE_CHAR_SET = [' // ', '/', ';', ',', ':', '-']
ESCAPE_CHAR_MAP = {
    ' // ': '@slashes@',
    '/': '@slash@',
    ';': '@semicolon@',
    ',': '@comma@',
    ':': '@colon@',
    '-': '@dash@',
    '_': '@underscore@'}


def escape(text, char_set=ESCAPE_CHAR_SET):
    for char in char_set:
        if char in ESCAPE_CHAR_MAP:
            text = text.replace(char, ESCAPE_CHAR_MAP[char])
        else:
            warn('escape rule for {} undefined'.format(char))
    return text


def unescape(text, char_set=ESCAPE_CHAR_SET):
    for char in char_set:
        if char in ESCAPE_CHAR_MAP:
            text = text.replace(ESCAPE_CHAR_MAP[char], char)
        else:
            warn('unescape rule for {} undefined'.format(char))
    return text


class ParseTokenError(Exception):
    pass


class Token(object):
    def __init__(self, word, lemma, pos):
        self.word = word
        self.lemma = lemma
        self.pos = pos

    def __eq__(self, other):
        return self.word == other.word and self.lemma == other.lemma \
               and self.pos == other.pos

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_representation(self, use_lemma=True):
        if use_lemma:
            return self.lemma.lower()
        else:
            return self.word.lower()

    def to_text(self):
        return '{}/{}/{}'.format(
            escape(self.word), escape(self.lemma), escape(self.pos))

    token_re = re.compile(
        r'^(?P<word>[^/]*)/(?P<lemma>[^/]*)/(?P<pos>[^/]*)$')

    @classmethod
    def from_text(cls, text):
        match = cls.token_re.match(text)
        if not match:
            raise ParseTokenError('cannot parse Token from {}'.format(text))
        groups = match.groupdict()
        return cls(
            unescape(groups['word']),
            unescape(groups['lemma']),
            unescape(groups['pos'])
        )

    @classmethod
    def from_token(cls, token):
        if not isinstance(token, document.Token):
            raise ParseTokenError(
                'from_token must be called with a {}.{} instance'.format(
                    document.Token.__module__, document.Token.__name__))
        word = token.word
        lemma = token.lemma
        pos = token.pos
        return cls(word, lemma, pos)


class Argument(Token):
    def __init__(self, word, lemma, pos, entity_idx=-1, mention_idx=-1):
        super(Argument, self).__init__(word, lemma, pos)
        if not (isinstance(entity_idx, int) and entity_idx >= -1):
            raise ParseTokenError(
                'entity_idx must be a natural number or -1 (no entity)')
        self.entity_idx = entity_idx
        if not isinstance(mention_idx, int):
            raise ParseTokenError('mention_idx must be an integer')
        if (not (entity_idx == -1 and mention_idx == -1)) \
                and (not (entity_idx >= 0 and mention_idx >= 0)):
            raise ParseTokenError(
                'mention_idx must be a natural number when entity_idx is set, '
                'or -1 when entity_idx is -1')
        self.mention_idx = mention_idx

    def __eq__(self, other):
        return self.word == other.word and self.lemma == other.lemma \
               and self.pos == other.pos \
               and self.entity_idx == other.entity_idx \
               and self.mention_idx == other.mention_idx

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_representation(self, use_entity=False, entity_list=None,
                           use_ner=True, use_lemma=True):
        assert (not use_entity) or entity_list, \
            'entity_list cannot be None or empty when use_entity is specified'
        if use_entity and self.entity_idx != -1:
            assert 0 <= self.entity_idx < len(entity_list), \
                'entity_idx {} out of range'.format(self.entity_idx)
            assert all(isinstance(entity, Entity) for entity in entity_list), \
                'entity_list must contains only Entity element'
            return entity_list[self.entity_idx].get_representation(
                use_ner, use_lemma)
        else:
            return super(Argument, self).get_representation(use_lemma)

    def get_entity(self, entity_list):
        if self.entity_idx != -1:
            assert self.entity_idx < len(entity_list), \
                'entity_idx {} out of range'.format(self.entity_idx)
            return entity_list[self.entity_idx]
        return None

    def get_mention(self, entity_list):
        entity = self.get_entity(entity_list)
        if entity:
            assert self.mention_idx < len(entity.mentions), \
                'mention_idx {} out of range'.format(self.mention_idx)
            return entity.get_mention(self.mention_idx)
        return None

    def to_text(self):
        text = super(Argument, self).to_text()
        if self.entity_idx != -1 and self.mention_idx != -1:
            text += '//entity-{}-{}'.format(self.entity_idx, self.mention_idx)
        return text

    arg_re = re.compile(
        r'^(?P<word>[^/]*)/(?P<lemma>[^/]*)/(?P<pos>[^/]*)'
        r'((?://entity-)(?P<entity_idx>\d+)(?:-)(?P<mention_idx>\d+))?$')

    @classmethod
    def from_text(cls, text):
        match = cls.arg_re.match(text)
        if not match:
            raise ParseTokenError('cannot parse Argument from {}'.format(text))
        groups = match.groupdict()

        return cls(
            unescape(groups['word']),
            unescape(groups['lemma']),
            unescape(groups['pos']),
            int(groups['entity_idx']) if groups['entity_idx'] else -1,
            int(groups['mention_idx']) if groups['mention_idx'] else -1
        )

    @classmethod
    def from_token(cls, token):
        if not isinstance(token, document.Token):
            raise ParseTokenError(
                'from_token must be called with a {}.{} instance'.format(
                    document.Token.__module__, document.Token.__name__))
        word = token.word
        lemma = token.lemma
        pos = token.pos
        coref_idx = token.coref.idx if token.coref else -1
        mention_idx = token.mention.mention_idx if token.mention else -1
        return cls(word, lemma, pos, coref_idx, mention_idx)


class Predicate(Token):
    def __init__(self, word, lemma, pos, neg=False, prt=''):
        super(Predicate, self).__init__(word, lemma, pos)
        if not isinstance(neg, bool):
            raise ParseTokenError(
                'neg must be a boolean value in initializing a Predicate')
        self.neg = neg
        self.prt = prt

    def __eq__(self, other):
        return self.word == other.word and self.lemma == other.lemma \
               and self.pos == other.pos and self.neg == other.neg \
               and self.prt == other.prt

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_representation(
            self, use_lemma=True, include_neg=True, include_prt=True):
        result = super(Predicate, self).get_representation(use_lemma)
        if include_neg and self.neg:
            result = 'not_' + result
        if include_prt and self.prt:
            result += '_' + self.prt
        return result

    def to_text(self):
        text = super(Predicate, self).to_text()
        if self.neg:
            text = 'not//' + text
        if self.prt != '':
            text += '//' + escape(self.prt)
        return text

    pred_re = re.compile(
        r'^((?P<neg>not)(?://))?(?P<word>[^/]*)/(?P<lemma>[^/]*)/(?P<pos>[^/]*)'
        r'((?://)(?P<prt>[^/]*))?$')

    @classmethod
    def from_text(cls, text):
        match = cls.pred_re.match(text)
        if not match:
            raise ParseTokenError('cannot parse Predicate from {}'.format(text))
        groups = match.groupdict()
        return cls(
            unescape(groups['word']),
            unescape(groups['lemma']),
            unescape(groups['pos']),
            True if groups['neg'] is not None else False,
            unescape(groups['prt']) if groups['prt'] is not None else ''
        )

    @classmethod
    def from_token(cls, token, **kwargs):
        if not isinstance(token, document.Token):
            raise ParseTokenError(
                'from_token must be called with a {}.{} instance'.format(
                    document.Token.__module__, document.Token.__name__))
        if not token.pos.startswith('VB'):
            raise ParseTokenError(
                'from_token cannot be called with a {} token'.format(token.pos))
        if 'neg' not in kwargs or (not isinstance(kwargs['neg'], bool)):
            raise ParseTokenError(
                'a boolean value neg must be passed in as a keyword parameter')
        word = token.word
        lemma = token.lemma
        pos = token.pos
        prt = ''
        if token.compounds:
            if len(token.compounds) > 1:
                warn('Predicate token should contain at most one compound '
                     '(particle), found {}'.format(len(token.compounds)))
            prt = token.compounds[0].lemma
        return cls(word, lemma, pos, kwargs['neg'], prt)


class Event(object):
    def __init__(self, pred, subj, obj, pobj_list):
        if not isinstance(pred, Predicate):
            raise ParseEventError('pred must be a Predicate instance')
        self.pred = pred
        if not (subj is None or isinstance(subj, Argument)):
            raise ParseEventError('subj must be None or an Argument instance')
        self.subj = subj
        if not (obj is None or isinstance(obj, Argument)):
            raise ParseEventError('obj must be None or an Argument instance')
        self.obj = obj
        if not all(prep != '' for prep, _ in pobj_list):
            warn('some of prep(s) in pobj_list are empty')
        if not all(isinstance(pobj, Argument) for _, pobj in pobj_list):
            raise ParseEventError('every pobj must be an Argument instance')
        self.pobj_list = pobj_list
        self.pred_text = ''
        self.subj_text = ''
        self.obj_text = ''
        self.pobj_text_list = []

    def __eq__(self, other):
        return self.pred == other.pred and self.subj == other.subj and \
               self.obj == other.obj and \
               all(prep == other_prep and pobj == other_pobj
                   for (prep, pobj), (other_prep, other_pobj)
                   in zip(self.pobj_list, other.pobj_list))

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_all_args(self):
        all_args = [self.subj, self.obj] + [pobj for _, pobj in self.pobj_list]
        all_args = [arg for arg in all_args if arg is not None]
        return all_args

    def to_text(self):
        return '{} :SUBJ: {} :OBJ: {}{}'.format(
            self.pred.to_text(),
            self.subj.to_text() if self.subj is not None else 'NONE',
            self.obj.to_text() if self.obj is not None else 'NONE',
            ''.join([' :POBJ: {} : {}'.format(prep, pobj.to_text())
                     for prep, pobj in self.pobj_list])
        )

    event_re = re.compile(
        r'^(?P<pred>[^:]*) :SUBJ: (?P<subj>[^:]*) :OBJ: (?P<obj>[^:]*)'
        r'((?: :POBJ: )(?P<prep>[^:]*)(?: : )(?P<pobj>[^:]*))*$')

    @classmethod
    def from_text(cls, text):
        parts = [p for p in re.split(' :(?:SUBJ|OBJ|POBJ): ', text)]
        if len(parts) < 3:
            raise ParseEventError('expected at least 3 parts, separated by ::, '
                                  'got {}: {}'.format(len(parts), text))
        pred = Predicate.from_text(parts[0])
        subj = None
        if parts[1] != 'NONE':
            subj = Argument.from_text(parts[1])
        obj = None
        if parts[2] != 'NONE':
            obj = Argument.from_text(parts[2])
        pobj_list = []
        if len(parts) > 3:
            for part in parts[3:]:
                prep, pobj = part.split(' : ')
                if prep != '':
                    pobj_list.append((prep, Argument.from_text(pobj)))
        return cls(pred, subj, obj, pobj_list)

    @classmethod
    def from_event(cls, ev):
        if not isinstance(ev, event_script.Event):
            raise ParseEventError(
                'from_event must be called with a {}.{} instance'.format(
                    event_script.Event.__module__, event_script.Event.__name__))
        return cls.from_tokens(ev.pred, ev.neg, ev.subj, ev.obj, ev.pobj_list)

    @classmethod
    def from_tokens(cls, pred_token, neg, subj_token, obj_token,
                    pobj_token_list):
        pred = Predicate.from_token(pred_token, neg=neg)
        subj = Argument.from_token(subj_token) \
            if subj_token is not None else None
        obj = Argument.from_token(obj_token) if obj_token is not None else None
        pobj_list = [(prep, Argument.from_token(pobj)) for prep, pobj
                     in pobj_token_list]
        return cls(pred, subj, obj, pobj_list)


class ParseEventError(Exception):
    pass


class Mention(object):
    def __init__(self, sent_idx, start_token_idx, end_token_idx, head_token_idx,
                 rep, tokens, ner=''):
        self.sent_idx = sent_idx
        if not (start_token_idx <= head_token_idx < end_token_idx):
            raise ParseMentionError(
                'head_token_idx {} must be between start_token_idx {} '
                'and end_token_idx {}'.format(
                    head_token_idx, start_token_idx, end_token_idx
                ))
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.head_token_idx = head_token_idx
        if not isinstance(rep, bool):
            raise ParseMentionError('rep must be a boolean value')
        self.rep = rep
        if not tokens:
            raise ParseMentionError('must provide at least one token')
        if not all(isinstance(token, Token) for token in tokens):
            raise ParseMentionError(
                'every token must be a Token instance, found {}'.format(
                    [type(token) for token in tokens]))
        self.tokens = tokens
        self.head_token = \
            self.tokens[self.head_token_idx - self.start_token_idx]
        if not (ner == '' or ner in consts.VALID_NER_TAGS):
            raise ParseMentionError('ner {} is not a valid ner tag'.format(ner))
        self.ner = ner

    def __eq__(self, other):
        return self.sent_idx == other.sent_idx \
               and self.start_token_idx == other.start_token_idx \
               and self.end_token_idx == other.end_token_idx \
               and self.head_token_idx == other.head_token_idx \
               and self.rep == other.rep and self.ner == other.ner \
               and all(token == other_token for token, other_token
                       in zip(self.tokens, other.tokens)) \
               and self.ner == other.ner

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_head_token(self):
        return self.head_token

    def get_ner(self):
        return self.ner

    def get_representation(self, use_ner=True, use_lemma=True):
        if use_ner and self.ner != '':
            return self.ner
        else:
            return self.head_token.get_representation(use_lemma)

    def to_text(self):
        return '{}:{}:{}:{}:{}:{}:{}'.format(
            self.sent_idx,
            self.start_token_idx,
            self.end_token_idx,
            self.head_token_idx,
            1 if self.rep else 0,
            self.ner if self.ner != '' else 'NONE',
            ':'.join([token.to_text() for token in self.tokens])
        )

    @classmethod
    def from_text(cls, text):
        parts = [p.strip() for p in text.split(':')]
        if len(parts) < 7:
            raise ParseMentionError(
                'expected at least 7 parts, separated by ;, got {}: {}'.format(
                    len(parts), text))
        sent_idx = int(parts[0])
        start_token_idx = int(parts[1])
        end_token_idx = int(parts[2])
        head_token_idx = int(parts[3])
        rep = True if int(parts[4]) == 1 else False
        ner = parts[5] if parts[5] != 'NONE' else ''
        tokens = [Token.from_text(token_text.strip()) for token_text
                  in parts[6:]]
        return cls(sent_idx, start_token_idx, end_token_idx, head_token_idx,
                   rep, tokens, ner)

    @classmethod
    def from_mention(cls, mention):
        if not isinstance(mention, document.Mention):
            raise ParseMentionError(
                'from_mention must be called with a {}.{} instance'.format(
                    document.Mention.__module__, document.Mention.__name__))
        # FIXME: use ner of head token as ner for the mention, might be wrong
        ner = mention.head_token.ner
        if ner not in consts.VALID_NER_TAGS:
            ner = ''
        return cls(
            mention.sent_idx,
            mention.start_token_idx,
            mention.end_token_idx,
            mention.head_token_idx,
            mention.rep,
            [Token.from_token(token) for token in mention.tokens],
            ner
        )


class ParseMentionError(Exception):
    pass


class Entity(object):
    def __init__(self, mentions):
        if not mentions:
            raise ParseEntityError('must provide at least one mention')
        if not all(isinstance(mention, Mention) for mention in mentions):
            raise ParseEntityError('every mention must be a Mention instance')
        self.mentions = mentions
        self.rep_mention = None
        for mention in self.mentions:
            if mention.rep:
                if self.rep_mention is None:
                    self.rep_mention = mention
                else:
                    raise ParseEntityError(
                        'cannot have more than one representative mentions')
        if self.rep_mention is None:
            raise ParseEntityError('no representative mention provided')
        ner_counter = Counter()
        for mention in self.mentions:
            if mention.ner != '':
                ner_counter[mention.ner] += 1
        if len(ner_counter):
            self.ner = ner_counter.most_common(1)[0][0]
        else:
            self.ner = ''

    def __eq__(self, other):
        return all(mention == other_mention for mention, other_mention
                   in zip(self.mentions, other.mentions))

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_rep_mention(self):
        return self.rep_mention

    def get_representation(self, use_ner=True, use_lemma=True):
        if use_ner and self.ner != '':
            return self.ner
        return self.get_rep_mention().get_representation(use_ner, use_lemma)

    def to_text(self):
        return ' :: '.join([mention.to_text() for mention in self.mentions])

    @classmethod
    def from_text(cls, text):
        return cls([Mention.from_text(mention_text.strip())
                    for mention_text in text.split(' :: ')])

    @classmethod
    def from_coref(cls, coref):
        if not isinstance(coref, document.Coreference):
            raise ParseEntityError(
                'from_coref must be called with a {}.{}.instance'.format(
                    document.Coreference.__module__,
                    document.Coreference.__name__))
        return cls([Mention.from_mention(mention)
                    for mention in coref.mentions])


class ParseEntityError(Exception):
    pass


class Script(object):
    def __init__(self, doc_name, entities, events):
        self.doc_name = doc_name
        if not all(isinstance(entity, Entity) for entity in entities):
            raise ParseScriptError('every entity must be a Entity instance')
        self.entities = entities
        if not all(isinstance(ev, Event) for ev in events):
            raise ParseScriptError('every event must be a Event instance')
        self.events = events

    def __eq__(self, other):
        return self.doc_name == other.doc_name \
               and all(en == other_en for en, other_en
                       in zip(self.entities, other.entities)) \
               and all(ev == other_ev for ev, other_ev
                       in zip(self.events, other.events))

    def __ne__(self, other):
        return not self.__eq__(other)

    def has_entities(self):
        return len(self.entities) > 0

    def has_events(self):
        return len(self.events) > 0

    def check_entity_idx_range(self):
        for ev in self.events:
            for arg in ev.get_all_args():
                if arg.entity_idx != -1 and arg.mention_idx != -1:
                    assert 0 < arg.entity_idx < len(self.entities), \
                        '{} in {} has entity_idx {} out of range'.format(
                            arg.to_text(), ev.to_text(), arg.entity_idx)
                    assert 0 < arg.mention_idx < len(
                        self.entities[arg.entity_idx].mentions), \
                        '{} in {} has mention_idx {} out of range'.format(
                            arg.to_text(), ev.to_text(), arg.mention_idx)

    def to_text(self):
        entities_text = '\n'.join(['entity-{:0>3d}\t{}'.format(
            en_idx, en.to_text()) for en_idx, en in enumerate(self.entities)])
        events_text = '\n'.join(['event-{:0>4d}\t{}'.format(
            ev_idx, ev.to_text()) for ev_idx, ev in enumerate(self.events)])
        return '{}\n\nEntities:\n{}\n\nEvents:\n{}\n'.format(
            self.doc_name, entities_text, events_text)

    # entity_line_re = re.compile(r'^entity-(?P<en_idx>[\d]{3})\t(?P<en>.*)$')
    # event_line_re = re.compile(r'^event-(?P<ev_idx>[\d]{4})\t(?P<ev>.*)$')

    @classmethod
    def from_text(cls, text):
        if len(text.splitlines()) == 1:
            # Empty document: error in event extraction
            return cls(text.strip(), [], [])

        sections = utils.split_sections(
            (l.strip() for l in text.splitlines()), ["Entities:", "Events:"])
        # First comes the doc name
        doc_name = sections[0][0].strip()
        # Then a whole section giving the entities, matched by entity_line_re
        entity_lines = [line.partition('\t')[2] for line in sections[1] if line]
        entities = [Entity.from_text(line) for line in entity_lines]
        # Then another section giving the events, matched by event_line_re
        event_lines = [line.partition('\t')[2] for line in sections[2] if line]
        events = [Event.from_text(line) for line in event_lines]

        return cls(doc_name, entities, events)

    @classmethod
    def from_script(cls, ev_script):
        if not isinstance(ev_script, event_script.EventScript):
            raise ParseEventError(
                'from_script must be called with a {}.{} instance'.format(
                    event_script.EventScript.__module__,
                    event_script.EventScript.__name__))
        if not ev_script.events:
            warn('EventScript {} has no events'.format(ev_script.doc_name))
        if not ev_script.corefs:
            warn('EventScript {} has no corefs'.format(ev_script.doc_name))
        return cls(
            ev_script.doc_name,
            [Entity.from_coref(coref) for coref in ev_script.corefs],
            [Event.from_event(ev) for ev in ev_script.events]
        )

    @classmethod
    def from_doc(cls, doc):
        if not isinstance(doc, document.Document):
            raise ParseEventError(
                'from_doc must be called with a {}.{} instance'.format(
                    document.Document.__module__, document.Document.__name__))
        # get all events from document
        events = []
        # iterate through all sentences
        for sent in doc.sents:
            # iterate through all tokens
            for pred_token in sent.tokens:
                if pred_token.pos.startswith('VB'):
                    # exclude "be" verbs
                    if pred_token.lemma == 'be':
                        continue
                    # exclude stop verbs
                    if pred_token.lemma in consts.STOP_VERBS:
                        continue
                    # TODO: exclude verbs in quotes
                    # exclude modifying verbs
                    if sent.dep_graph.lookup_label(
                            'gov', pred_token.token_idx, 'xcomp'):
                        continue

                    neg = False
                    if sent.dep_graph.lookup_label('gov', pred_token.token_idx,
                                                   'neg'):
                        neg = True

                    subj_list = sent.get_subj_list(pred_token.token_idx)
                    obj_list = sent.get_obj_list(pred_token.token_idx)
                    pobj_list = sent.get_pobj_list(pred_token.token_idx)

                    if (not subj_list) and (not obj_list):
                        continue
                    if not subj_list:
                        subj_list.append(None)
                    if not obj_list:
                        obj_list.append(None)

                    for arg_tuple in product(subj_list, obj_list):
                        events.append(Event.from_tokens(
                            pred_token, neg, arg_tuple[0], arg_tuple[1],
                            pobj_list))
        if not events:
            warn('doc {} has no events'.format(doc.doc_name))
        if not doc.corefs:
            warn('doc {} has no corefs'.format(doc.doc_name))
        # get all entities from document
        entities = [Entity.from_coref(coref) for coref in doc.corefs]
        return cls(doc.doc_name, entities, events)


class ParseScriptError(Exception):
    pass


class ScriptCorpus(object):
    def __init__(self):
        self.scripts = []

    def __eq__(self, other):
        return all(script == other_script for script, other_script
                   in zip(self.scripts, other.scripts))

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def num_scripts(self):
        return len(self.scripts)

    def add_script(self, script):
        self.scripts.append(script)

    def to_text(self):
        return '\n###DOC###\n'.join(script.to_text() for script in self.scripts)

    @classmethod
    def from_text(cls, text):
        script_corpus = cls()
        for script_text in text.split('\n###DOC###\n'):
            script_corpus.add_script(Script.from_text(script_text))
        return script_corpus
