from sentence import Sentence
from coreference import Coreference
from util import get_class_name


class Document(object):
    def __init__(self, doc_name):
        self.doc_name = doc_name
        self.sents = []
        self.corefs = []

    def add_sent(self, sent):
        assert isinstance(sent, Sentence), \
            'add_sent must be called with a {} instance'.format(
                get_class_name(Sentence))
        sent.build_dep_graph()
        sent.process_verb_prt()
        self.sents.append(sent)

    def add_coref(self, coref):
        assert isinstance(coref, Coreference), \
            'add_coref must be called with a {} instance'.format(
                get_class_name(Coreference))
        self.corefs.append(coref)

    def get_sent(self, idx):
        assert 0 <= idx < len(self.sents), \
            '{} out of sentence index'.format(idx)
        result = self.sents[idx]
        assert isinstance(result, Sentence), \
            'return value of get_sent must be a {} instance'.format(
                get_class_name(Sentence))
        return result

    def get_coref(self, idx):
        assert 0 <= idx < len(self.corefs), \
            '{} out of coreference index'.format(idx)
        result = self.corefs[idx]
        assert isinstance(result, Coreference), \
            'return value of get_coref must be a {} instance'.format(
                get_class_name(Coreference))
        return result

    def get_token(self, sent_idx, token_idx):
        return self.get_sent(sent_idx).get_token(token_idx)

    def get_mention(self, coref_idx, mention_idx):
        return self.get_coref(coref_idx).get_mention(mention_idx)

    def __str__(self):
        result = '\t\t#SENT#\t\t'.join([str(sent) for sent in self.sents])
        result += '\t\t\t#DOC#\t\t'
        result += '\t\t#COREF#\t\t'.join([str(coref) for coref in self.corefs])
        return result

    def pretty_print(self):
        result = '\n'.join(
            ['Sent #{}\n\t'.format(sent_idx) + sent.pretty_print()
             for sent_idx, sent in enumerate(self.sents)])
        result += '\nCoreferences:\n'
        result += '\n'.join(['\t' + str(coref) for coref in self.corefs])
        return result

    def to_plain_text(self):
        return '\n'.join([sent.to_plain_text() for sent in self.sents])

    def preprocessing(self):
        for coref in self.corefs:
            for mention in coref.mentions:
                sent = self.get_sent(mention.sent_idx)
                # set mention.head_token_idx if it is -1 (unset)
                if mention.head_token_idx == -1:
                    mention.set_head_token_idx(sent.dep_graph)
                # set mention.tokens and mention.head_token
                mention.add_token_info(sent.tokens)
                # set the pointers to coref and mention of the head token
                mention.head_token.add_coref_info(coref, mention)
            # set rep_mention if it is None
            if coref.rep_mention is None:
                coref.find_rep_mention()

    @classmethod
    def construct(cls, doc_name, all_sents, all_corefs):
        doc = cls(doc_name)
        for sent in all_sents:
            doc.add_sent(sent)
        for coref in all_corefs:
            doc.add_coref(coref)
        doc.preprocessing()
        return doc
