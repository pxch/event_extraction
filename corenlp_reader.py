from copy import deepcopy
from document import *
from sentence import *
from coreference import *
from lxml import etree
import consts

# dependency_type = 'collapsed-ccprocessed-dependencies'
dependency_type = 'enhanced-plus-plus-dependencies'


class CoreNLPTarget(object):
    def __init__(self):
        self.sents = []
        self.corefs = []
        self.sent = None
        self.coref = None
        self.tag = ''
        self.word = ''
        self.lemma = ''
        self.pos = ''
        self.ner = ''
        self.dep_label = ''
        self.gov_idx = -1
        self.dep_idx = -1
        self.extra = False
        self.sent_idx = -1
        self.start_token_idx = -1
        self.end_token_idx = -1
        self.head_token_idx = -1
        self.rep = False
        self.text = ''
        self.parse_sent = False
        self.parse_dep = False
        self.parse_coref = False

    def start(self, tag, attrib):
        self.tag = tag
        if tag == 'sentences':
            self.parse_sent = True
        elif tag == 'sentence':
            if self.parse_sent:
                self.sent = Sent(int(attrib['id']) - 1)
        elif tag == 'dependencies':
            if attrib['type'] == dependency_type and self.parse_sent:
                self.parse_dep = True
        elif tag == 'dep':
            if self.parse_dep:
                self.dep_label = attrib['type']
                if 'extra' in attrib:
                    self.extra = True
        elif tag == 'governor':
            if self.parse_dep:
                self.gov_idx = int(attrib['idx']) - 1
        elif tag == 'dependent':
            if self.parse_dep:
                self.dep_idx = int(attrib['idx']) - 1
        elif tag == 'coreference':
            if not self.parse_coref:
                self.parse_coref = True
            self.coref = Coref(len(self.corefs))
        elif tag == 'mention':
            if self.parse_coref:
                if 'representative' in attrib:
                    self.rep = True

    def data(self, data):
        data = data.strip()
        if data != '':
            if self.parse_sent:
                if self.tag == 'word':
                    self.word += data
                elif self.tag == 'lemma':
                    self.lemma += data
                elif self.tag == 'POS':
                    self.pos += data
                elif self.tag == 'NER':
                    self.ner += data
            elif self.parse_coref:
                if self.tag == 'sentence':
                    self.sent_idx = int(data) - 1
                elif self.tag == 'start':
                    self.start_token_idx = int(data) - 1
                elif self.tag == 'end':
                    self.end_token_idx = int(data) - 1
                elif self.tag == 'head':
                    self.head_token_idx = int(data) - 1
                elif self.tag == 'text':
                    self.text += data

    def end(self, tag):
        self.tag = ''
        if tag == 'sentences':
            if self.parse_sent:
                self.parse_sent = False
        elif tag == 'sentence':
            if self.parse_sent:
                if self.sent is not None:
                    self.sents.append(deepcopy(self.sent))
                    self.sent = None
        elif tag == 'token':
            token = Token(self.word, self.lemma, self.pos)
            if self.ner != '':
                # map corenlp ner tags to coerse grained ner tags
                token.set_attrib(
                    'ner', consts.convert_corenlp_ner_tag(self.ner))
            self.sent.add_token(deepcopy(token))
            self.word = ''
            self.lemma = ''
            self.pos = ''
            self.ner = ''
        elif tag == 'dependencies':
            if self.parse_dep:
                self.parse_dep = False
        elif tag == 'dep':
            if self.parse_dep:
                if self.dep_label != 'root':
                    dep = Dep(self.dep_label, self.gov_idx,
                              self.dep_idx, self.extra)
                    self.sent.add_dep(deepcopy(dep))
                self.dep_label = ''
                self.gov_idx = -1
                self.dep_idx = -1
                self.extra = False
        elif tag == 'coreference':
            if self.parse_coref:
                if self.coref is not None:
                    self.corefs.append(deepcopy(self.coref))
                    self.coref = None
                else:
                    self.parse_coref = False
        elif tag == 'mention':
            mention = Mention(self.sent_idx, self.start_token_idx,
                              self.end_token_idx)
            mention.set_attrib('head_token_idx', self.head_token_idx)
            mention.set_attrib('rep', self.rep)
            mention.set_attrib('text', self.text)
            self.coref.add_mention(deepcopy(mention))
            self.sent_idx = -1
            self.start_token_idx = -1
            self.end_token_idx = -1
            self.head_token_idx = -1
            self.rep = False
            self.text = ''

    def close(self):
        return 'success'


def read_doc_from_corenlp(input_xml):
    print 'Reading document from {}'.format(input_xml.name)
    xml_parser = etree.XMLParser(target=CoreNLPTarget())
    etree.parse(input_xml, xml_parser)
    doc = Document.construct(xml_parser.target.sents, xml_parser.target.corefs)
    return doc
