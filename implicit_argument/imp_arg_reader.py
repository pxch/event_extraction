from copy import deepcopy

from lxml import etree

from node import Node
from predicate import Predicate


class ImplicitArgumentTarget(object):
    def __init__(self):
        self.annotations = []
        self.predicate = {}
        self.argument = {}
        self.attribute_data = False

    def start(self, tag, attrib):
        if tag == 'annotations':
            if attrib:
                self.predicate['node'] = attrib['for_node']
                self.predicate['arguments'] = []
        if tag == 'annotation':
            self.argument['node'] = attrib['node']
            self.argument['label'] = attrib['value']
            self.argument['attribute'] = ''
        if tag == 'attribute':
            self.attribute_data = True

    def end(self, tag):
        if tag == 'annotations':
            if self.predicate:
                self.annotations.append(deepcopy(self.predicate))
            self.predicate = {}
        if tag == 'annotation':
            self.predicate['arguments'].append(deepcopy(self.argument))
            self.argument = {}
        if tag == 'attribute':
            self.attribute_data = False

    def data(self, data):
        if self.attribute_data:
            self.argument['attribute'] = data

    def close(self):
        return 'success'


def read_imp_arg_dataset(input_xml):
    xml_parser = etree.XMLParser(target=ImplicitArgumentTarget())
    print '\nReading implicit argument dataset from: ' + input_xml + '\n'
    etree.parse(input_xml, xml_parser)

    annotations = []
    for annotation in xml_parser.target.annotations:
        pred_node = Node.parse(annotation['node'])
        predicate = Predicate(pred_node)
        predicate.add_args_from_ia(annotation['arguments'])
        annotations.append(predicate)

    annotations.sort(key=lambda p: p.node.file_id)

    return annotations
