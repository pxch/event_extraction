from collections import defaultdict

from node import Node


def parse_pointer(ptr, prefix):
    return Node.parse(','.join([prefix + n for n in ptr.split(',')]))


class PropBankInstance(object):
    def __init__(self, file_id, sent_id, token_id, base_form, sense_num):
        self.file_id = file_id
        self.sent_id = sent_id
        self.token_id = token_id
        self.base_form = base_form
        self.sense_num = sense_num
        self.arguments = []

    def __str__(self):
        return ' '.join([
            self.file_id,
            str(self.sent_id),
            str(self.token_id),
            self.base_form,
            self.sense_num,
            ' '.join([label + '-' + str(node) for label, node
                      in self.arguments])])

    @classmethod
    def parse_nombank(cls, s):
        items = s.split(' ')
        file_id = items[0].split('/')[-1].split('.')[0]
        sent_id = int(items[1])
        token_id = int(items[2])
        base_form = items[3]
        sense_num = items[4]
        instance = cls(file_id, sent_id, token_id, base_form, sense_num)
        node_prefix = str(file_id) + ':' + str(sent_id) + ':'
        for item in items[5:]:
            hyphen_idx = item.find('-')
            pointer = item[:hyphen_idx]
            label = item[hyphen_idx + 1:].strip().lower()
            # if len(pointer.split('*')) > 1:
            #     continue
            # else:
            #     instance.arguments.append(
            #         (label, parse_pointer(pointer, node_prefix)))
            for ptr in pointer.split('*'):
                instance.arguments.append(
                    (label, parse_pointer(ptr, node_prefix)))
        return instance

    @classmethod
    def parse_propbank(cls, s):
        items = s.split(' ')
        file_id = items[0].split('/')[-1].split('.')[0]
        sent_id = int(items[1])
        token_id = int(items[2])
        base_form, sense_num = items[4].split('.')
        instance = cls(file_id, sent_id, token_id, base_form, sense_num)
        node_prefix = str(file_id) + ':' + str(sent_id) + ':'
        for item in items[6:]:
            hyphen_idx = item.find('-')
            pointer = item[:hyphen_idx]
            label = item[hyphen_idx + 1:].strip().lower()
            # if len(pointer.split('*')) > 1:
            #     continue
            # else:
            #     instance.arguments.append(
            #         (label, parse_pointer(pointer, node_prefix)))
            for ptr in pointer.split('*'):
                instance.arguments.append(
                    (label, parse_pointer(ptr, node_prefix)))
        return instance


class PropBankReader(object):
    def __init__(self):
        self.all_instances = []
        self.instances_by_file = defaultdict(list)

    def read_nombank(self, path):
        print '\nread NomBank file from {}'.format(path)
        fin = open(path, 'r')
        cnt = 0
        for line in fin:
            cnt += 1
            instance = PropBankInstance.parse_nombank(line)
            self.all_instances.append(instance)
            self.instances_by_file[instance.file_id].append(instance)
        print 'found {} instances\n'.format(cnt)

    def read_propbank(self, path):
        print '\nread PropBank file from {}'.format(path)
        fin = open(path, 'r')
        cnt = 0
        for line in fin:
            cnt += 1
            instance = PropBankInstance.parse_propbank(line)
            self.all_instances.append(instance)
            self.instances_by_file[instance.file_id].append(instance)
        print 'found {} instances\n'.format(cnt)

    def search_by_file(self, file_id):
        if file_id in self.instances_by_file:
            return self.instances_by_file[file_id]
        else:
            return []
