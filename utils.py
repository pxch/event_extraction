import logging
from os import listdir
from os.path import isdir, isfile, join
from bz2 import BZ2File
from rich_script import SingleTrainingInput, PairTrainingInput
import numpy
from math import ceil

def get_console_logger(name, debug=False):
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Prepare a logger
    log = logging.getLogger(name)
    log.setLevel(level)

    if not log.handlers:
        # Just log to the console
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        # Put a timestamp on everything
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh.setFormatter(formatter)

        log.addHandler(sh)

    return log


def split_sections(input_iter, section_heads):
    """
    Divide up the lines of a file by searching for the given section heads
    (whole lines) in the order they're given.

    A list of the lines of each section is returned in a list, in the same 
    order as the given section head list.

    If the given heads are not all found, None values are returned in place of
    those sections (which will be at the end of the list).
    The number of returned sections will always be len(section_heads)+1 -- 
    an extra one for the text before the first head.

    Note that, although this is designed for use with lines of text, there's 
    nothing about it specific to text: the objects in the list (and section
    head list) could be of any type.

    """
    input_iter = iter(input_iter)
    section_heads = iter(section_heads)
    next_head = section_heads.next()
    sections = [[]]

    try:
        for line in input_iter:
            if line == next_head:
                # Move onto the next section
                sections.append([])
                next_head = section_heads.next()
            else:
                # Add this line to the current section
                sections[-1].append(line)
    except StopIteration:
        # Reached the end of the list of section names: include the remainder of
        # the input in the last section
        sections[-1].extend(list(input_iter))

    # Pad out the sections if there are heads we didn't use
    remaining_heads = list(input_iter)
    if remaining_heads:
        sections.extend([None] * len(remaining_heads))

    return sections


class IntListsWriter(object):
    def __init__(self, filename):
        self.filename = filename
        self.writer = open(filename, 'w')

    def write(self, lst):
        self.writer.write("%s\n" % ",".join(str(num) for num in lst))

    def close(self):
        self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self


class IntListsReader(object):
    def __init__(self, filename):
        self.filename = filename

    def lists(self):
        with open(self.filename, 'r') as reader:
            for line in reader:
                # Remove the line break at the end
                line = line[:-1]
                # Catch the empty case
                if line:
                    yield [int(val) if val != "None" else None for val in
                           line.split(",")]
                else:
                    yield []

    def __iter__(self):
        return self.lists()


class GroupedIntListsWriter(object):
    def __init__(self, filename):
        self.filename = filename
        self.writer = open(filename, 'w')

    def write(self, lsts):
        self.writer.write("%s\n" % " / ".join(
            ",".join(str(num) for num in lst) for lst in lsts))

    def close(self):
        self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self


class GroupedIntListsReader(object):
    def __init__(self, filename):
        self.filename = filename
        self._length = None

    def __iter__(self):
        with open(self.filename, 'r') as reader:
            for line in reader:
                # Remove the line break at the end
                line = line.strip("\n ")
                # Catch the empty case
                if line:
                    yield [[int(val) if val != "None" else None for val in
                            lst.split(",") if len(val)]
                           for lst in line.split(" / ")]
                else:
                    yield []

    def __len__(self):
        if self._length is None:
            with open(self.filename, 'r') as reader:
                self._length = sum(1 for __ in reader)
        return self._length


class GroupedIntTuplesReader(GroupedIntListsReader):
    def __iter__(self):
        for grp in GroupedIntListsReader.__iter__(self):
            yield [tuple(lst) for lst in grp]


class IndexedCorpusReader(object):
    def __init__(self, corpus_type, corpus_dir):
        assert corpus_type in ['pretraining', 'pair_tuning'], \
            'corpus_type can only be pretraining on pair_tuning'
        self.corpus_type = corpus_type
        if corpus_type == 'pretraining':
            self.from_text_fn = SingleTrainingInput.from_text
        else:
            self.from_text_fn = PairTrainingInput.from_text
        assert isdir(corpus_dir), '{} is not a directory'.format(corpus_dir)
        self.corpus_dir = corpus_dir
        try:
            self.length = int(
                open(join(corpus_dir, 'line_count'), 'r').readline().strip())
        except:
            raise IOError('File {}/line_count not found!'.format(corpus_dir))
        self.filenames = sorted(
            [join(corpus_dir, f) for f in listdir(corpus_dir)
             if isfile(join(corpus_dir, f)) and not f.endswith('line_count')])

    def __len__(self):
        return self.length

    def __iter__(self):
        for filename in self.filenames:
            if filename.endswith('bz2'):
                index_file = BZ2File(filename, 'r')
            else:
                index_file = open(filename, 'r')
            for line in index_file.readlines():
                line = line.strip()
                if line:
                    yield self.from_text_fn(line)


class PretrainingCorpusIterator(object):
    def __init__(self, corpus_dir, model, layer_input=-1, batch_size=1):
        self.corpus_dir = corpus_dir
        self.reader = IndexedCorpusReader('pretraining', self.corpus_dir)
        self.model = model
        self.layer_input = layer_input
        self.batch_size = batch_size
        self.num_batch = int(ceil(float(len(self.reader)) / batch_size))
        if layer_input == -1:
            # Compile the expression for the deepest hidden layer
            self.projection_fn = model.projection_model.project
        else:
            # Compile the theano expression for this layer's input
            self.projection_fn = \
                model.projection_model.get_layer_input_function(layer_input)

    def __len__(self):
        return len(self.reader)

    def __iter__(self):
        pred_inputs = numpy.zeros(self.batch_size, dtype=numpy.int32)
        subj_inputs = numpy.zeros(self.batch_size, dtype=numpy.int32)
        obj_inputs = numpy.zeros(self.batch_size, dtype=numpy.int32)
        pobj_inputs = numpy.zeros(self.batch_size, dtype=numpy.int32)

        data_point_index = 0

        for input in self.reader:
            pred_inputs[data_point_index] = input.pred_input
            subj_inputs[data_point_index] = input.subj_input
            obj_inputs[data_point_index] = input.obj_input
            pobj_inputs[data_point_index] = input.pobj_input
            data_point_index += 1

            # If we've filled up the batch, yield it
            if data_point_index == self.batch_size:
                yield self.projection_fn(
                    pred_inputs, subj_inputs, obj_inputs, pobj_inputs)
                data_point_index = 0

        if data_point_index > 0:
            # We've partially filled a batch: yield this as the last item
            yield self.projection_fn(
                pred_inputs, subj_inputs, obj_inputs, pobj_inputs)
