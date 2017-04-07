from simple_script import ScriptCorpus
from os import listdir
from os.path import isfile, join
from bz2 import BZ2File
import argparse
from word2vec import Word2VecModel
import utils

parser = argparse.ArgumentParser()
parser.add_argument('input_path', help='directory for ScriptCorpus files')
parser.add_argument('output_path', help='path to write training sequence')
parser.add_argument('word2vec', help='path to word2vec vector file')
parser.add_argument('word2vec_vocab', help='path to word2vec vocab file')
parser.add_argument('--use_lemma', action='store_true',
                    help='if turned on, use the lemma form of a token,'
                         'otherwise use the word form')
parser.add_argument('--include_neg', action='store_true',
                    help='include negation for predicate when applicable')
parser.add_argument('--include_prt', action='store_true',
                    help='include particle for predicate when applicable')
parser.add_argument('--use_entity', action='store_true',
                    help='if turned on, use entity representation for arguments'
                         'with entity_idx, otherwise use token representation')
parser.add_argument('--use_ner', action='store_true',
                    help='if turned on, use ner tag for entities, otherwise use'
                         'head_token of rep_mention')
parser.add_argument('--include_prep', action='store_true',
                    help='include preposition word in pobj representations')

args = parser.parse_args()

writer = utils.GroupedIntListsWriter(args.output_path)

input_files = sorted([join(args.input_path, f) for f in listdir(args.input_path)
                      if isfile(join(args.input_path, f))
                      and f.endswith('.bz2')])

model = Word2VecModel()
model.load_model(args.word2vec, fvocab=args.word2vec_vocab, binary=True)


def get_script_index_list(script, model):
    script.get_all_representations(
        use_lemma=args.use_lemma,
        include_neg=args.include_neg,
        include_prt=args.include_prt,
        use_entity=args.use_entity,
        use_ner=args.use_ner,
        include_prep=args.include_prep
    )
    result = []
    entity_idx_list = [model.get_word_index(
        en.get_representation(use_ner=args.use_ner, use_lemma=args.use_lemma))
        for en in script.entities]
    result.append(entity_idx_list)
    for ev in script.events:
        event_idx_list = [model.get_word_index(word) for word in
                          [ev.pred_text, ev.subj_text,
                           ev.obj_text] + ev.pobj_text_list]
        result.append(event_idx_list)
    return result


for input_f in input_files:
    with BZ2File(input_f, 'r') as fin:
        script_corpus = ScriptCorpus.from_text(fin.read())
        for s in script_corpus.scripts:
            script_index_list = get_script_index_list(s, model)
            writer.write(script_index_list)

writer.close()
