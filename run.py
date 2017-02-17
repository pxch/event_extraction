from document_reader import read_doc
from bz2 import BZ2File
import sys
from os import listdir, mkdir
from os.path import isfile, isdir, join
import random
from embedding import Embedding
from event_script import EvalStats
import argparse

'''
def print_accuracy(num_all, num_correct):
  print '\t{} correct in {}'.format(num_correct, num_all)
  print '\tAccuracy: {}\n'.format(float(num_correct)/num_all \
      if num_all != 0 else 0)
'''

def eval_most_freq_coref(all_docs):
  print 'Evaluation based on most frequent coreference chain:'
  eval_stats = EvalStats()
  for doc in all_docs:
    eval_stats.add_eval_stats(doc.eval_most_freq_coref())
  eval_stats.pretty_print()
  '''
  total_num_all = 0
  total_num_correct = 0
  for doc in all_docs:
    num_all, num_correct = doc.eval_most_freq_coref()
    total_num_all += num_all
    total_num_correct += num_correct
  print_accuracy(total_num_all, total_num_correct)
  '''

def eval_most_sim_arg(all_docs, model, use_other_args=False, \
    syntax_suffix=False, head_only=False, rep_only=False):
  print 'Evaluation based on most similar argument, ' + \
      'use_other_args = {}, head_only = {}, rep_only = {}:'.format(
          use_other_args, head_only, rep_only)
  eval_stats = EvalStats()
  for doc in all_docs:
    eval_stats.add_eval_stats(doc.eval_most_sim_arg(model, use_other_args, \
        syntax_suffix, head_only, rep_only))
  eval_stats.pretty_print()
  '''
  total_num_all = 0
  total_num_correct = 0
  for doc in all_docs:
    num_all, num_correct = doc.eval_most_sim_arg(model, use_other_args, \
        syntax_suffix, head_only, rep_only)
    total_num_all += num_all
    total_num_correct += num_correct
  print_accuracy(total_num_all, total_num_correct)
  '''

def eval_most_sim_event(all_docs, model, use_max_score=False, \
    syntax_suffix=False, head_only=False, rep_only=False):
  print 'Evaluation based on most similar event, ' + \
      'use_max_score = {}, head_only = {}, rep_only = {}:'.format(
          use_max_score, head_only, rep_only)
  eval_stats = EvalStats()
  for doc in all_docs:
    eval_stats.add_eval_stats(doc.eval_most_sim_event(model, use_max_score, \
        syntax_suffix, head_only, rep_only))
  eval_stats.pretty_print()
  '''
  total_num_all = 0
  total_num_correct = 0
  for doc in all_docs:
    num_all, num_correct = doc.eval_most_sim_event(model, use_max_score, \
        syntax_suffix, head_only, rep_only)
    total_num_all += num_all
    total_num_correct += num_correct
  print_accuracy(total_num_all, total_num_correct)
  '''

parser = argparse.ArgumentParser()

parser.add_argument("model_path", help="path to embedding model file")
parser.add_argument("list_of_filenames", \
    help="file containing list of all filenames to be evaluated, " + \
    "one filename each line")
parser.add_argument("input_dir", \
    help="directory containing input files to be evaluates")

parser.add_argument("-b", "--binary_model", \
    help="whether the model file is binary", action="store_true")
parser.add_argument("-s", "--syntax_model", \
    help="whether the model includes syntax labels with word forms", \
    action="store_true")
parser.add_argument("-w", "--write_output_scripts", \
    help="whether or not to write output scripts", action="store_true")

parser.add_argument("-o", "--output_dir", help="directory to write output scripts")

args = parser.parse_args()

assert (not args.write_output_scripts) or args.output_dir, \
    'Must set output_dir when write_output_scripts is set'

model = Embedding()
model.load_model(args.model_path, zipped=args.binary_model)

syntax_model = args.syntax_model

list_of_filenames = [line.strip() for line in open(args.list_of_filenames, 'r').readlines()]
input_dir = args.input_dir

'''
model_path = sys.argv[1]
model_zipped = False
if model_path.endswith('gz'):
  model_zipped = True
model = Embedding()
model.load_model(model_path, zipped=model_zipped)

syntax_model = True

list_of_filenames = [line.strip() for line in open(sys.argv[2], 'r').readlines()]
input_dir = sys.argv[3]
'''

if args.write_output_scripts:
  output_dir = args.output_dir
  if not isdir(output_dir):
    mkdir(output_dir)

all_docs = []
print 'Reading all documents from: {}'.format(input_dir)
for filename in list_of_filenames:
  file_path = join(input_dir, filename)
  #print 'Processing file: ' + file_path
  doc = read_doc(BZ2File(file_path, 'r'))
  doc.preprocessing()
  doc.extract_event_script()
  all_docs.append(doc)

  if args.write_output_scripts:
    output_path = join(output_dir, filename[:-8] + '.script')

    fout = open(output_path, 'w')
    fout.write(doc.to_plain_text() + '\n')
    fout.write('\nDependency parse and coreference resolution:\n')
    fout.write(doc.pretty_print() + '\n')
    fout.write('\nEvent script:\n')
    fout.write(doc.script.pretty_print() + '\n')
    fout.close()

print 'Done\n'

eval_most_freq_coref(all_docs)

eval_most_sim_arg(all_docs, model, use_other_args=False, syntax_suffix=syntax_model, head_only=False, rep_only=False)
eval_most_sim_arg(all_docs, model, use_other_args=True, syntax_suffix=syntax_model, head_only=False, rep_only=False)

eval_most_sim_event(all_docs, model, use_max_score=False, syntax_suffix=syntax_model, head_only=False, rep_only=False)
eval_most_sim_event(all_docs, model, use_max_score=True, syntax_suffix=syntax_model, head_only=False, rep_only=False)

'''
eval_most_sim_arg(all_docs, model, use_other_args=False, syntax_suffix=syntax_model, head_only=False, rep_only=True)
eval_most_sim_arg(all_docs, model, use_other_args=True, syntax_suffix=syntax_model, head_only=False, rep_only=True)

eval_most_sim_event(all_docs, model, use_max_score=False, syntax_suffix=syntax_model, head_only=False, rep_only=True)
eval_most_sim_event(all_docs, model, use_max_score=True, syntax_suffix=syntax_model, head_only=False, rep_only=True)
'''

