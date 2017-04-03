from corenlp_reader import read_doc_from_corenlp
from simple_script import Script, ScriptCorpus
import sys
from os import listdir
from os.path import isdir, isfile, join
from bz2 import BZ2File

input_path = sys.argv[1]
output_path = sys.argv[2]

fout = BZ2File(output_path, 'w')

input_files = sorted([join(input_path, f) for f in listdir(input_path)
                      if isfile(join(input_path, f)) and f.endswith('xml.bz2')])

script_corpus = ScriptCorpus()

for input_f in input_files:
    with BZ2File(input_f, 'r') as fin:
        doc = read_doc_from_corenlp(fin)
        script = Script.from_doc(doc)
        script_corpus.add_script(script)

fout.write(script_corpus.to_text())
fout.close()
