import sys
from bz2 import BZ2File
from os import listdir
from os.path import isfile, join

from dataset import read_doc_from_corenlp
from script import Script, ScriptCorpus

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
        if script.has_events():
            script_corpus.add_script(script)

fout.write(script_corpus.to_text())
fout.close()
