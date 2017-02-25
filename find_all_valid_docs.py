from document_reader import read_doc
from bz2 import BZ2File
from gzip import GzipFile
import sys
from os import listdir, makedirs
from os.path import isfile, isdir, join

input_dir = sys.argv[1]
output_dir = sys.argv[2]

if not isdir(output_dir):
    makedirs(output_dir)

list_of_file_fout = open(join(output_dir, 'list_of_docs'), 'w')

min_num_coref = 3
min_occurrence_count = 3

for input_subdir in listdir(input_dir):
    subdir_path = join(input_dir, input_subdir)
    if isdir(subdir_path):
        print 'Processing subdir: ' + subdir_path
        for input_f in listdir(subdir_path):
            input_path = join(subdir_path, input_f)
            if input_path.endswith('.xml.bz2') and isfile(input_path):
                # print 'Processing file: ' + input_path
                doc = read_doc(BZ2File(input_path, 'r'))
                doc.preprocessing()
                doc.extract_event_script()

                if doc.validate_script(min_num_coref, min_occurrence_count, True):
                    list_of_file_fout.write(input_path + '\n')

                    script_path = join(output_dir, input_f[:-8] + '.script')
                    script_fout = open(script_path, 'w')
                    script_fout.write(doc.to_plain_text() + '\n')
                    script_fout.write(doc.script.pretty_print() + '\n')
                    '''
                    script_fout.write('\nEvent script:\n')
                    script_fout.write(doc.script.pretty_print() + '\n')
                    script_fout.write('\nDependency parse and coreference resolution:\n')
                    script_fout.write(doc.pretty_print() + '\n')
                    '''
                    script_fout.close()

list_of_file_fout.close()
