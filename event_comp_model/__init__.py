import platform
import timeit
from os.path import join

from event_composition_model import EventCompositionModel
from event_composition_trainer import EventCompositionTrainer

system_name = platform.system()
dist_name = platform.linux_distribution()[0]

if system_name == 'Darwin':
    root_dir = '/Users/pengxiang/corpora/spaces'
elif system_name == 'Linux':
    if dist_name == 'Ubuntu':
        root_dir = '/scratch/cluster/pxcheng/corpora/enwiki-20160901' \
                   '/event_comp_training/results'
    elif dist_name == 'CentOS':
        root_dir = '/work/03155/pxcheng/maverick/corpora/enwiki-20160901/' \
                   'event_comp_training/results'
else:
    raise RuntimeError('Unrecognized system: {}'.format(system_name))

event_comp_dir_dict = {
    '8M_training_w_salience':
        '20170519/fine_tuning_full/iter_13',
    '40M_training_w_salience':
        '20170530/fine_tuning_full/iter_19',
    '8M_training_wo_salience':
        '20170609/fine_tuning_full/iter_19',
    '40M_training_wo_salience':
        '20170611/fine_tuning_full/iter_19',
    '8M_training_w_salience_wo_first_loc':
        '20170727/fine_tuning_full/iter_19',
    '8M_training_w_salience_wo_head_count':
        '20170729/fine_tuning_full/iter_19',
    '8M_training_w_salience_wo_num_mentions':
        '20170730/fine_tuning_full/iter_19',
    '8M_training_w_salience_only_num_mentions':
        '20170731/fine_tuning_full/iter_19',
    '8M_training_w_salience_only_num_mentions_wo_total':
        '20170801/fine_tuning_full/iter_19',
    '8M_training_w_salience_w_wo_arg':
        '20170901/fine_tuning_full/iter_19',
    '8M_training_w_salience_w_two_args':
        '20170902/fine_tuning_full/iter_14',
    '16M_training_w_salience_mixed_from_stage_2':
        '20170908/fine_tuning_full/iter_14',
    '8M_training_w_salience_w_16M_mixed':
        '20170909/fine_tuning_full/iter_7',
    '8M_training_w_salience_w_10M_mixed_no_wo_arg':
        '20170910/fine_tuning_full/iter_19',
}

default_model_key = '40M_training_w_salience'


def load_event_comp_model(model_key=default_model_key):
    if model_key not in event_comp_dir_dict:
        model_key = default_model_key

    event_comp_dir = join(root_dir, event_comp_dir_dict[model_key])

    print '\nLoading event composition model from {}'.format(event_comp_dir)
    start_time = timeit.default_timer()

    event_comp_model = EventCompositionModel.load_model(event_comp_dir)

    elapsed = timeit.default_timer() - start_time
    print '\tDone in {:.3f} seconds'.format(elapsed)

    return event_comp_model
