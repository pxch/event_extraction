from copy import deepcopy

from consts import core_arg_list, predicate_core_arg_mapping
from predicate import Predicate
from rich_implicit_argument import RichImplicitArgument
from rich_script.indexed_event import IndexedEvent
from rich_script.rich_argument import RichArgument
from rich_script.rich_entity import EntitySalience
from util import get_class_name


class RichPredicate(object):
    def __init__(self, fileid, n_pred, v_pred, exp_args, imp_args,
                 num_candidates):
        self.fileid = fileid
        self.n_pred = n_pred
        self.v_pred = v_pred
        self.pred_wv = -1
        assert all(isinstance(arg, RichArgument) for arg in exp_args)
        self.exp_args = exp_args
        assert all(isinstance(arg, RichImplicitArgument) for arg in imp_args)
        self.imp_args = imp_args

        self.num_candidates = num_candidates

        self.rich_subj = None
        self.rich_obj = None
        self.rich_pobj = None
        for exp_arg in self.exp_args:
            if exp_arg.arg_type == 'SUBJ':
                assert self.rich_subj is None
                self.rich_subj = exp_arg
            elif exp_arg.arg_type == 'OBJ':
                assert self.rich_obj is None
                self.rich_obj = exp_arg
            else:
                assert self.rich_pobj is None
                self.rich_pobj = exp_arg

        self.sum_dice = 0.0
        self.num_gt = 0
        self.num_model = 0

        self.thres = 0.0

    def num_imp_args(self):
        return len([imp_arg for imp_arg in self.imp_args if imp_arg.exist])

    def num_missing_args(self):
        return len(self.imp_args)

    def eval(self, thres, comp_wo_arg=True):
        self.sum_dice = 0.0
        self.num_gt = 0
        self.num_model = 0

        for imp_arg in self.imp_args:
            if imp_arg.has_coherence_score:
                if imp_arg.exist:
                    self.num_gt += 1
                if imp_arg.max_coherence_score >= thres:
                    if (not comp_wo_arg) or imp_arg.max_coherence_score >= \
                            imp_arg.coherence_score_wo_arg:
                        self.num_model += 1
                        self.sum_dice += imp_arg.get_max_dice_score()

        return self.sum_dice, self.num_gt, self.num_model

    def set_pred_wv(self, pred_wv_mapping):
        self.pred_wv = pred_wv_mapping[self.v_pred]

    def get_index(self, model, include_type=True, use_unk=True):
        for exp_arg in self.exp_args:
            exp_arg.get_index(model, include_type=include_type, use_unk=use_unk)
        for imp_arg in self.imp_args:
            imp_arg.get_index(model, include_type=include_type, use_unk=use_unk)

    def get_eval_input_list_all(self, include_salience=True):
        pos_input = IndexedEvent(
            self.pred_wv,
            self.rich_subj.get_pos_wv() if self.rich_subj else -1,
            self.rich_obj.get_pos_wv() if self.rich_obj else -1,
            self.rich_pobj.get_pos_wv() if self.rich_pobj else -1)

        eval_input_list_all = []

        for imp_arg in self.imp_args:
            eval_input_list = []

            arg_idx = imp_arg.get_arg_idx()

            eval_input = deepcopy(pos_input)
            eval_input.set_argument(arg_idx, -1)
            if include_salience:
                eval_input_list.append(
                    (eval_input, EntitySalience(**{})))
            else:
                eval_input_list.append(eval_input)

            for candidate_wv, candidate in zip(
                    imp_arg.candidate_wv_list, imp_arg.rich_candidate_list):
                eval_input = deepcopy(pos_input)
                eval_input.set_argument(arg_idx, candidate_wv)
                if include_salience:
                    eval_input_list.append(
                        (eval_input, candidate.entity_salience))
                else:
                    eval_input_list.append(eval_input)

            eval_input_list_all.append(
                (imp_arg.label, arg_idx, eval_input_list))

        return eval_input_list_all

    @classmethod
    def build(cls, predicate, corenlp_reader, use_lemma=True, use_entity=True,
              use_corenlp_tokens=True):
        assert isinstance(predicate, Predicate), \
            'RichPredicate must be built from a {} instance'.format(
                get_class_name(Predicate))

        exp_args = []
        exist_pobj = False
        for label, fillers in predicate.exp_args.items():
            if label in core_arg_list:
                assert len(fillers) == 1
                arg_type = predicate_core_arg_mapping[predicate.v_pred][label]

                if arg_type.startswith('PREP'):
                    if exist_pobj:
                        continue
                    else:
                        exist_pobj = True

                core_argument = fillers[0].get_core_argument(
                    corenlp_reader, use_lemma=use_lemma, use_entity=use_entity)

                exp_arg = RichArgument(arg_type, core_argument)
                exp_args.append(exp_arg)

        missing_labels = []
        for label in predicate_core_arg_mapping[predicate.v_pred].keys():
            if label not in predicate.exp_args:
                missing_labels.append(label)

        imp_args = []
        for label in missing_labels:
            fillers = predicate.imp_args.get(label, [])
            arg_type = predicate_core_arg_mapping[predicate.v_pred][label]
            imp_arg = RichImplicitArgument.build(
                label, arg_type, fillers, predicate.candidates, corenlp_reader,
                use_lemma=use_lemma, use_entity=use_entity,
                use_corenlp_tokens=use_corenlp_tokens)
            imp_args.append(imp_arg)

        num_candidates = len(predicate.candidates)

        return cls(
            fileid=predicate.fileid,
            n_pred=predicate.n_pred,
            v_pred=predicate.v_pred,
            exp_args=exp_args,
            imp_args=imp_args,
            num_candidates=num_candidates
        )
