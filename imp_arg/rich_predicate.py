import random
from copy import deepcopy
from itertools import permutations

from consts import core_arg_list, predicate_core_arg_mapping
from predicate import Predicate
from rich_implicit_argument import RichImplicitArgument
from rich_script.indexed_event import IndexedEvent
from rich_script.rich_argument import RichArgument
from rich_script.rich_entity import EntitySalience
from util import get_class_name


class RichPredicate(object):
    def __init__(self, fileid, n_pred, v_pred, exp_args, imp_args,
                 candidate_core_list, candidate_salience_list, num_candidates):
        self.fileid = fileid
        self.n_pred = n_pred
        self.v_pred = v_pred
        self.pred_wv = -1
        assert all(isinstance(arg, RichArgument) for arg in exp_args)
        self.exp_args = exp_args
        assert all(isinstance(arg, RichImplicitArgument) for arg in imp_args)
        self.imp_args = imp_args

        self.candidate_core_list = candidate_core_list
        self.candidate_salience_list = candidate_salience_list
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
                        self.sum_dice += imp_arg.get_eval_dice_score()

        return self.sum_dice, self.num_gt, self.num_model

    def set_pred_wv(self, pred_wv_mapping):
        self.pred_wv = pred_wv_mapping[self.v_pred]

    def get_index(self, model, include_type=True, use_unk=True):
        for exp_arg in self.exp_args:
            exp_arg.get_index(model, include_type=include_type, use_unk=use_unk)
        for imp_arg in self.imp_args:
            imp_arg.get_index(model, include_type=include_type, use_unk=use_unk)

    def get_pos_input(self):
        if self.pred_wv == -1:
            raise RuntimeError('found pred_wv == -1')
        pos_input = IndexedEvent(
            self.pred_wv,
            self.rich_subj.get_pos_wv() if self.rich_subj else -1,
            self.rich_obj.get_pos_wv() if self.rich_obj else -1,
            self.rich_pobj.get_pos_wv() if self.rich_pobj else -1)
        return pos_input

    def get_eval_input_list_all(self, include_salience=True):
        pos_input = self.get_pos_input()

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

    def get_pair_input_list(self, pair_type, **kwargs):
        # include the following 3 types of pair tuning inputs
        # 1) event w/ true arg vs event w/ false arg
        # 2) event w/ arg (which imp_arg exists) vs event w/o arg
        # and event w/o arg (which imp_arg doesn't exist) vs event w/ arg
        # 3) event w/ cand_a as arg_i vs event w/ cand_a as arg_j
        # (which cand_a is the true arg for imp_arg_i,
        # and both imp_arg_i and imp_arg_j exist)
        base_input = self.get_pos_input()
        if base_input is None:
            return []

        assert pair_type in ['tf_arg', 'wo_arg', 'two_args']

        pair_input_list = []

        if pair_type == 'tf_arg':
            neg_sample_type = kwargs['neg_sample_type']
            assert neg_sample_type in ['one', 'all']
            for imp_arg in self.imp_args:
                if imp_arg.exist:
                    if len(imp_arg.rich_candidate_list) <= 1:
                        continue

                    arg_idx = imp_arg.get_arg_idx()

                    pos_input = deepcopy(base_input)
                    pos_input.set_argument(arg_idx, imp_arg.get_pos_wv())
                    pos_salience = imp_arg.get_pos_candidate().entity_salience
                    if pos_salience is None:
                        pos_salience = EntitySalience(**{})

                    neg_cand_list = zip(imp_arg.get_neg_candidate_list(),
                                        imp_arg.get_neg_wv_list())

                    if neg_sample_type == 'one':
                        neg_cand, neg_wv = random.choice(neg_cand_list)
                        neg_input = deepcopy(base_input)
                        neg_input.set_argument(arg_idx, neg_wv)
                        neg_salience = neg_cand.entity_salience
                        if neg_salience is None:
                            neg_salience = EntitySalience(**{})

                        pair_input_list.append((
                            pos_input, neg_input, arg_idx, arg_idx,
                            pos_salience, neg_salience))
                    else:
                        for neg_cand, neg_wv in neg_cand_list:
                            neg_input = deepcopy(base_input)
                            neg_input.set_argument(arg_idx, neg_wv)
                            neg_salience = neg_cand.entity_salience
                            if neg_salience is None:
                                neg_salience = EntitySalience(**{})

                            pair_input_list.append((
                                pos_input, neg_input, arg_idx, arg_idx,
                                pos_salience, neg_salience))

        elif pair_type == 'wo_arg':
            for imp_arg in self.imp_args:
                arg_idx = imp_arg.get_arg_idx()
                arg_type = imp_arg.arg_type

                if imp_arg.exist:
                    pos_input = deepcopy(base_input)
                    pos_input.set_argument(arg_idx, imp_arg.get_pos_wv())
                    pos_salience = imp_arg.get_pos_candidate().entity_salience

                    neg_input = deepcopy(base_input)
                    neg_input.set_argument(arg_idx, -1)
                    neg_salience = EntitySalience(**{})

                else:
                    if self.num_candidates <= 0:
                        continue
                    pos_input = deepcopy(base_input)
                    pos_input.set_argument(arg_idx, 1)
                    pos_salience = EntitySalience(**{})
                    neg_input = deepcopy(base_input)
                    neg_core, neg_salience = random.choice(
                        zip(self.candidate_core_list,
                            self.candidate_salience_list))
                    neg_wv = neg_core.get_index(
                        kwargs['model'],
                        arg_type=arg_type if kwargs['include_type'] else '',
                        use_unk=kwargs['use_unk'])
                    neg_input.set_argument(arg_idx, neg_wv)

                pair_input_list.append((
                    pos_input, neg_input, arg_idx, arg_idx,
                    pos_salience, neg_salience))

        else:
            existing_imp_arg = \
                [imp_arg for imp_arg in self.imp_args if imp_arg.exist]
            for pos_imp_arg, neg_imp_arg in permutations(existing_imp_arg, 2):
                pos_arg_idx = pos_imp_arg.get_arg_idx()
                neg_arg_idx = neg_imp_arg.get_arg_idx()
                pos_wv = pos_imp_arg.get_pos_wv()

                pos_input = deepcopy(base_input)
                pos_input.set_argument(pos_arg_idx, pos_wv)
                pos_input.set_argument(neg_arg_idx, -1)

                neg_input = deepcopy(base_input)
                neg_input.set_argument(pos_arg_idx, -1)
                neg_input.set_argument(neg_arg_idx, pos_wv)

                pos_salience = pos_imp_arg.get_pos_candidate().entity_salience
                neg_salience = pos_salience

                pair_input_list.append((
                    pos_input, neg_input, pos_arg_idx, neg_arg_idx,
                    pos_salience, neg_salience))

        # random.shuffle(pair_input_list)
        return pair_input_list

    @classmethod
    def build(cls, predicate, corenlp_reader, use_lemma=True, use_entity=True,
              use_corenlp_tokens=True, labeled_arg_only=False):
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

        if labeled_arg_only:
            missing_labels = predicate.imp_args.keys()
        else:
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

        candidate_core_list = [
            candidate.arg_pointer.get_core_argument(
                corenlp_reader, use_lemma=use_lemma, use_entity=use_entity)
            for candidate in predicate.candidates]
        candidate_salience_list = [
            candidate.arg_pointer.get_entity_salience(
                corenlp_reader, use_entity=use_entity)
            for candidate in predicate.candidates]

        num_candidates = len(predicate.candidates)

        return cls(
            fileid=predicate.fileid,
            n_pred=predicate.n_pred,
            v_pred=predicate.v_pred,
            exp_args=exp_args,
            imp_args=imp_args,
            candidate_core_list=candidate_core_list,
            candidate_salience_list=candidate_salience_list,
            num_candidates=num_candidates
        )
