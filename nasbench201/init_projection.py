import os
import sys
import numpy as np
import torch
import torch.nn.functional as f
sys.path.insert(0, '../')
import nasbench201.utils as ig_utils
import logging
import torch.utils
import copy
import scipy.stats as ss
from collections import OrderedDict
from foresight.pruners import *
from op_score import Jocab_Score, get_ntk_n
import gc
from nasbench201.linear_region import Linear_Region_Collector

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

# global-edge-iter: similar toglobal-op-iterbut iteratively selects edge e from E based on the average score of all operations on each edge
def global_op_greedy_pt_project(proj_queue, model, args): 
    def project(model, args):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op

        ##get remain eid numbers 
        remain_eids = torch.nonzero(model.candidate_flags).cpu().numpy().T[0]
        compare = lambda x, y : x < y

        crit_extrema = None
        best_eid = None
        input, target = next(iter(proj_queue))
        for eid in remain_eids:
            for opid in range(num_op):
                # projection
                weights = model.get_projected_weights()
                proj_mask = torch.ones_like(weights[eid])
                proj_mask[opid] = 0
                weights[eid] = weights[eid] * proj_mask

                ## proj evaluation
                if args.proj_crit == 'jacob':
                    valid_stats = Jocab_Score(model, input, target, weights=weights)
                    crit = valid_stats

                if crit_extrema is None or compare(crit, crit_extrema):
                    crit_extrema = crit
                    best_opid = opid
                    best_eid = eid

        logging.info('best opid %d', best_opid)
        return best_eid, best_opid

    tune_epochs = model.arch_parameters()[0].shape[0]

    for epoch in range(tune_epochs):
        logging.info('epoch %d', epoch) 
        logging.info('project')
        selected_eid, best_opid = project(model, args)
        model.project_op(selected_eid, best_opid)

    return

# global-edge-iter: similar toglobal-op-oncebut uses the average score of operations on edges to obtain the edge discretization order
def global_edge_greedy_pt_project(proj_queue, model, args):
    def select_eid(model, args):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op

        ##get remain eid numbers 
        remain_eids = torch.nonzero(model.candidate_flags).cpu().numpy().T[0]
        compare = lambda x, y : x < y

        crit_extrema = None
        best_eid = None
        input, target = next(iter(proj_queue))
        for eid in remain_eids:
            eid_score = []
            for opid in range(num_op):
                # projection
                weights = model.get_projected_weights()
                proj_mask = torch.ones_like(weights[eid])
                proj_mask[opid] = 0
                weights[eid] = weights[eid] * proj_mask

                ## proj evaluation
                if args.proj_crit == 'jacob':
                    valid_stats = Jocab_Score(model, input,  target, weights=weights)
                    crit = valid_stats
                eid_score.append(crit)
            eid_score = np.mean(eid_score)

            if crit_extrema is None or compare(eid_score, crit_extrema):
                crit_extrema = eid_score
                best_eid = eid
        return best_eid
    
    def project(model, args, selected_eid):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op

        ## select the best operation
        if args.proj_crit == 'jacob':
            crit_idx = 3
            compare = lambda x, y: x < y
        else:
            crit_idx = 4
            compare = lambda x, y: x < y
        
        best_opid = 0
        crit_list = []
        op_ids = []
        input, target = next(iter(proj_queue))
        for opid in range(num_op):
            ## projection
            weights = model.get_projected_weights()
            proj_mask = torch.ones_like(weights[selected_eid])
            proj_mask[opid] = 0
            weights[selected_eid] = weights[selected_eid] * proj_mask

            ## proj evaluation
            if args.proj_crit == 'jacob':
                valid_stats = Jocab_Score(model, input,  target, weights=weights)
                crit = valid_stats
           
            crit_list.append(crit)
            op_ids.append(opid)
            
        best_opid = op_ids[np.nanargmin(crit_list)]

        logging.info('best opid %d', best_opid)
        logging.info(crit_list)
        return selected_eid, best_opid

    num_edges = model.arch_parameters()[0].shape[0]

    for epoch in range(num_edges):
        logging.info('epoch %d', epoch)
        
        logging.info('project')
        selected_eid = select_eid(model, args)
        selected_eid, best_opid = project(model, args, selected_eid)
        model.project_op(selected_eid, best_opid)
    return

# global-op-once: only evaluates S(A−(e,o)) for all operations once to obtain a ranking order of the operations, and discretizes the edgesEaccording to this order
def global_op_once_pt_project(proj_queue, model, args):
    def order(model, args):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op

        ##get remain eid numbers 
        remain_eids = torch.nonzero(model.candidate_flags).cpu().numpy().T[0]
        compare = lambda x, y : x < y

        edge_score = OrderedDict()
        input, target = next(iter(proj_queue))
        for eid in remain_eids:       
            crit_list = []
            for opid in range(num_op):
                # projection
                weights = model.get_projected_weights()
                proj_mask = torch.ones_like(weights[eid])
                proj_mask[opid] = 0
                weights[eid] = weights[eid] * proj_mask

                ## proj evaluation
                if args.proj_crit == 'jacob':
                    valid_stats = Jocab_Score(model, input,  target, weights=weights)
                    crit = valid_stats

                crit_list.append(crit)
            edge_score[eid] = np.nanargmin(crit_list)
        return edge_score

    def project(model, args, selected_eid):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op
        ## select the best operation
        if args.proj_crit == 'jacob':
            crit_idx = 3
            compare = lambda x, y: x < y
        else:
            crit_idx = 4
            compare = lambda x, y: x < y
        
        best_opid = 0
        crit_list = []
        op_ids = []
        input, target = next(iter(proj_queue))
        for opid in range(num_op):
            ## projection
            weights = model.get_projected_weights()
            proj_mask = torch.ones_like(weights[selected_eid])
            proj_mask[opid] = 0
            weights[selected_eid] = weights[selected_eid] * proj_mask

            ## proj evaluation
            if args.proj_crit == 'jacob':
                crit = Jocab_Score(model, input,  target, weights=weights)
            crit_list.append(crit)
            op_ids.append(opid)
            
        best_opid = op_ids[np.nanargmin(crit_list)]

        logging.info('best opid %d', best_opid)
        logging.info(crit_list)
        return selected_eid, best_opid
    
    num_edges = model.arch_parameters()[0].shape[0]

    eid_order = order(model, args)
    for epoch in range(num_edges):
        logging.info('epoch %d', epoch)
        logging.info('project')
        selected_eid, _ = eid_order.popitem()
        selected_eid, best_opid = project(model, args, selected_eid)
        model.project_op(selected_eid, best_opid)

    return

# global-edge-once: similar toglobal-op-oncebut uses the average score of operations on dges to obtain the edge discretization order
def global_edge_once_pt_project(proj_queue, model, args):
    def order(model, args):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op

        ##get remain eid numbers 
        remain_eids = torch.nonzero(model.candidate_flags).cpu().numpy().T[0]
        compare = lambda x, y : x < y

        edge_score = OrderedDict()
        crit_extrema = None
        best_eid = None
        input, target = next(iter(proj_queue))
        for eid in remain_eids:       
            crit_list = []
            for opid in range(num_op):
                # projection
                weights = model.get_projected_weights()
                proj_mask = torch.ones_like(weights[eid])
                proj_mask[opid] = 0
                weights[eid] = weights[eid] * proj_mask

                ## proj evaluation
                if args.proj_crit == 'jacob':
                    crit = Jocab_Score(model, input,  target, weights=weights)

                crit_list.append(crit)
            edge_score[eid] = np.mean(crit_list)
        return edge_score

    def project(model, args, selected_eid):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op
        ## select the best operation
        if args.proj_crit == 'jacob':
            crit_idx = 3
            compare = lambda x, y: x < y
        else:
            crit_idx = 4
            compare = lambda x, y: x < y
        
        best_opid = 0
        crit_extrema = None
        crit_list = []
        op_ids = []
        input, target = next(iter(proj_queue))
        for opid in range(num_op):
            ## projection
            weights = model.get_projected_weights()
            proj_mask = torch.ones_like(weights[selected_eid])
            proj_mask[opid] = 0
            weights[selected_eid] = weights[selected_eid] * proj_mask

            ## proj evaluation
            if args.proj_crit == 'jacob':
                crit = Jocab_Score(model, input,  target, weights=weights)      
            crit_list.append(crit)
            op_ids.append(opid)
            
        best_opid = op_ids[np.nanargmin(crit_list)]

        logging.info('best opid %d', best_opid)
        logging.info(crit_list)
        return selected_eid, best_opid
    
    num_edges = model.arch_parameters()[0].shape[0]

    eid_order = order(model, args)
    for epoch in range(num_edges):
        logging.info('epoch %d', epoch)
        logging.info('project')
        selected_eid, _ = eid_order.popitem()
        selected_eid, best_opid = project(model, args, selected_eid)
        model.project_op(selected_eid, best_opid)

    return

# fixed [reverse, order]: discretizes the edges in a fixed order, where in our experiments we discretize from the222input towards the output of the cell struct
# random: discretizes the edges in a random order (DARTS-PT)
# NOTE: Only this methods allows use other zero-cost proxy metrics 
def pt_project(proj_queue, model, args):
    def project(model, args):
        ## macros,一共6条边，每条边有5个操作
        num_edge, num_op = model.num_edge, model.num_op

        ## select an edge
        remain_eids = torch.nonzero(model.candidate_flags).cpu().numpy().T[0]
        # print('candidate_flags:', model.candidate_flags)
        # print(model.candidate_flags)
        # 选边的方法
        if args.edge_decision == "random":
            # 选出来了一个数组，取其中的一个元素
            selected_eid = np.random.choice(remain_eids, size=1)[0]
        elif args.edge_decision == "reverse":
            selected_eid = remain_eids[-1]
        else:
            selected_eid = remain_eids[0]

        ## select the best operation
        if args.proj_crit == 'jacob':
            crit_idx = 3
            compare = lambda x, y: x < y
        else:
            crit_idx = 4
            compare = lambda x, y: x < y

        if args.dataset == 'cifar100':
            n_classes = 100
        elif args.dataset == 'imagenet16-120':
            n_classes = 120
        else:
            n_classes = 10

        best_opid = 0
        crit_extrema = None
        crit_list = []
        op_ids = []
        input, target = next(iter(proj_queue))
        for opid in range(num_op):
            ## projection
            weights = model.get_projected_weights()
            proj_mask = torch.ones_like(weights[selected_eid])
            # print(selected_eid, weights[selected_eid])
            proj_mask[opid] = 0
            weights[selected_eid] = weights[selected_eid] * proj_mask


            ## proj evaluation
            if args.proj_crit == 'jacob':
                crit = Jocab_Score(model, input,  target, weights=weights)
            else:
                cache_weight = model.proj_weights[selected_eid]
                cache_flag =  model.candidate_flags[selected_eid]


                for idx in range(num_op):
                    if idx == opid:
                        model.proj_weights[selected_eid][opid] = 0
                    else:
                        model.proj_weights[selected_eid][idx] = 1.0/num_op


                model.candidate_flags[selected_eid] = False
                # print(model.get_projected_weights())

                if args.proj_crit == 'comb':
                    synflow = predictive.find_measures(model,
                                        proj_queue,
                                        ('random', 1, n_classes),
                                        torch.device("cuda"),
                                        measure_names=['synflow'])
                    var = predictive.find_measures(model,
                                        proj_queue,
                                        ('random', 1, n_classes),
                                        torch.device("cuda"),
                                        measure_names=['var'])
                    # print(synflow, var)
                    comb = np.log(synflow['synflow'] + 1) / (var['var'] + 0.1)
                    measures = {'comb': comb}
                else:
                    measures = predictive.find_measures(model,
                                             proj_queue,
                                             ('random', 1, n_classes),
                                             torch.device("cuda"),
                                             measure_names=[args.proj_crit])

                # print(measures)
                for idx in range(num_op):
                    model.proj_weights[selected_eid][idx] = 0
                model.candidate_flags[selected_eid] = cache_flag
                crit = measures[args.proj_crit]

            crit_list.append(crit)
            op_ids.append(opid)


        best_opid = op_ids[np.nanargmin(crit_list)]
        # best_opid = op_ids[np.nanargmax(crit_list)]

        logging.info('best opid %d', best_opid)
        logging.info('current edge id %d', selected_eid)
        logging.info(crit_list)
        return selected_eid, best_opid
    
    num_edges = model.arch_parameters()[0].shape[0]

    for epoch in range(num_edges):
        logging.info('epoch %d', epoch)        
        logging.info('project')
        selected_eid, best_opid = project(model, args)
        model.project_op(selected_eid, best_opid)

    return

def tenas_project(proj_queue, model, model_thin, args):
    def project(model, args):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op

        ##get remain eid numbers 
        remain_eids = torch.nonzero(model.candidate_flags).cpu().numpy().T[0]
        compare = lambda x, y : x < y

        ntks = []
        lrs = []
        edge_op_id = []
        best_eid = None
        
        if args.proj_crit == 'tenas':
            lrc_model = Linear_Region_Collector(input_size=(1000, 1, 3, 3), sample_batch=3, dataset=args.dataset, data_path=args.data, seed=args.seed)
        for eid in remain_eids:
            for opid in range(num_op):
                # projection
                weights = model.get_projected_weights()
                proj_mask = torch.ones_like(weights[eid])
                proj_mask[opid] = 0
                weights[eid] = weights[eid] * proj_mask

                ## proj evaluation
                if args.proj_crit == 'tenas':
                    lrc_model.reinit(ori_models=[model_thin], seed=args.seed, weights=weights)
                    lr = lrc_model.forward_batch_sample()
                    lrc_model.clear()
                    ntk = get_ntk_n(proj_queue, [model], recalbn=0, train_mode=True, num_batch=1, weights=weights)
                    ntks.append(ntk)
                    lrs.append(lr)
                    edge_op_id.append('{}:{}'.format(eid, opid))
        print('ntls', ntks)
        print('lrs', lrs)
        ntks_ranks = ss.rankdata(ntks)
        lrs_ranks = ss.rankdata(lrs)
        ntks_ranks = len(ntks_ranks) - ntks_ranks.astype(int)
        op_ranks = []
        for i in range(len(edge_op_id)):
            op_ranks.append(ntks_ranks[i]+lrs_ranks[i])
        
        best_op_index = edge_op_id[np.nanargmin(op_ranks[0:num_op])]
        best_eid, best_opid = [int(x) for x in best_op_index.split(':')]

        logging.info(op_ranks)
        logging.info('best eid %d', best_eid)
        logging.info('best opid %d', best_opid)
        return best_eid, best_opid
    num_edges = model.arch_parameters()[0].shape[0]

    for epoch in range(num_edges):
        logging.info('epoch %d', epoch)        
        logging.info('project')
        selected_eid, best_opid = project(model, args)
        model.project_op(selected_eid, best_opid)

    return

#new methods 
#Randomly propose candidate of networks and transfer it to supernet, then perform global op selection in this subspace
def shrink_pt_project(proj_queue, model, args):
    def project(model, args):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op

        ## select an edge
        remain_eids = torch.nonzero(model.candidate_flags).cpu().numpy().T[0]
        selected_eid = np.random.choice(remain_eids, size=1)[0]


        ## select the best operation
        if args.proj_crit == 'jacob':
            crit_idx = 3
            compare = lambda x, y: x < y
        else:
            crit_idx = 4
            compare = lambda x, y: x < y

        if args.dataset == 'cifar100':
            n_classes = 100
        elif args.dataset == 'imagenet16-120':
            n_classes = 120
        else:
            n_classes = 10

        best_opid = 0
        crit_extrema = None
        crit_list = []
        op_ids = []
        input, target = next(iter(proj_queue))
        for opid in range(num_op):
            ## projection
            weights = model.get_projected_weights()
            proj_mask = torch.ones_like(weights[selected_eid])
            proj_mask[opid] = 0
            weights[selected_eid] = weights[selected_eid] * proj_mask

            ## proj evaluation
            if args.proj_crit == 'jacob':
                crit = Jocab_Score(model, input,  target, weights=weights)
            else:
                cache_weight = model.proj_weights[selected_eid]
                cache_flag =  model.candidate_flags[selected_eid]

                for idx in range(num_op):
                    if idx == opid:
                        model.proj_weights[selected_eid][opid] = 0
                    else:
                        model.proj_weights[selected_eid][idx] = 1.0/num_op
                model.candidate_flags[selected_eid] = False
                
                measures = predictive.find_measures(model,
                                    train_queue,
                                    ('random', 1, n_classes), 
                                    torch.device("cuda"),
                                    measure_names=[args.proj_crit])
                for idx in range(num_op):
                    model.proj_weights[selected_eid][idx] = 0
                model.candidate_flags[selected_eid] = cache_flag
                crit = measures[args.proj_crit]

            crit_list.append(crit)
            op_ids.append(opid)
            
        best_opid = op_ids[np.nanargmin(crit_list)]

        logging.info('best opid %d', best_opid)
        logging.info('current edge id %d', selected_eid)
        logging.info(crit_list)
        return selected_eid, best_opid
    
    def global_project(model, args):
        ## macros
        num_edge, num_op = model.num_edge, model.num_op

        ##get remain eid numbers 
        remain_eids = torch.nonzero(model.subspace_candidate_flags).cpu().numpy().T[0]
        compare = lambda x, y : x < y

        crit_extrema = None
        best_eid = None
        best_opid = None
        input, target = next(iter(proj_queue))
        for eid in remain_eids:
            remain_oids = torch.nonzero(model.proj_weights[eid]).cpu().numpy().T[0]
            for opid in remain_oids:
                # projection
                weights = model.get_projected_weights()
                proj_mask = torch.ones_like(weights[eid])
                proj_mask[opid] = 0
                weights[eid] = weights[eid] * proj_mask
                ## proj evaluation
                if args.proj_crit == 'jacob':
                    valid_stats = Jocab_Score(model, input, target, weights=weights)
                    crit = valid_stats

                if crit_extrema is None or compare(crit, crit_extrema):
                    crit_extrema = crit
                    best_opid = opid
                    best_eid = eid


        logging.info('best eid %d', best_eid)
        logging.info('best opid %d', best_opid)
        model.subspace_candidate_flags[best_eid] = False
        proj_mask = torch.zeros_like(model.proj_weights[best_eid])
        model.proj_weights[best_eid] = model.proj_weights[best_eid] * proj_mask
        model.proj_weights[best_eid][best_opid] = 1
        return best_eid, best_opid

    num_edges = model.arch_parameters()[0].shape[0]

    #subspace
    logging.info('Start subspace proposal')
    subspace = copy.deepcopy(model.proj_weights)
    for i in range(20):
        model.reset_arch_parameters()
        for epoch in range(num_edges):
            logging.info('epoch %d', epoch)        
            logging.info('project')
            selected_eid, best_opid = project(model, args)
            model.project_op(selected_eid, best_opid)
        subspace += model.proj_weights
    
    model.reset_arch_parameters()
    subspace = torch.gt(subspace, 0).int().float()
    subspace = f.normalize(subspace, p=1, dim=1)
    model.proj_weights += subspace
    for i in range(num_edges):
        model.candidate_flags[i] = False
    logging.info('Start final search in subspace')
    logging.info(subspace)

    model.subspace_candidate_flags = torch.tensor(len(model._arch_parameters) * [True], requires_grad=False, dtype=torch.bool).cuda()
    for epoch in range(num_edges):
        logging.info('epoch %d', epoch) 
        logging.info('project')
        selected_eid, best_opid = global_project(model, args)
        model.printing(logging)
        #model.project_op(selected_eid, best_opid)
    return