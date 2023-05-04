import sys
sys.path.insert(0, '../../')
import numpy as np
import torch
import logging
import torch.utils
from copy import deepcopy
from foresight.pruners import *

torch.set_printoptions(precision=4, sci_mode=False)

def sample_op(model, input, target, args, cell_type, selected_eid=None):
    ''' operation '''
    #### macros
    num_edges, num_ops = model.num_edges, model.num_ops
    candidate_flags = model.candidate_flags[cell_type]
    proj_crit = args.proj_crit[cell_type]

    #### select an edge
    if selected_eid is None:
        remain_eids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
        selected_eid = np.random.choice(remain_eids, size=1)[0]
        logging.info('selected edge: %d %s', selected_eid, cell_type)
    
    select_opid = np.random.choice(np.array(range(num_ops)), size=1)[0]
    return selected_eid, select_opid

def project_op(model, input, target, args, cell_type, proj_queue=None, selected_eid=None):
    ''' operation '''
    #### macros
    num_edges, num_ops = model.num_edges, model.num_ops
    candidate_flags = model.candidate_flags[cell_type]
    proj_crit = args.proj_crit[cell_type]
    
    #### select an edge
    if selected_eid is None:
        remain_eids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
        # print(num_edges, num_ops, remain_eids)
        if args.edge_decision == "random":
            selected_eid = np.random.choice(remain_eids, size=1)[0]
            logging.info('selected edge: %d %s', selected_eid, cell_type)
        elif args.edge_decision == 'reverse':
            selected_eid = remain_eids[-1]
            logging.info('selected edge: %d %s', selected_eid, cell_type)
        else:
            selected_eid = remain_eids[0]
            logging.info('selected node: %d %s', selected_eid, cell_type)

    #### select the best operation
    if proj_crit == 'jacob':
        crit_idx = 3
        compare = lambda x, y: x < y
    else:
        crit_idx = 0
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
    for opid in range(num_ops):
        ## projection
        weights = model.get_projected_weights(cell_type)
        proj_mask = torch.ones_like(weights[selected_eid])
        proj_mask[opid] = 0
        weights[selected_eid] = weights[selected_eid] * proj_mask

        # ## proj evaluation
        # with torch.no_grad():
        #     valid_stats = Jocab_Score(model, cell_type, input, target, weights=weights)
        #     crit = valid_stats
        #     crit_list.append(crit)
        #     if crit_extrema is None or compare(crit, crit_extrema):
        #         crit_extrema = crit
        #         best_opid = opid

        ## proj evaluation
        if proj_crit == 'jacob':
            crit = Jocab_Score(model,cell_type, input, target, weights=weights)
        else:
            cache_weight = model.proj_weights[cell_type][selected_eid]
            cache_flag = model.candidate_flags[cell_type][selected_eid]

            for idx in range(num_ops):
                if idx == opid:
                    model.proj_weights[cell_type][selected_eid][opid] = 0
                else:
                    model.proj_weights[cell_type][selected_eid][idx] = 1.0 / num_ops

            model.candidate_flags[cell_type][selected_eid] = False
            # print(model.get_projected_weights())
            measures = predictive.find_measures(model,
                                                proj_queue,
                                                ('random', 1, n_classes),
                                                torch.device("cuda"),
                                                measure_names=[proj_crit])

            # print(measures)
            for idx in range(num_ops):
                model.proj_weights[cell_type][selected_eid][idx] = 0
            model.candidate_flags[cell_type][selected_eid] = cache_flag
            crit = measures[proj_crit]

        crit_list.append(crit)
        op_ids.append(opid)

    best_opid = op_ids[np.nanargmin(crit_list)]



    #### project
    logging.info('best opid: %d', best_opid)
    logging.info(crit_list)
    return selected_eid, best_opid
    
def project_global_op(model, input, target, args, infer, cell_type, selected_eid=None):
    ''' operation '''
    #### macros
    num_edges, num_ops = model.num_edges, model.num_ops
    candidate_flags = model.candidate_flags[cell_type]
    proj_crit = args.proj_crit[cell_type]
    
    remain_eids = torch.nonzero(candidate_flags).cpu().numpy().T[0]

    #### select the best operation
    if proj_crit == 'jacob':
        crit_idx = 3
        compare = lambda x, y: x < y

    best_opid = 0
    crit_extrema = None
    best_eid = None
    for eid in remain_eids:
        for opid in range(num_ops):
            ## projection
            weights = model.get_projected_weights(cell_type)
            proj_mask = torch.ones_like(weights[eid])
            proj_mask[opid] = 0
            weights[eid] = weights[eid] * proj_mask

            ## proj evaluation
            
            #weights_dict = {cell_type:weights}
            with torch.no_grad():
                valid_stats = Jocab_Score(model, cell_type, input, target, weights=weights)
                crit = valid_stats
                if crit_extrema is None or compare(crit, crit_extrema):
                    crit_extrema = crit
                    best_opid = opid
                    best_eid = eid

    #### project
    logging.info('best opid: %d', best_opid)
    #logging.info(crit_list)
    return best_eid, best_opid

def sample_edge(model, input, target, args, cell_type, selected_eid=None):
    ''' topology '''
    #### macros
    candidate_flags = model.candidate_flags_edge[cell_type]
    proj_crit = args.proj_crit[cell_type]

    #### select an node
    remain_nids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
    selected_nid = np.random.choice(remain_nids, size=1)[0]
    logging.info('selected node: %d %s', selected_nid, cell_type)
    
    eids = deepcopy(model.nid2eids[selected_nid])

    while len(eids) > 2:
        elected_eid = np.random.choice(eids, size=1)[0]
        eids.remove(elected_eid)

    return selected_nid, eids

def project_edge(model, input, target, args, cell_type):
    ''' topology '''
    #### macros
    candidate_flags = model.candidate_flags_edge[cell_type]
    proj_crit = args.proj_crit[cell_type]

    #### select an node
    remain_nids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
    if args.edge_decision == "random":
        selected_nid = np.random.choice(remain_nids, size=1)[0]
        logging.info('selected node: %d %s', selected_nid, cell_type)
    elif args.edge_decision == 'reverse':
        selected_nid = remain_nids[-1]
        logging.info('selected node: %d %s', selected_nid, cell_type)
    else:
        selected_nid = np.random.choice(remain_nids, size=1)[0]
        logging.info('selected node: %d %s', selected_nid, cell_type)
    
    #### select top2 edges
    if proj_crit == 'jacob':
        crit_idx = 3
        compare = lambda x, y: x < y
    else:
        crit_idx = 3
        compare = lambda x, y: x < y

    eids = deepcopy(model.nid2eids[selected_nid])
    crit_list = []
    while len(eids) > 2:
        eid_todel = None
        crit_extrema = None
        for eid in eids:
            weights = model.get_projected_weights(cell_type)
            weights[eid].data.fill_(0)

            ## proj evaluation
            with torch.no_grad():
                valid_stats = Jocab_Score(model, cell_type, input, target, weights=weights)
                crit = valid_stats

                crit_list.append(crit)
                if crit_extrema is None or not compare(crit, crit_extrema): # find out bad edges
                    crit_extrema = crit
                    eid_todel = eid

        eids.remove(eid_todel)

    #### project
    logging.info('top2 edges: (%d, %d)', eids[0], eids[1])
    #logging.info(crit_list)
    return selected_nid, eids


def pt_project(train_queue, model, args):
    model.eval()

    #### macros
    num_projs = model.num_edges + len(model.nid2eids.keys()) 
    args.proj_crit = {'normal':args.proj_crit_normal, 'reduce':args.proj_crit_reduce}
    proj_queue = train_queue

    epoch = 0
    for step, (input, target) in enumerate(proj_queue):
        if epoch < model.num_edges:
            logging.info('project op')
            
            if args.edge_decision == 'global_op_greedy':
                selected_eid_normal, best_opid_normal = project_global_op(model, input, target, args, cell_type='normal')
            elif args.edge_decision == 'sample':
                selected_eid_normal, best_opid_normal  = sample_op(model, input, target, args, cell_type='normal')
            else:
                selected_eid_normal, best_opid_normal = project_op(model, input, target, args, proj_queue=proj_queue, cell_type='normal')
            model.project_op(selected_eid_normal, best_opid_normal, cell_type='normal')
            if args.edge_decision == 'global_op_greedy':
                selected_eid_reduce, best_opid_reduce = project_global_op(model, input, target, args, cell_type='reduce')
            elif args.edge_decision == 'sample':
                selected_eid_reduce, best_opid_reduce  = sample_op(model, input, target, args, cell_type='reduce')
            else:
                selected_eid_reduce, best_opid_reduce = project_op(model, input, target, args, proj_queue=proj_queue, cell_type='reduce')
            model.project_op(selected_eid_reduce, best_opid_reduce, cell_type='reduce')

        else:
            logging.info('project edge')
            if args.edge_decision == 'sample':
                selected_nid_normal, eids_normal = sample_edge(model, input, target, args, cell_type='normal')
                model.project_edge(selected_nid_normal, eids_normal, cell_type='normal')
                selected_nid_reduce, eids_reduce = sample_edge(model, input, target, args, cell_type='reduce')
                model.project_edge(selected_nid_reduce, eids_reduce, cell_type='reduce')
            else:
                selected_nid_normal, eids_normal = project_edge(model, input, target, args, cell_type='normal')
                model.project_edge(selected_nid_normal, eids_normal, cell_type='normal')
                selected_nid_reduce, eids_reduce = project_edge(model, input, target, args, cell_type='reduce')
                model.project_edge(selected_nid_reduce, eids_reduce, cell_type='reduce')
        epoch+=1

        if epoch == num_projs:
            break

    return

def Jocab_Score(ori_model, cell_type, input, target, weights=None):
    model = deepcopy(ori_model)
    model.eval()
    if cell_type == 'reduce':
        model.proj_weights['reduce'] = weights
        model.proj_weights['normal'] = model.get_projected_weights('normal')
    else:
        model.proj_weights['normal'] = weights
        model.proj_weights['reduce'] = model.get_projected_weights('reduce')

    batch_size = input.shape[0]
    model.K = torch.zeros(batch_size, batch_size).cuda()
    def counting_forward_hook(module, inp, out):
        try:
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K = x @ x.t()
            K2 = (1.-x) @ (1.-x.t())
            model.K = model.K + K + K2
        except:
            pass

    for name, module in model.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)
    
    input = input.cuda()

    model(input, using_proj=True)
    score = hooklogdet(model.K.cpu().numpy())

    del model
    return score

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld