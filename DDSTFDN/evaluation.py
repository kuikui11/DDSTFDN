import torch
import numpy as np
import os
import torch.nn.functional as F
import torch.utils.data as tordata
import torch.autograd as autograd
from tqdm import tqdm
import time
import sys
import json
from DenseGait_BaseOpenGait.model.base_model import BaseModel
from DenseGait_BaseOpenGait.data.dataset import DataSet
import DenseGait_BaseOpenGait.data.sampler as Samplers
from DenseGait_BaseOpenGait.data.collate_fn import CollateFn
from DenseGait_BaseOpenGait.data.transform import get_transform
from DenseGait_BaseOpenGait.model.model_tools import *
from DenseGait_BaseOpenGait.utils.common import init_seeds,config_loader,get_attr_from,get_valid_args,get_ddp_module

from utils import *

def compute_ACC_mAP(distmat, q_pids, g_pids, q_views=None, g_views=None, rank=1):
    num_q, _ = distmat.shape
    # indices = np.argsort(distmat, axis=1)
    # matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_ACC = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        q_idx_dist = distmat[q_idx]
        q_idx_glabels = g_pids
        if q_views is not None and g_views is not None:
            q_idx_mask = np.isin(g_views, q_views[q_idx], invert=True) | np.isin(
                g_pids, q_pids[q_idx], invert=True)
            q_idx_dist = q_idx_dist[q_idx_mask]
            q_idx_glabels = q_idx_glabels[q_idx_mask]

        assert(len(q_idx_glabels) >
               0), "No gallery after excluding identical-view cases!"
        q_idx_indices = np.argsort(q_idx_dist)
        q_idx_matches = (q_idx_glabels[q_idx_indices]
                         == q_pids[q_idx]).astype(np.int32)

        # binary vector, positions with value 1 are correct matches
        # orig_cmc = matches[q_idx]
        orig_cmc = q_idx_matches
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_ACC.append(cmc[rank-1])

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()

        if num_rel > 0:
            num_valid_q += 1.
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

    # all_ACC = np.asarray(all_ACC).astype(np.float32)
    ACC = np.mean(all_ACC)
    mAP = np.mean(all_AP)

    return ACC, mAP
def de_diag(acc, each_angle=False):
    # Exclude identical-view cases
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result

def single_view_gallery_evaluation(data, dataset, metric):
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},
                      'OUMVLP': {'NM': ['00']},
                     }
    gallery_seq_dict = {'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                        'OUMVLP': ['01'],
                       }
    # msg_mgr = get_msg_mgr()
    acc = {}
    feature, view, seq_type, label = data
    view_list = sorted(np.unique(view))
    num_rank = 1
    view_num = len(view_list)
    label = np.array(label)

    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros((view_num, view_num, num_rank)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            # print(pseq_mask)
            # print(label)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset]) & np.isin(
                    view, [gallery_view])
                gallery_y = label[gseq_mask]
                gallery_x = feature[gseq_mask, :]
                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                acc[type_][v1, v2, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                                     0) * 100 / dist.shape[0], 2)

    result_dict = {}
    print('===Rank-1 (Exclude identical-view cases)===')
    out_str = ""
    for rank in range(num_rank):
        out_str = ""
        for type_ in probe_seq_dict[dataset].keys():
            sub_acc = de_diag(acc[type_][:,:,rank], each_angle=True)
            if rank == 0:
                print(f'{type_}@R{rank+1}: {sub_acc}')
                result_dict[f'scalar/test_accuracy/{type_}@R{rank+1}'] = np.mean(sub_acc)
            out_str += f"{type_}@R{rank+1}: {np.mean(sub_acc):.2f}%\t"
        print(out_str)
    return result_dict

def cross_view_gallery_evaluation(data, dataset, metric):#feature, label, seq_type, view,
    '''More details can be found: More details can be found in
        [A Comprehensive Study on the Evaluation of Silhouette-based Gait Recognition](https://ieeexplore.ieee.org/document/9928336).
    '''
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-01'], 'BG': ['bg-01'], 'CL': ['cl-01']},
                      'OUMVLP': {'NM': ['00']}}

    gallery_seq_dict = {'CASIA-B': ['nm-02', 'bg-02', 'cl-02'],
                        'OUMVLP': ['01']}

    # msg_mgr = get_msg_mgr()

    mean_ap = {}
    acc = {}
    feature, view, seq_type, label = data
    view_list = sorted(np.unique(view))
    num_rank = 1
    view_num = len(view_list)
    label = np.array(label)
    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros(len(view_list)) - 1.
        mean_ap[type_] = np.zeros(len(view_list)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset])
            gallery_y = label[gseq_mask]
            gallery_x = feature[gseq_mask, :]
            dist = cuda_dist(probe_x, gallery_x, metric)
            eval_results = compute_ACC_mAP(
                dist.cpu().numpy(), probe_y, gallery_y, view[pseq_mask], view[gseq_mask])
            acc[type_][v1] = np.round(eval_results[0] * 100, 2)
            mean_ap[type_][v1] = np.round(eval_results[1] * 100, 2)

    result_dict = {}
    print(
        '===Cross View Gallery Evaluation (Excluded identical-view cases)===')
    out_acc_str = "========= Rank@1 Acc =========\n"
    out_map_str = "============= mAP ============\n"
    for type_ in probe_seq_dict[dataset].keys():
        avg_acc = np.mean(acc[type_])
        avg_map = np.mean(mean_ap[type_])
        result_dict[f'scalar/test_accuracy/{type_}-Rank@1'] = avg_acc
        result_dict[f'scalar/test_accuracy/{type_}-mAP'] = avg_map
        out_acc_str += f"{type_}:\t{acc[type_]}, mean: {avg_acc:.2f}%\n"
        out_map_str += f"{type_}:\t{mean_ap[type_]}, mean: {avg_map:.2f}%\n"
    print(f'========= Rank@1 Acc =========')
    print(f'{out_acc_str}')
    print(f'========= mAP =========')
    print(f'{out_map_str}')
    return result_dict

def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin

def evaluation(data, config):
    # dataset = config['dataset_name'].split('-')[0]
    dataset = config['dataset_name']
    feature, view, seq_type, label = data
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    sample_num = len(feature)

    probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}
    gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x,metric='euc')
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)

    return acc


def ts2var(x):
    return autograd.Variable(x).cuda()
def np2var(x):

    return torch.from_numpy(x)


def transform(configs, model_weight = None):
    # init_seeds(seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    torch.cuda.empty_cache()
    print("using {}device.".format(device))
    # configs = config_loader(path='configs/DenseGait_casiab.yaml')
    # print(configs)
    test_loader = get_loader(data_cfg=configs,train=False)
    # print(dataloader)
    net = BaseModel(model_cfg=configs['model_cfg'],device=device)
    weight_path = model_weight
    assert os.path.exists(weight_path), "file {} does exist.".format(weight_path)  
    net.load_state_dict(torch.load(weight_path), strict=False)  
    net.to(device)
    net.eval()
    print(test_loader.__len__())
    feature_list = list()
    view_list = list()
    seq_type_list = list()
    label_list = list()
    with torch.no_grad():
        for step, x in enumerate(test_loader):
            frames, label, seq_type, view, batch_frame = x
            data = inputs_pretreament(cfgs=configs,inputs=x,train=False)

            # print(data[0].size)
            embed,logits = net(data)
            # n, num_bin, _ = feature.size()
            # feature_list.append(feature.view(n, -1).cpu().numpy())
            feature_list.append(logits.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            # print(label)
            label_list += label
            # view_list.append(view)
            # seq_type_list.append(seq_type)
            # label_list.append(label)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            if (step+1) % 1000 == 0:
                print("当前进度{}/{}".format((step+1)*16,80000))
            # print(step)
            if (step+1)*16 >= 80000:
                print("特征保存完毕")
                break
    return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list
    # return feature_list, view_list, seq_type_list, label_list

def test_acc(model_weight_path):
    init_seeds(seed=42)
    configs = config_loader(path='configs/DenseGait_casiab.yaml')
    test = transform(configs=configs, model_weight = model_weight_path)

    # acc = evaluation(data=test,config=configs['data_cfg'])
    # acc = single_view_gallery_evaluation(data=test, dataset='OUMVLP', metric='euc')
    acc = single_view_gallery_evaluation(data=test, dataset='CASIA-B', metric='euc')
    print(acc)

