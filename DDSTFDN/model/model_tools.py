import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata
from tqdm import tqdm
import time
import sys
import os
import json

from DenseGait_BaseOpenGait.data.dataset import DataSet
import DenseGait_BaseOpenGait.data.sampler as Samplers
from DenseGait_BaseOpenGait.data.collate_fn import CollateFn
from DenseGait_BaseOpenGait.data.transform import get_transform
from DenseGait_BaseOpenGait.utils.common import get_valid_args, is_list, is_dict,get_attr_from,get_valid_args,np2var,ts2np,list2var

def get_loader(data_cfg, train=True):
    sampler_cfg = data_cfg['trainer_cfg']['sampler'] if train else data_cfg['evaluator_cfg']['sampler']
    dataset = DataSet(data_cfg['data_cfg'], train)

    Sampler = get_attr_from([Samplers], sampler_cfg['type'])
    vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=[
        'sample_type', 'type'])
    sampler = Sampler(dataset, **vaild_args)

    loader = tordata.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=CollateFn(dataset.label_set, sampler_cfg),
        num_workers=data_cfg['data_cfg']['num_workers'])
    return loader

def inputs_pretreament(cfgs,inputs,train=True):
    """Conduct transforms on input data.

    Args:
        inputs: the input data.
    Returns:
        tuple: training data including inputs, labels, and some meta data.
    """
    trainer_trfs = get_transform(cfgs['trainer_cfg']['transform'])
    evaluator_trfs = get_transform(cfgs['evaluator_cfg']['transform'])
    seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
    seq_trfs = trainer_trfs if train else evaluator_trfs
    if len(seqs_batch) != len(seq_trfs):
        raise ValueError(
            "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
    requires_grad = bool(train)
    seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
            for trf, seq in zip(seq_trfs, seqs_batch)]

    typs = typs_batch
    vies = vies_batch

    labs = list2var(labs_batch).long()

    if seqL_batch is not None:
        seqL_batch = np2var(seqL_batch).int()
    seqL = seqL_batch

    if seqL is not None:
        seqL_sum = int(seqL.sum().data.cpu().numpy())
        ipts = [_[:, :seqL_sum] for _ in seqs]
    else:
        ipts = seqs
    del seqs
    return ipts, labs, typs, vies, seqL


def get_optimizer(net, optimizer_cfg):
    print(optimizer_cfg)
    optimizer = get_attr_from([optim], optimizer_cfg['solver'])
    valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
    optimizer = optimizer(
        filter(lambda p: p.requires_grad, net.parameters()), **valid_arg)
    return optimizer

def get_scheduler(optimizer, scheduler_cfg):
    print(scheduler_cfg)
    Scheduler = get_attr_from(
        [optim.lr_scheduler], scheduler_cfg['scheduler'])
    valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
    scheduler = Scheduler(optimizer, **valid_arg)
    return scheduler