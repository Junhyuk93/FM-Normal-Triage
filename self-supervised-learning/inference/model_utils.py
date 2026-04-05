# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import csv
import logging
from typing import Dict, Optional
from tqdm import tqdm
import torch
from torch import nn
from torchmetrics import MetricCollection
import numpy as np

from data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
# import dinov2.distributed as distributed
# from dinov2.logging import MetricLogger


# logger = logging.getLogger("dinov2")


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx
    def forward(self, images):
        with torch.inference_mode():
            features = self.feature_model.get_intermediate_layers(
                images, self.n_last_blocks, return_class_token=True
            )
        return features



@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    # postprocessors: Dict[str, nn.Module],
    # metrics: Dict[str, MetricCollection],
    postprocessors,
    
    # criterion: Optional[nn.Module] = None,
):
    device = 'cuda'
    model.eval()
    postprocessors.eval()
    # if criterion is not None:
    #     criterion.eval()

    # for metric in metrics.values():
    #     metric = metric.to(device)

    # metric_logger = MetricLogger(delimiter="  ")
    # header = "Test:"
    
    f = open('inference.csv','w',newline='')
    wr = csv.writer(f)
    wr.writerow(['인덱스','경로','타겟 클래스','예측 클래스', 'non-porosis 확률', 'porosis 확률']) 
    
    sm = nn.Softmax(dim=1)
    for idx, (samples, targets, paths) in tqdm(enumerate(data_loader)):
        # print(samples)
        # print(targets)
        outputs = model(samples.to(device))
        #targets = targets#.to(device)
        # print(targets)
        linear_outputs = postprocessors(outputs)
        
        linear_outputs = linear_outputs.detach().cpu()#.numpy()
        prob = sm(linear_outputs)[0]
        # print(prob[0], prob[1])
        # print(torch.argmax(prob,dim=0))
        wr.writerow([str(idx),paths[0],targets.item(),torch.argmax(prob,dim=0).item(),prob[0].item(),prob[1].item()])
        
        # if criterion is not None:
        #     loss = criterion(outputs, targets)
        #     # metric_logger.update(loss=loss.item())

        # for k, metric in metrics.items():
        #     metric_inputs = postprocessors[k](outputs, targets)
        #     metric.update(**metric_inputs)
    f.close
    # metric_logger.synchronize_between_processes()
    # logger.info(f"Averaged stats: {metric_logger}")

    # stats = {k: metric.compute() for k, metric in metrics.items()}
    # metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # return metric_logger_stats, stats


# def all_gather_and_flatten(tensor_rank):
#     tensor_all_ranks = torch.empty(
#         distributed.get_global_size(),
#         *tensor_rank.shape,
#         dtype=tensor_rank.dtype,
#         device=tensor_rank.device,
#     )
#     tensor_list = list(tensor_all_ranks.unbind(0))
#     torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
#     return tensor_all_ranks.flatten(end_dim=1)

@torch.inference_mode()
def evaluate_2stage(
    model1: nn.Module,
    model2: nn.Module,
    data_loader,
    postprocessors1,
    postprocessors2
):
    device = 'cuda'
    model1.eval()
    model2.eval()
    postprocessors1.eval()
    postprocessors2.eval()
    
    f = open('inference_2stage.csv','w',newline='')
    wr = csv.writer(f)
    wr.writerow(['인덱스','imgpath','target','stage1:pred', 'non-porosis:prob', 'porosis:prob','|','stage2:pred','normal:prob','penia:prob']) 
    
    sm = nn.Softmax(dim=1)
    for idx, (samples, targets, paths) in tqdm(enumerate(data_loader)):

        ## non-porosis vs porosis
        outputs1 = model1(samples.to(device))
        linear_outputs1 = postprocessors1(outputs1)
        
        linear_outputs1 = linear_outputs1.detach().cpu()#.numpy()
        prob1 = sm(linear_outputs1)[0]

        path = paths[0]
        if torch.argmax(prob1,dim=0).item()==1:
            st1_pred = 2
        else:
            st1_pred = torch.argmax(prob1,dim=0).item()

        ## normal vs penia
        if (torch.argmax(prob1,dim=0) == 0):
            outputs2 = model2(samples.to(device))
            linear_outputs2 = postprocessors2(outputs2)
            
            linear_outputs2 = linear_outputs2.detach().cpu()#.numpy()
            prob2 = sm(linear_outputs2)[0]
            st2_pred = torch.argmax(prob2,dim=0).item()

        else:
            st2_pred = np.nan
            prob2 = torch.Tensor([np.nan, np.nan])

        wr.writerow([str(idx),path,targets.item(),st1_pred,prob1[0].item(),prob1[1].item(),
                     ' ',st2_pred, prob2[0].item(),prob2[1].item()])
    f.close

@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    # metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    # logger.info(f"Features shape: {tuple(features.shape)}")
    # logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels


@torch.inference_mode()
def evaluate_reg(
    model: nn.Module,
    data_loader,
    # postprocessors: Dict[str, nn.Module],
    # metrics: Dict[str, MetricCollection],
    postprocessors,
    
    # criterion: Optional[nn.Module] = None,
):
    device = 'cuda'
    model.eval()
    postprocessors.eval()
    # if criterion is not None:
    #     criterion.eval()

    # for metric in metrics.values():
    #     metric = metric.to(device)

    # metric_logger = MetricLogger(delimiter="  ")
    # header = "Test:"
    
    f = open('inference_age.csv','w',newline='')
    wr = csv.writer(f)
    wr.writerow(['인덱스','타겟','예측']) 
    
    for idx, (samples, targets) in tqdm(enumerate(data_loader)):

        outputs = model(samples.to(device))

        linear_outputs = postprocessors(outputs)
        
        linear_outputs = linear_outputs.detach().cpu()#.numpy()

        wr.writerow([str(idx),targets.item(),linear_outputs.item()])

    f.close

@torch.inference_mode()
def evaluate_multi(
    model: nn.Module,
    data_loader,
    postprocessors,
    # criterion: Optional[nn.Module] = None,
):
    device = 'cuda'
    model.eval()
    postprocessors.eval()

    f = open('inference_3cls.csv','w',newline='')
    wr = csv.writer(f)
    wr.writerow(['인덱스','타겟 클래스','예측 클래스', 'normal_prob', 'penia_prob','porosis_prob'])

    sm = nn.Softmax(dim=1)
    for idx, (samples, targets) in tqdm(enumerate(data_loader)):
        outputs = model(samples.to(device))
        linear_outputs = postprocessors(outputs)

        linear_outputs = linear_outputs.detach().cpu()#.numpy()
        prob = sm(linear_outputs)[0]
        
        wr.writerow([str(idx),targets.item(),torch.argmax(prob,dim=0).item(),prob[0].item(),prob[1].item(),prob[2].item()])

    f.close

