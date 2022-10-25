import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torchattacks
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

def sample_configs(choices, is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False):

    config = {}
    depth = choices['depth']

    if is_visual_prompt_tuning == False and is_adapter == False and is_LoRA == False and is_prefix==False:
        visual_prompt_depth = random.choice(choices['visual_prompt_depth'])
        lora_depth = random.choice(choices['lora_depth'])
        adapter_depth = random.choice(choices['adapter_depth'])
        prefix_depth = random.choice(choices['prefix_depth'])
        config['visual_prompt_dim'] = [random.choice(choices['visual_prompt_dim']) for _ in range(visual_prompt_depth)] + [0] * (depth - visual_prompt_depth)
        config['lora_dim'] = [random.choice(choices['lora_dim']) for _ in range(lora_depth)] + [0] * (depth - lora_depth)
        config['adapter_dim'] = [random.choice(choices['adapter_dim']) for _ in range(adapter_depth)] + [0] * (depth - adapter_depth)
        config['prefix_dim'] = [random.choice(choices['prefix_dim']) for _ in range(prefix_depth)] + [0] * (depth - prefix_depth)

    else:
        if is_visual_prompt_tuning:
            config['visual_prompt_dim'] = [choices['super_prompt_tuning_dim']] * (depth)
        else:
            config['visual_prompt_dim'] = [0] * (depth)
        
        if is_adapter:
             config['adapter_dim'] = [choices['super_adapter_dim']] * (depth)
        else:
            config['adapter_dim'] = [0] * (depth)

        if is_LoRA:
            config['lora_dim'] = [choices['super_LoRA_dim']] * (depth)
        else:
            config['lora_dim'] = [0] * (depth)

        if is_prefix:
            config['prefix_dim'] = [choices['super_prefix_dim']] * (depth)
        else:
            config['prefix_dim'] = [0] * (depth)
        
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False, classes=100, gen_model=None):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)
    
    y_out = []
    y_pred = []
    y_true = []
    eps = 0.01/255
    alpha = 0.02/255
    steps = 1

    # atk = torchattacks.APGDT(model, eps=0.1/255, steps=1, n_classes=classes)
    # atk = torchattacks.Jitter(model, eps=0.1/255, alpha=0.2/255, steps=1)
    # atk.set_mode_targeted_least_likely(1)
    # atk.set_normalization_used(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # sample random config
        if mode == 'baseline':
            config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
            # print("current iter config: {}".format(config))
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
            gen_model_module = unwrap_model(gen_model)
            gen_model_module.set_sample_config(config=config)
        elif mode == 'super':
            # sample
            config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
            # print("current iter config: {}".format(config))
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
            gen_model_module = unwrap_model(gen_model)
            gen_model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            outputs = model(samples)

            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()
        
        y_out.extend(outputs.detach().cpu().numpy())
        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs) # Save Prediction
        labels = targets.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()
        # Feed forward and backward for syn images
        image_syn = syn_image(samples, targets, gen_model, device)
        label_left = targets[0: samples.shape[0]//2, ...]

        outputs = model(image_syn)
        if teacher_model:
            with torch.no_grad():
                teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
            loss_syn = 1 / 2 * criterion(outputs, label_left) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
        else:
            loss_syn = criterion(outputs, label_left)
        loss_syn_value = loss_syn.item()
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss_syn, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss_syn.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_syn=loss_syn_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, y_pred, y_true, y_out

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def match_loss(gw_syn, gw_real, device, dis_metric):
    dis = torch.tensor(0.0).to(device)

    if dis_metric == 'wb':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'% dis_metric)

    return dis

def syn_image(samples, labels, model, device):
    labels = labels.max(1)[1]
    batch_size = samples.shape[0]
    image_left = samples[0: batch_size//2, ...]
    image_right = samples[batch_size//2: , ...]
    label_left = labels[0: batch_size//2, ...]
    label_right = labels[batch_size//2: , ...]
    net_parameters = []
    for param in list(model.parameters()):
        if param.requires_grad:
            net_parameters.append(param)
    
    # import ipdb
    # ipdb.set_trace()
    image_syn = nn.Parameter(image_left.clone(), requires_grad=True)
    optimizer_img = torch.optim.SGD([image_syn, ], lr=0.1, momentum=0.5) # optimizer_img for synthetic data
    optimizer_img.zero_grad()
    criterion = nn.CrossEntropyLoss().to(device)

    # loss = torch.tensor(0.).to(device)
    for i in range(1):
        output_left = model(image_syn)
        loss_left = criterion(output_left, label_left)
        wg_left = torch.autograd.grad(loss_left, net_parameters, create_graph=True)
        
        output_right = model(image_right)
        loss_right = criterion(output_right, label_right)
        wg_right = torch.autograd.grad(loss_right, net_parameters)
        wg_right = list((_.detach().clone() for _ in wg_right))
        # print(wg_right.shape)
        loss = match_loss(wg_left, wg_right, device, 'wb')
    
        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
    diff = (image_syn.data - image_left.data)
    # abs_diff = abs(diff[0])
    print(torch.mean(torch.abs(diff[0])))
    print('*' * 100)
    print(image_left.max(), image_left.min())
    print('*' * 100)

    print(image_left.shape)
    print(IMAGENET_DEFAULT_MEAN)
    image_left_orig = image_left[0,:,:,:].detach().cpu()* torch.tensor(IMAGENET_DEFAULT_STD)[:,None,None] + torch.tensor(IMAGENET_DEFAULT_MEAN)[:,None,None]
    diff_orig = diff[0].detach().cpu()
    plt.imshow(abs(diff_orig.permute(1,2,0).detach().cpu().numpy())/abs(diff_orig.permute(1,2,0).detach().cpu().numpy()).max())
    print(diff_orig)
    plt.savefig('diff')
    plt.close()
    plt.imshow(abs(image_left_orig.permute(1,2,0).detach().cpu().numpy()))
    plt.savefig('original')
    exit()
    # model.zero_grad()
    # for param in list(model.parameters()):
    #     if param.requires_grad:
    #         print(param.grad)
    return image_syn.data

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=False)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))
    
    y_out = []
    y_pred = []
    y_true = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        y_out.extend(output.cpu().numpy())
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        labels = target.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # classes = np.arange(0, 37)
    # cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
    #                     columns = [i for i in classes])
    # plt.figure(figsize = (36,12))
    # sn.heatmap(df_cm, annot=True)
    # plt.savefig('./saves/output.png')
    # plt.close()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, y_pred, y_true, y_out
