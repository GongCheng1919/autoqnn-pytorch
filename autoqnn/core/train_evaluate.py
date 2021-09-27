import torch
from torch import nn
import torch.distributed as dist
import numpy as np
import time
from ..datasets.base import data_prefetcher

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.reset()       # __init__():reset parameters

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))            # args.lr = 0.1 , 即每30步，lr = lr /10
    for param_group in optimizer.param_groups:       # 将更新的lr 送入优化器 optimizer 中，进行下一次优化
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def top1(output, target):
    return accuracy(output, target,topk=(1,))[0]

def top5(output, target):
    return accuracy(output, target,topk=(5,))[0]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def validate_on_batch(val_loader, model, criterion, metric_meds=[]):
    input, target = next(iter(val_loader))
    input_var = torch.autograd.Variable(input, requires_grad=False).cuda()
    target_var = torch.autograd.Variable(target, requires_grad=False).cuda()
    model.eval()
    end = time.time()
    output = model(input_var)
    loss = criterion(output, target_var)
    metric_vals = [0 for _ in range(len(metric_meds))]
    metric_res = [met(output.data, target_var) for met in metric_meds]
    metric_list=["" for _ in range(len(metric_res))]
    for ind, res in enumerate(metric_res):
        metric_vals[ind]=res[0]
        metric_list[ind]="{met_name} {met:.3f}".format(met_name=metric_meds[ind].__name__,met=metric_vals[ind])
    metric_str="\t".join(metric_list)
    batch_time=time.time() - end
    print('Time {batch_time:.3f}\t'
          'Loss {loss:.4f} \t'
          '{metric_str}'.format(
           batch_time=batch_time,loss=loss, metric_str=metric_str))
    del input
    del target
    return metric_vals
    
def validate(val_loader, model, criterion, metric_meds=[], print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    metric_vals = [AverageMeter() for _ in range(len(metric_meds))]
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input.cuda(), requires_grad=False)
        target_var = torch.autograd.Variable(target.cuda(), requires_grad=False)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
#         prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        metric_res = [met(output.data, target_var) for met in metric_meds]
        metric_list=[""]*len(metric_res)
        for ind, res in enumerate(metric_res):
            metric_vals[ind].update(res[0],input.size(0))
            metric_list[ind]="{met_name} {met.val:.3f} ({met.avg:.3f})".format(met_name=metric_meds[ind].__name__,met=metric_vals[ind])
        metric_str="\t".join(metric_list)
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  '{metric_str}'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, metric_str=metric_str))
    metric_list=[""]*len(metric_res)
    for ind, res in enumerate(metric_vals):
        metric_list[ind]="{met_name} {met.avg:.3f}".format(met_name=metric_meds[ind].__name__,met=metric_vals[ind])
    print(' * {metric_str}'.format(metric_str="\t".join(metric_list)))

    return [met.avg for met in metric_vals] 

def train_on_batch(train_loader, model, criterion, optimizer, metric_meds=[]):
    input, target = next(iter(train_loader))
    input_var = torch.autograd.Variable(input, requires_grad=False).cuda()
    target_var = torch.autograd.Variable(target, requires_grad=False).cuda()
    model.train()
    end = time.time()
    # compute output
    output = model(input_var)
    # criterion 为定义过的损失函数
    loss = criterion(output, target_var)
    
    # measure accuracy and record loss
    metric_vals = [0 for _ in range(len(metric_meds))]
    metric_res = [met(output.data, target_var) for met in metric_meds]
    metric_list=["" for _ in range(len(metric_res))]
    for ind, res in enumerate(metric_res):
        metric_vals[ind]=res[0]
        metric_list[ind]="{met_name} {met:.3f}".format(met_name=metric_meds[ind].__name__,met=metric_vals[ind])
    metric_str="\t".join(metric_list)

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time=time.time() - end

    print('Time {batch_time:.3f}\t'
          'Loss {loss:.4f}\t'
          '{metric_str}'.format(
          batch_time=batch_time,loss=loss, metric_str=metric_str))
    del input
    del target
    return metric_vals


def train(train_loader, model, criterion, optimizer, epoch=1,metric_meds=[], print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    metric_vals = [AverageMeter() for _ in range(len(metric_meds))]
    print_freq=print_freq
    iteration = 0
    # switch to train mode
    model.train()
    prefetcher = data_prefetcher(train_loader)
    end = time.time()
    input, target = prefetcher.next()
#     for i, (input, target) in enumerate(train_loader):
    while iteration<len(train_loader):
        iteration += 1
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = torch.autograd.Variable(input.cuda(), requires_grad=False)
        target_var = torch.autograd.Variable(target.cuda(), requires_grad=False)

        # compute output
        output = model(input_var)
        # criterion 为定义过的损失函数
        loss = criterion(output, target_var)        

        # measure accuracy and record loss
        metric_res = [met(output.data, target_var) for met in metric_meds]
        metric_list=[""]*len(metric_res)
        for ind, res in enumerate(metric_res):
            metric_vals[ind].update(res[0],input.size(0))
            metric_list[ind]="{met_name} {met.val:.3f} ({met.avg:.3f})".format(met_name=metric_meds[ind].__name__,met=metric_vals[ind])
        metric_str="\t".join(metric_list)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:     # default=10
            print('\rEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  '{metric_str}'.format(
                   epoch, iteration, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, metric_str=metric_str),end="")
        
        input, target = prefetcher.next()
