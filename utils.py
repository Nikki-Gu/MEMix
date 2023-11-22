import argparse
import sys
import random
import os, torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from enum import Enum
import torch.nn as nn
import math
from timm.data.mixup import  one_hot
def seed_torch(seed=2):
    print('seed=',seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='datasets/CASME2', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=1000,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--batch_size', type=int, default=34, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=7000, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    parser.add_argument('--patchup_prob', type=float, default=.7, help='PatchUp probability')

    return parser.parse_args()

def To_img(tensor_image):
    numpy_image = tensor_image.mul(255).cpu().numpy()
    integer_image = numpy_image.astype(np.uint8)
    hwc_image = np.transpose(integer_image, (1, 2, 0))
    return hwc_image

class Logger(object):
    def __init__(self,log_name):
        self.terminal = sys.stdout
        self.log_name=log_name
        self.log = open(self.log_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass

def transmix_label(label1, label2,mask ,attn,mixup_lam,device='cuda',num_classes=5,smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    target1 = one_hot(label1, num_classes, on_value=on_value, off_value=off_value, device=device)
    target2 = one_hot(label2, num_classes, on_value=on_value, off_value=off_value, device=device)
    B=len(attn)
    mask = mask.view(B, -1)#.repeat(len(attn), 1)
    w1, w2 = torch.sum(mask * attn, dim=1)+torch.sum((1-mask)*attn,dim=1)*mixup_lam, torch.sum((1-mask) * attn, dim=1)*(1-mixup_lam)
    lam = w1 / (w1 + w2)  # (b, )
    #print('lam:',lam)
    target = target1 * lam.unsqueeze(1) + target2 * (1. -lam).unsqueeze(1)
    return target

def generate_flow_mask(x, lam_alpha=1.0, mask_token_num_start=14, min_num_patches=1,type='avg',width=14,height=14):
    # input: x[B,C,224,224],output:mask[B,C,14,14],lam[B,]
    B, C, _, _ = x.shape
    mask = np.ones(shape=(B, width, height), dtype=np.int64)
    lam_list = []
    min_aspect = 0.3
    log_aspect_ratio = (math.log(min_aspect), math.log(1 / min_aspect))
    max_num_patches = width * height  # total patch num
    all_flow = torch.abs(x).sum(axis=1)  # [B,224,224]
    if type=='avg':
        avg_pooling = nn.AvgPool2d(kernel_size=int(224/width))
    elif type=='max':
        avg_pooling = nn.MaxPool2d(kernel_size=int(224/width))
    else:
        print('false')

    with torch.no_grad():
        all_flow = avg_pooling(all_flow)  # [B,14,14]
        all_flow = all_flow.cpu()
    # print('all_flow shape[b,14,14]:', all_flow.shape)
    for i in range(B):
        lam = np.random.beta(lam_alpha, lam_alpha)
        mask_ratio = lam
        num_masking_patches = min(width * height, int(mask_ratio * width * height) + mask_token_num_start)
        mask_count = 0
        # rank by patch
        flow_patch = all_flow[i, :, :]  # [14,14]
        idx = torch.argsort(torch.ravel(flow_patch), descending=True)
        idx = np.unravel_index(idx, flow_patch.shape)
        idx_list = np.column_stack(idx)
        patch_idx = 0
        while mask_count < num_masking_patches:
            center_idx=idx_list[patch_idx]
            mask[i,center_idx[0],center_idx[1]]=0
            mask_count+=1
            # print('patch idx:', patch_idx)
            # print('mask count:', mask_count)
            patch_idx += 1
        # lam_list.append(1.0 - mask_count / max_num_patches)
        lam_list.append(mask_count / max_num_patches)
    mask = torch.from_numpy(mask).float().cuda().unsqueeze(1)  
    return mask, torch.tensor(lam_list).cuda()  

def generate_flow_mask_random(x, lam_alpha=1.0,mask_token_num_start=14):
    B, C, _, _ = x.shape
    mask = np.ones(shape=(B, 14, 14), dtype=np.int)
    lam_list = []
    width=14
    height=14
    max_num_patches = width * height
    for i in range(B):
        tmp_mask=np.ones(shape=(14*14),dtype=np.int)
        lam = np.random.beta(lam_alpha, lam_alpha)
        patch_num= min(width * height, int(lam * width * height) + mask_token_num_start)
        mask_idx=np.random.permutation(14*14)[:patch_num]
        tmp_mask[mask_idx]=0
        tmp_mask=tmp_mask.reshape(14,14)
        mask[i,:,:]=tmp_mask
        lam_list.append(patch_num/max_num_patches)
    mask = torch.from_numpy(mask).float().cuda().unsqueeze(1)
    return mask,torch.tensor(lam_list).cuda()

def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    return Variable(y_onehot.cuda(), requires_grad=False)

def cutmix(x1,x2,target,indices,alpha,use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    W, H = x1.size(2), x1.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    x1[:, :, bbx1:bbx2, bby1:bby2] = x1[indices, :, bbx1:bbx2, bby1:bby2]
    x2[:, :, bbx1:bbx2, bby1:bby2] = x2[indices, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x1,x2,target,target[indices],lam

def mixup_data(x1,x2,y,alpha=1.0,use_cuda=True):
    if alpha>0:
        lam=np.random.beta(alpha,alpha)
    else:
        lam=1.
    batch_size=x1.size()[0]
    if use_cuda:
        index=torch.randperm(batch_size).cuda()
    else:
        index=torch.randperm(batch_size)

    mixed_x1=lam*x1+(1-lam)*x1[index,:]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    y_a,y_b=y,y[index]

    return mixed_x1,mixed_x2,y_a,y_b,lam

class TwoLabelSoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(TwoLabelSoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
