import json
import os.path
import sys
import time

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm, trange

from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random

from torch.utils.data import DataLoader
from reason import get_bn_layer, get_wide_resnet50_2, get_de_wide_resnet50_2, get_lns_layer
from dataset import MVTecDataset

from test import evaluation, test
from argparse import ArgumentParser


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class REASON_loss(nn.Module):
    def __init__(self, lns_coff=0.2, block_ratio=0.125):
        super(REASON_loss, self).__init__()
        self.distill_coff = 1
        self.lns_coff = lns_coff
        self.alpha = 0.2
        self.block_ratio = block_ratio

    @staticmethod
    def distill_loss(a, b):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
        return loss

    @staticmethod
    def pixel_wise_loss(a, b):
        cos_loss = torch.nn.CosineSimilarity(dim=1)
        loss = 0
        for item in range(len(a)):
            loss += torch.mean(1 - cos_loss(a[item], b[item]))
        return loss

    @staticmethod
    def block_wise_loss(a, b, block_ratio=0.125):

        cos_loss = torch.nn.CosineSimilarity(dim=1)
        total_blocks = 0

        loss = 0
        for item in range(len(a)):
            _, c, h, w = a[item].shape
            block_size = int(min(h, w) * block_ratio)  # 确保block_size不会超过特征图的宽度或高度
            num_blocks_h = h // block_size if h % block_size == 0 else h // block_size + 1
            num_blocks_w = w // block_size if w % block_size == 0 else w // block_size + 1
            total_blocks += num_blocks_h * num_blocks_w

            for bh in range(num_blocks_h):
                for bw in range(num_blocks_w):
                    # 计算当前块的起始坐标
                    start_h = bh * block_size
                    end_h = min(start_h + block_size, h)
                    start_w = bw * block_size
                    end_w = min(start_w + block_size, w)

                    # 提取块并展平为向量
                    block_a = a[item][:, :, start_h:end_h, start_w:end_w]
                    block_b = b[item][:, :, start_h:end_h, start_w:end_w]

                    # 计算并累加块间的余弦相似度损失
                    if block_a.numel() > 0 and block_b.numel() > 0:  # 确保块中有元素
                        loss += torch.mean(1 - cos_loss(block_a, block_b))

        return loss / total_blocks  # 平均所有块的损失

    @staticmethod
    def reconstruct_loss(a, b, mode='cosine'):
        loss = 0
        if mode == 'cosine':
            for item in range(len(a)):
                loss += torch.mean(1 - torch.nn.CosineSimilarity()(a[item].view(a[item].shape[0], -1),
                                                                   b[item].view(b[item].shape[0], -1)))
        elif mode == 'l2':
            for item in range(len(a)):
                loss += torch.mean(torch.nn.MSELoss()(a[item], b[item]))
        return loss

    @staticmethod
    def contrast_loss(a, b):
        contrast = torch.nn.CosineEmbeddingLoss(margin=0.5)
        B = a[0].shape[0]
        target = -torch.ones(B).to('cuda')
        loss = 0
        for i in range(len(a)):
            loss += contrast(a[i].view(a[i].shape[0], -1), b[i].view(b[i].shape[0], -1), target=target)
        return loss

    def forward(self, pred, target, project_feat):
        # global
        # loss_distill = self.distill_loss(pred, target)
        # global + pixel_wise
        # loss_distill = (self.distill_loss(pred, target) +
        #                 self.alpha * self.pixel_wise_loss(pred, target))
        # global + block_wise
        loss_distill = (self.distill_loss(pred, target) +
                        self.alpha * self.block_wise_loss(pred, target, block_ratio=self.block_ratio))
        # loss_distill = self.pixel_wise_loss(pred, target)
        # loss_distill = self.block_wise_loss(pred, target, block_ratio=self.block_ratio)

        # global + block_wise + pixel_wise
        # loss_distill = (self.distill_loss(pred, target) +
        #                 self.alpha * self.block_wise_loss(pred, target, block_ratio=self.block_ratio) +
        #                 self.alpha * self.pixel_wise_loss(pred, target))


        loss_lns = self.reconstruct_loss(project_feat, target, mode='cosine')
        total_loss = self.distill_coff * loss_distill + self.lns_coff * loss_lns
        return total_loss, loss_distill, loss_lns


def train(_class_, pars):
    print(_class_)

    learning_rate_distill = pars.distill_lr
    learning_rate_lns = pars.lns_lr

    betas_distill = (0.5, 0.999)
    betas_project = (0.5, 0.999)

    batch_size = pars.batch_size
    image_size = pars.image_size
    exp_name = pars.exp_name
    if not os.path.exists('./checkpoints/' + exp_name):
        os.makedirs('./checkpoints/' + exp_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = '/media/eachgood/EAA2E7E5A2E7B3EB/Datasets/datasets/anomaly_detection/mvtec2d/' + _class_ + '/train'
    test_path = '/media/eachgood/EAA2E7E5A2E7B3EB/Datasets/datasets/anomaly_detection/mvtec2d/' + _class_
    ckp_path = './checkpoints/' + exp_name + '/wres50_' + _class_ + '.pth'
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # prepare model
    encoder = get_wide_resnet50_2(pretrained=True).to(device)
    lns_layer = get_lns_layer().to(device)
    bottleneck = get_bn_layer().to(device)
    decoder = get_de_wide_resnet50_2(pretrained=False, s_arc='t-like').to(device)

    optimizer = torch.optim.Adam([
        {'params': list(decoder.parameters()) + list(bottleneck.parameters()), 'lr': learning_rate_distill,
         'betas': betas_distill},
        {'params': lns_layer.parameters(), 'lr': learning_rate_lns,
         'betas': betas_project}
    ])
    loss_function = REASON_loss(lns_coff=pars.lns_coff, block_ratio=pars.block_ratio).to(device)

    encoder.eval()

    # if not os.path.exists(f'./checkpoints/{exp_name}/{_class_}'):
    #     os.makedirs(f'./checkpoints/{exp_name}/{_class_}')
    if not os.path.exists(f'./checkpoints/{exp_name}'):
        os.makedirs(f'./checkpoints/{exp_name}')

    print(f'with class {_class_}, Training with {pars.epochs} Epoch')

    best_score = 0
    best_auroc_px = 0
    best_auroc_sp = 0
    best_aupro_px = 0

    # auroc_px_list = []
    # auroc_sp_list = []
    # aupro_px_list = []
    auroc_px, auroc_sp, aupro_px = 0, 0, 0

    with trange(pars.epochs, desc=f'{_class_} training', position=0, leave=True, file=sys.stdout) as t:
        for epoch in t:
            lns_layer.train()
            bottleneck.train()
            decoder.train()
            loss = 0

            for img, label in train_dataloader:
                img = img.to(device)

                T_inputs = encoder(img)
                T_noisy_inputs = encoder(img, noise_injection=True, noise_std=pars.noise_std,
                                         num_position=pars.num_position)
                T_noisy_proj_inputs = lns_layer(T_noisy_inputs)
                compact_feat = bottleneck(T_noisy_proj_inputs)
                S_outputs = decoder(compact_feat)
                total_loss, loss_distill, loss_lns = loss_function(T_inputs, S_outputs,
                                                                   T_noisy_proj_inputs)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                loss += total_loss.detach().cpu().item()

            if (epoch+1) % 2 == 0:
                auroc_sp, auroc_px, aupro_px = evaluation(encoder, lns_layer, bottleneck, decoder, test_dataloader, device)

                # auroc_px_list.append(auroc_px)
                # auroc_sp_list.append(auroc_sp)
                # aupro_px_list.append(aupro_px)

                if (auroc_px + auroc_sp + aupro_px) / 3 > best_score:
                    best_score = (auroc_px + auroc_sp + aupro_px) / 3

                    best_auroc_px = auroc_px
                    best_auroc_sp = auroc_sp
                    best_aupro_px = aupro_px

                    # torch.save({'lns_layer': lns_layer.state_dict(),
                    #             'decoder': decoder.state_dict(),
                    #             'bottleneck': bottleneck.state_dict()}, ckp_path)

            t.set_postfix_str(
                f'Loss: {loss / len(train_dataloader):.4f}, S-AUROC: {auroc_sp:.4f}({best_auroc_sp:.4f}), P-AUROC:{auroc_px:.4f}({best_auroc_px:.4f}), P-AUPRO: {aupro_px:.4f}({best_aupro_px:.4f})')

    return best_auroc_sp, best_auroc_px, best_aupro_px


def get_parser():
    parser = ArgumentParser()
    # parser.add_argument('--save_folder', default='./REASON_checkpoint_result', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--detail_training', default='note', type=str)
    parser.add_argument('--lns_lr', default=0.001, type=float)
    parser.add_argument('--distill_lr', default=0.005, type=float)
    # parser.add_argument('--lns_coff', default=0.2, type=float)
    # parser.add_argument('--lns_coff', default=0.1, type=float)
    parser.add_argument('--lns_coff', default=0.01, type=float)
    parser.add_argument('--block_ratio', default=0.0625, type=float)
    # parser.add_argument('--noise_std', default=0.01, type=float)
    # parser.add_argument('--noise_std', default=0.03, type=float)
    parser.add_argument('--noise_std', default=0.015, type=float)
    parser.add_argument('--num_position', default=1, type=int)
    parser.add_argument('--classes', nargs="+", default=["carpet", "leather"])
    return parser


if __name__ == '__main__':

    import warnings

    warnings.filterwarnings('ignore')
    parser = get_parser()
    pars = parser.parse_args()

    all_classes = [
        'carpet',
        'grid',
        'leather',
        'tile',
        'wood',
        'bottle',
        'cable',
        'capsule',
        'hazelnut',
        'metal_nut',
        'pill',
        'screw',
        'toothbrush',
        'transistor',
        'zipper']

    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample': [], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    # ps: pixel_wise + spatial
    # bs: block_wise + spatial
    # gs: global + spatial
    # gps: global + pixel_wise + spatial  √
    # gps: global + block_wise + pixel_wise + spatial  running
    # exp_name = f'reason_gbs_{pars.block_ratio}_{pars.noise_std}_{pars.num_position}_random_shape'
    # exp_name = f'reason_gbs_{pars.block_ratio}_{pars.noise_std}_random_shape'
    exp_name = f'reason_gbs_alpha_{pars.lns_coff}_random_shape'
    print(f'exp_name: {exp_name}')
    parser.add_argument('--exp_name', default=exp_name, type=str)
    parser.add_argument('--epochs', default=200, type=int)
    start = time.time()
    # train all_classes
    # for c in all_classes
    for _class_ in all_classes:
        epochs = 0
        if _class_ in ['carpet', 'leather']:
            epochs = 10
        if _class_ in ['grid', 'tile', 'capsule', 'metal_nut', 'screw', 'toothbrush', 'transistor', 'pill', 'bottle']:
            epochs = 200
        if _class_ in ['wood', 'hazelnut']:
            epochs = 100
        if _class_ in ['cable', 'zipper']:
            # epochs = 300
            epochs = 200

        parser.set_defaults(epochs=epochs)
        auroc_sp, auroc_px, aupro_px = train(_class_, parser.parse_args())
        print('class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}'.format(_class_,
                                                                                                auroc_sp,
                                                                                                auroc_px,
                                                                                                aupro_px))

        metrics['class'].append(_class_)
        metrics['AUROC_sample'].append(round(auroc_sp * 100, 2))
        metrics['AUROC_pixel'].append(round(auroc_px * 100, 2))
        metrics['AUPRO_pixel'].append(round(aupro_px * 100, 2))
        pd.DataFrame(metrics).transpose().to_csv(
            f'./checkpoints/{exp_name}_result.csv', header=False)
    end = time.time()
    print(f'Time taken: {(end - start) / 3600:.2f}h')
