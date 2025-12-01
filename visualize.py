import os
import torch
from dataset import get_data_transforms
import numpy as np
from torch.utils.data import DataLoader
from reason import get_lns_layer, get_bn_layer, get_wide_resnet50_2, get_de_wide_resnet50_2
from old_reason.reason import get_lns_layer as GLL
from old_reason.reason import get_wide_resnet50_2 as GWR
from old_reason.reason import get_de_wide_resnet50_2 as GDR
# from old_reason.reason import get_bn_layer as GBN
from rd4ad_de_resnet import de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter
import cv2
import torchvision.transforms as transforms

__all__ = ['cv2heatmap', 'heatmap_on_image', 'min_max_norm', 'save_anomaly_map']


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def create_folders(tag_path):
    if not os.path.exists(tag_path):
        os.makedirs(tag_path)


def cv2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def save_anomaly_map(anomaly_map, input_img, mask, file_path):
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))

    anomaly_map_norm = min_max_norm(anomaly_map)
    heatmap = cv2heatmap(anomaly_map_norm * 255)

    heatmap_on_img = heatmap_on_image(heatmap, input_img)
    create_folders(file_path)

    cv2.imwrite(os.path.join(file_path, 'input.jpg'), input_img)
    cv2.imwrite(os.path.join(file_path, 'heatmap.jpg'), heatmap)
    cv2.imwrite(os.path.join(file_path, 'heatmap_on_img.jpg'), heatmap_on_img)
    cv2.imwrite(os.path.join(file_path, 'mask.jpg'), mask * 255)


def inv_transform_to_numpy(tensor):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    # 逆向操作：取消标准化
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean_train, std_train)],
        std=[1 / s for s in std_train]
    )
    # 取消标准化
    tensor = inv_normalize(tensor)
    # 确保像素值在 [0, 1] 之间
    tensor = torch.clamp(tensor, 0, 1)
    np_img = tensor.permute(1, 2, 0).cpu().numpy()
    # 将像素值从 [0, 1] 转换为 [0, 255]
    np_img = (np_img * 255).astype(np.uint8)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return np_img


def visualize_anomaly_map(exp_name, _class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '/media/eachgood/EAA2E7E5A2E7B3EB/Datasets/datasets/anomaly_detection/mvtec2d/' + _class_
    ckp_path = './checkpoints/' + exp_name + '/wres50_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder = get_wide_resnet50_2(pretrained=True).to(device)
    lns_layer = get_lns_layer().to(device)
    bottleneck = get_bn_layer().to(device)
    decoder = get_de_wide_resnet50_2(pretrained=False, s_arc='t-like').to(device)
    encoder.eval()

    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bottleneck'].items()):
        if 'memory' in k:
            ckp['bottleneck'].pop(k)

    lns_layer.load_state_dict(ckp['lns_layer'])
    decoder.load_state_dict(ckp['decoder'])
    bottleneck.load_state_dict(ckp['bottleneck'])
    bottleneck.eval()
    lns_layer.eval()
    decoder.eval()

    with torch.no_grad():
        i = 0
        for img, gt, label, defect_name in test_dataloader:
            i += 1
            img = img.to(device)
            T_inputs = encoder(img)
            T_lns_inputs = lns_layer(T_inputs)
            compact_feat = bottleneck(T_lns_inputs)
            S_outputs = decoder(compact_feat)

            anomaly_map, _ = cal_anomaly_map(T_inputs, S_outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            if label[0].item() == 1:
                save_anomaly_map(anomaly_map, torch.Tensor((inv_transform_to_numpy(img[0]))).to('cpu').numpy(),
                                 gt[0].permute(1, 2, 0).numpy(),
                                 './results/' + exp_name + '/' + _class_ + '/'
                                 + str(defect_name[0]) + '/' + str(i))

def visualize_anomaly_map_rd4ad(exp_name, _class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '/media/eachgood/EAA2E7E5A2E7B3EB/Datasets/datasets/anomaly_detection/mvtec2d/' + _class_
    ckp_path = '/media/eachgood/EAA2E7E5A2E7B3EB/python_projects/ADVault/REASON-main/checkpoints/naive_rd4ad/wres50_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder = GWR(pretrained=True).to(device)
    bn = GLL().to(device)
    decoder = GDR(pretrained=False, s_arc='t-like').to(device)
    encoder.eval()

    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)

    bn.load_state_dict(ckp['bn'], strict=False)
    decoder.load_state_dict(ckp['decoder'], strict=False)
    bn.eval()
    decoder.eval()

    with torch.no_grad():
        i = 0
        for img, gt, label, defect_name in test_dataloader:
            i += 1
            img = img.to(device)
            T_inputs = encoder(img)
            S_outputs = decoder(bn(T_inputs))

            anomaly_map, _ = cal_anomaly_map(T_inputs, S_outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            if label[0].item() == 1:
                save_anomaly_map(anomaly_map, torch.Tensor((inv_transform_to_numpy(img[0]))).to('cpu').numpy(),
                                 gt[0].permute(1, 2, 0).numpy(),
                                 './results/' + exp_name + '/' + _class_ + '/'
                                 + str(defect_name[0]) + '/' + str(i))


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood',
                 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor',
                 'zipper']
    # exp_name = 'rd4ad'
    exp_name = 'reason_gs_0.015_1'

    for item in item_list:
        print(item)
        # if item in ['carpet', 'transistor']:
        #     continue
        # visualize_anomaly_map_rd4ad(exp_name, item)
        visualize_anomaly_map(exp_name, item)

    # visualize_anomaly_map(exp_name, 'transistor')
