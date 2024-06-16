import os
import argparse
import torch
import numpy as np
from networks.hierarchical_vnet import VNet
from networks.hierarchical_unet_3d import UNet_3D
from networks.vnet_cct import VNet as VNet_cct
from networks.unet_3d_cct import UNet_3D as UNet_3D_cct
from test_util_singleclass_hv import test_all_case
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='sparse_mt_294', help='model_name')
parser.add_argument('--dataset', type=str,  default='la', help='dataset to use')
parser.add_argument('--data_version', type=str,  default='v2', help='dataset version to use')
parser.add_argument('--set_version', type=str,  default='0', help='dataset version to use')
parser.add_argument('--semantic_class', type=str, default='kidney', choices=['kidney', 'tumor'])
parser.add_argument('--list_num', type=str,  default='', help='data list to use')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--iteration', type=int,  default=6000, help='GPU to use')
parser.add_argument('--patch_size', type=int, default=112, help='patch size')
parser.add_argument('--model_type', type=str, default='vnet', help='model_type')
args = parser.parse_args()

root = "../"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = root + "model_" + args.dataset + "/" + args.model + "/"
test_save_path = root + "model_" + args.dataset + "/prediction/" + args.model + "_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

if args.dataset == 'la':
    with open(args.root_path + '/../test' + args.list_num + '.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
elif args.dataset == 'btcv':
    num_classes = 14
    with open(args.root_path + '/../test' + args.list_num + '.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + '/' + item.replace('\n', '') + ".h5" for item in image_list]
elif args.dataset == 'mact':
    num_classes = 9
    with open(args.root_path + '/../test' + args.list_num + '.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + '/' + item.replace('\n', '') + ".h5" for item in image_list]
elif args.dataset == 'brats':
    with open(args.root_path + '/../test_follow.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + '/' + item.replace('\n', '') + ".h5" for item in image_list]


def test_calculate_metric(epoch_num):
    if args.model_type == "vnet":
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False, pyramid_has_dropout=False).cuda()
    elif args.model_type == "unet_3d":
        net = UNet_3D(in_channels=1, n_classes=num_classes).cuda()
    elif args.model_type == "vnet_cct":
        net = VNet_cct(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    elif args.model_type == "unet_3d_cct":
        net = UNet_3D_cct(in_channels=1, n_classes=num_classes).cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    if args.dataset == 'la':
        if args.patch_size == 112:
            ps = (112, 112, 80)
        elif args.patch_size == 128:
            ps = (128, 128, 64)
        avg_metric = test_all_case(net, args.dataset, args.semantic_class, image_list, num_classes=num_classes, patch_size=ps,
                                   save_result=True, stride_xy=18, stride_z=4, test_save_path=test_save_path)

    elif args.dataset == 'btcv' or args.dataset == 'mact':
        patch_size = args.patch_size
        avg_metric = test_all_case(net, args.dataset, args.semantic_class, image_list, num_classes=num_classes, patch_size=(patch_size, patch_size, patch_size),
                                   save_result=True, stride_xy=12, stride_z=12, test_save_path=test_save_path)

    elif args.dataset == 'brats':
        patch_size = args.patch_size
        avg_metric = test_all_case(net, args.dataset, args.semantic_class, image_list, num_classes=num_classes, patch_size=(patch_size, patch_size, patch_size),
                                   save_result=True, stride_xy=64, stride_z=64, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric, std = test_calculate_metric(args.iteration)
    print(metric)
    with open(root + "model_" + args.dataset + "/prediction_v2.txt", "a") as f:
        f.write(args.model + " - " + str(args.iteration) + ": " + ", ".join(str(i) for i in metric) + "\n")
        f.write(args.model + " - " + str(args.iteration) + ": " + ", ".join(str(i) for i in std) + "\n")
