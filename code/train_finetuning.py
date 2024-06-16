import os
import gc
import pdb
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
import nibabel as nib
from scipy import ndimage
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA/processed_h5_rdm_4/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='name', help='model_name')
parser.add_argument('--dataset', type=str, default='la', help='dataset to use')
parser.add_argument('--label_num', type=int, default=16, help='number of labeled data')

parser.add_argument('--pretrain_model', type=str, default='vit_b', help='vit to select')
parser.add_argument('--patch_size', type=int, default=128, help='shape of data')
parser.add_argument('--input_size', type=int, default=1024, help='shape of data')
parser.add_argument('--num_classes', type=int, default=1, help='number of class')
parser.add_argument('--save_img', type=int, default=250, help='img saving iterations')
# load
parser.add_argument('--load', action="store_true", help='load reg & seg net')
parser.add_argument('--load_iter', type=int, default=0, help='load iter')

parser.add_argument('--save_iter', type=int, default=1000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')

parser.add_argument('--nt', type=int, default=2, help='nonlinear_transformation')
parser.add_argument('--nonlinear_rate', type=float, default=0.5, help='nonlinear_rate')
parser.add_argument('--rdmrotflip', action="store_true", help='rdmrotflip')

parser.add_argument("--lr_sam", type=float, default=0.001, help="sam learning rate")
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')

parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
args = parser.parse_args()

root = "../"

train_data_path = args.root_path
snapshot_path = root + "model_" + args.dataset + "/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size
n_gpu = len(args.gpu.split(','))
print(batch_size)
max_iterations, input_size, patch_size = args.max_iterations, args.input_size, args.patch_size
num_classes = args.num_classes
lr_sam = args.lr_sam

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.modules.loss import CrossEntropyLoss

from dataloaders.dataset import *
from segment_anything_lora import sam_model_registry as sam_model_registry_lora
from sam_lora_image_encoder import LoRA_Sam
from utils import ramps, losses
from utils.util import *

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + 'saveimg')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.dataset == 'la':
        db_train = LAHeart(base_dir=train_data_path,
                           num=args.label_num,
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               # RandomCrop((patch_size, patch_size, 80)),
                               RandomCrop((112, 112, 80)),
                               ToTensor(),
                           ]))

    elif args.dataset == 'btcv':
        db_train = BTCV(base_dir=train_data_path,
                        num=args.label_num,
                        transform=transforms.Compose([
                            RandomCrop((patch_size, patch_size, patch_size)),
                            ToTensor(),
                        ]))

    elif args.dataset == 'mact':
        db_train = MACT(base_dir=train_data_path,
                        num=args.label_num,
                        transform=transforms.Compose([
                            RandomCrop((patch_size, patch_size, patch_size)),
                            ToTensor(),
                        ]))

    elif args.dataset == 'brats':
        db_train = BraTS19(base_dir=train_data_path,
                           num=args.label_num,
                           transform=transforms.Compose([
                              RandomRotFlip(),
                              RandomCrop((patch_size, patch_size, patch_size)),
                              ToTensor(),
                          ]))

    multimask_output = True if num_classes > 2 else False
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    if args.pretrain_model == "vit_b":
        model_sam, img_embedding_size = sam_model_registry_lora["vit_b"](image_size=args.input_size, num_classes=num_classes, checkpoint='pre_weight/sam_vit_b_01ec64.pth', pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1])
    pkg = import_module(args.module)
    model_sam = pkg.LoRA_Sam(model_sam, args.rank).cuda()
    if args.load:
        sam_checkpoint = snapshot_path + "/sam_iter_" + str(args.load_iter) + ".pth"
        model_sam.load_lora_parameters(sam_checkpoint)
        print("init weight from {}".format(sam_checkpoint))
    if args.warmup:
        base_lr_sam = lr_sam / args.warmup_period
    else:
        base_lr_sam = lr_sam
        optimizer_sam = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_sam.parameters()), lr=base_lr_sam, betas=(0.9, 0.999), weight_decay=0.1)
    model_sam.train()

    # Set losses
    ce_loss = CrossEntropyLoss(ignore_index=255)
    dice_loss = losses.DiceLoss(num_classes+1)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = args.load_iter
    max_epoch = (max_iterations - args.load_iter) // len(trainloader) + 1
    kl_distance = torch.nn.KLDivLoss(reduction='none')
    lr_ = base_lr_sam

    for epoch_num in range(max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']      # [2, 1, 128, 128, 64], [2, 128, 128, 64]

            ### Train SAM Module
            sam_volume_batch = volume_batch.cpu().detach().numpy()
            sam_label_batch = label_batch.cpu().detach().numpy()

            ## labeled data
            image = sam_volume_batch       # [B, 1, 128, 128, 64]
            label = sam_label_batch        # [B, 128, 128, 64]

            image_inshape, label_inshape, n_row, n_col, pw, ph, ps, pww, phh, s_l, s_w = Spread_bs_aug(image, label, input_size, args.nt, args.nonlinear_rate)  # (B, 1024, 1024)

            if args.rdmrotflip:
                # 2d RandomRotFlip
                k = np.array([np.random.randint(0, 4) for _ in range(image_inshape.shape[0])])
                axis = np.array([np.random.randint(0, 2) for _ in range(image_inshape.shape[0])])
                for i in range(image_inshape.shape[0]):
                    image_inshape[i] = RandomRotFlip_2d(image_inshape[i], k[i], axis[i])
                    label_inshape[i] = RandomRotFlip_2d(label_inshape[i], k[i], axis[i])

            volume_batch_inshape, label_batch_inshape = ToTensor_sam_bs(image_inshape, label_inshape)  # [B, 3, 1024, 1024], [B, 1024, 1024]

            outputs = model_sam(volume_batch_inshape, multimask_output, input_size)
            output_masks, low_res_masks, iou_predictions = outputs['masks'], outputs['low_res_logits'], outputs['iou_predictions']     # [1, C=2/14, 1024, 1024], [1, 2, 256, 256], [1, 2]
            output_soft = F.softmax(output_masks, dim=1)                    # [1, 2, 1024, 1024]

            ## calculate the loss - labeled data
            loss_ce_sam = ce_loss(output_masks, label_batch_inshape)
            loss_dice_sam = dice_loss(output_soft, label_batch_inshape.unsqueeze(1))
            loss_sam = 0.5 * (loss_ce_sam + loss_dice_sam)
            optimizer_sam.zero_grad()
            loss_sam.backward()
            optimizer_sam.step()

            if iter_num % args.save_img == 0:
                prediction_inshape = torch.argmax(output_soft, dim=1)  # (B, 1024, 1024)
                prediction_inshape = prediction_inshape.cpu().detach().numpy()  # (B, 1024, 1024)

                nib.save(nib.Nifti1Image(image_inshape.astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/inshape_img_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(label_inshape.astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/inshape_gt_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(prediction_inshape.astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/inshape_pred_' + str(iter_num) + '_u.nii.gz')

                if args.rdmrotflip:
                    for i in range(image_inshape.shape[0]):
                        prediction_inshape[i] = RandomFlipRot_2d(prediction_inshape[i], k[i]*(-1), axis[i]*(-1))

                pred_single = np.zeros((batch_size, image.shape[-1] + ps * 2, image.shape[-2] + pw * 2, image.shape[-3] + ph * 2))  # [B, 80, 112, 112]  (1, 112, 114, 86)
                if pww < 0 or phh < 0:
                    pww_r, phh_r = pww, phh
                    prediction_inshape = np.pad(prediction_inshape, [(0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=255)
                    pww, phh = 0, 0
                for row in range(n_row):
                    for col in range(n_col):
                        if row * n_col + col < pred_single.shape[1]:
                            pred_single[:, row * n_col + col] = prediction_inshape[:, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w]
                pred_single = pred_single[:, ps:pred_single.shape[-3] - ps, pw:pred_single.shape[-2] - pw, ph:pred_single.shape[-1] - ph]  # (B, 64, 128, 128)
                pred_single = np.swapaxes(pred_single, -3, -1)  # (B, 128, 128, 64)
                prediction_sam = pred_single

                nib.save(nib.Nifti1Image(image[0,0].astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/img_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(label[0].astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/gt_' + str(iter_num) + '.nii.gz')
                nib.save(nib.Nifti1Image(prediction_sam[0].astype(np.float32), np.eye(4)), snapshot_path + '/saveimg' + '/pred_sam_' + str(iter_num) + '_u.nii.gz')

            logging.info('iter %d : sam loss : %f, ce loss : %f, dice loss : %f' % (iter_num, loss_sam, loss_ce_sam, loss_dice_sam.item()))

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_sam', loss_sam, iter_num)

            ## change lr
            if args.warmup and iter_num < args.warmup_period:
                lr_ = lr_sam * ((iter_num + 1) / args.warmup_period)
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period          # assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = lr_sam * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
            for param_group in optimizer_sam.param_groups:
                param_group['lr'] = lr_

            if iter_num % args.save_iter == 0:
                save_mode_path_sam = os.path.join(snapshot_path, 'sam_iter_' + str(iter_num) + '.pth')
                model_sam.save_lora_parameters(save_mode_path_sam)
                logging.info("save model to {}".format(save_mode_path_sam))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break
    save_mode_path_sam = os.path.join(snapshot_path, 'sam_iter_' + str(iter_num) + '.pth')
    model_sam.save_lora_parameters(save_mode_path_sam)
    logging.info("save model to {}".format(save_mode_path_sam))
    writer.close()


