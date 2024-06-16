import torch
import numpy as np
import SimpleITK as sitk
import pdb
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.bezier_curve import *
from dataloaders.dataset import resampling
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
from segment_anything.utils.transforms import ResizeLongestSide

def Spread_bs_aug(image, label, input_size, nt=0, nonlinear_rate=0.5, rdmrotflip=False, rdmcrop_2d=False, flip_2d=False, shuffle_2d=False):        # 每一张slice做bezier灰度augmentation
    spread_image = np.zeros((image.shape[0], input_size, input_size))     # (B, 1024, 1024)
    spread_label = np.zeros((image.shape[0], input_size, input_size)) + 255
    for i in range(image.shape[0]):
        # if nt == 3:
        #     rdm = random.random()
        #     t1, t2, t3, t4 = random.random(), random.random(), random.random(), random.random()

        single_image, single_label = image[i].squeeze(), label[i].squeeze()
        single_image, single_label = np.swapaxes(single_image, 0, 2), np.swapaxes(single_label, 0, 2)  # (64, 120, 120)
        if shuffle_2d:
            # 打乱顺序重新排列二维矩阵
            random_indices = np.random.permutation(64)
            # 对两个矩阵进行重新排列
            new_image = single_image[random_indices]
            new_label = single_label[random_indices]
            single_image, single_label = new_image, new_label

        for j in range(single_image.shape[0]):
            slice, slice_lbl = single_image[j], single_label[j]
            if flip_2d:
                k = np.random.randint(0, 4)
                image_slc = np.rot90(slice, k)
                label_slc = np.rot90(slice_lbl, k)
                axis = np.random.randint(0, 2)
                image_slc = np.flip(image_slc, axis=axis).copy()
                label_slc = np.flip(label_slc, axis=axis).copy()
                slice, slice_lbl = image_slc, label_slc
            if rdmcrop_2d:
                rdm = random.random()
                if rdm <= 0.33:
                    output_size = (int(slice.shape[0] * 0.8), slice.shape[1])
                    slice, slice_lbl = RandomCropResample_2d(slice, slice_lbl, output_size)
                elif rdm > 0.33 and rdm <= 0.66:
                    output_size = (slice.shape[0], int(slice.shape[1] * 0.8))
                    slice, slice_lbl = RandomCropResample_2d(slice, slice_lbl, output_size)
            if nt == 1:
                slice = nonlinear_transformation_r1(slice, nonlinear_rate)
            elif nt == 2:
                slice = nonlinear_transformation_r2(slice, nonlinear_rate)
            # elif nt == 3 and rdm >= 0.5:
            #     slice = nonlinear_transformation_r3(slice, t1, t2, t3, t4)
            elif nt == 4:
                slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))  # 0-1
                slice = nonlinear_transformation_r1(slice, nonlinear_rate)
            elif nt == 5:
                slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))  # 0-1
                slice = nonlinear_transformation_r2(slice, nonlinear_rate)
            elif nt == 6:
                slice = 2 * (slice - np.min(slice)) / (np.max(slice) - np.min(slice))  # -1-1
                slice = nonlinear_transformation_r4(slice)
            single_image[j], single_label[j] = slice, slice_lbl

        n_row, n_col = input_size // single_label.shape[1], input_size // single_label.shape[2]
        pw = round((input_size / n_row - single_label.shape[1])/2)  # (128-120)/2=4
        ph = round((input_size / n_col - single_label.shape[2])/2)
        ps = max((n_row * n_col - single_image.shape[0]) // 2, 0)

        single_image = np.pad(single_image, [(ps, ps), (pw, pw), (ph, ph)], mode='constant', constant_values=0)  # (64, 128, 128)
        single_label = np.pad(single_label, [(ps, ps), (pw, pw), (ph, ph)], mode='constant', constant_values=255)  # (64, 128, 128)
        s_l = single_label.shape[1]  # side length, 128/256/512
        s_w = single_label.shape[2]  # side width, 128/256/512
        pww = (input_size - single_image.shape[1] * n_row) // 2
        phh = (input_size - single_image.shape[2] * n_col) // 2
        if single_image.shape[0] < n_row * n_col:
            single_image = np.pad(single_image, [(0, n_row * n_col - single_image.shape[0]), (0, 0), (0, 0)], mode='constant', constant_values=0)
            single_label = np.pad(single_label, [(0, n_row * n_col - single_label.shape[0]), (0, 0), (0, 0)], mode='constant', constant_values=255)
        pww_r, phh_r = pww, phh
        if pww < 0:
            if i == 0:
                spread_image = np.pad(spread_image, [(0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=0)
                spread_label = np.pad(spread_label, [(0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=255)
            pww, phh = 0, 0

        for row in range(n_row):
            for col in range(n_col):
                spread_image[i, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w] = single_image[row * n_col + col]
                spread_label[i, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w] = single_label[row * n_col + col]
    if pww_r < 0:
        spread_image = spread_image[:, abs(pww_r):pww_r, abs(phh_r):phh_r]
        spread_label = spread_label[:, abs(pww_r):pww_r, abs(phh_r):phh_r]
    image, label = spread_image, spread_label  # (B, 1024, 1024)

    if nt >= 4:
        image = (image - np.mean(image)) / np.std(image)

    if rdmrotflip:
        # 2d RandomRotFlip
        k = np.array([np.random.randint(0, 4) for _ in range(image.shape[0])])
        axis = np.array([np.random.randint(0, 2) for _ in range(image.shape[0])])
        for i in range(image.shape[0]):
            image[i] = RandomRotFlip_2d(image[i], k[i], axis[i])
            label[i] = RandomRotFlip_2d(label[i], k[i], axis[i])
    return image, label, n_row, n_col, pw, ph, ps, pww_r, phh_r, s_l, s_w


def Spread_bs_aug_scale(image, label, input_size, scale, nt=0, nonlinear_rate=0.5):        # 每一张slice做bezier灰度augmentation
    spread_image = np.zeros((image.shape[0], input_size, input_size))     # (B, 1024, 1024)
    spread_label = np.zeros((image.shape[0], input_size, input_size)) + 255
    for i in range(image.shape[0]):
        single_image, single_label = image[i].squeeze(), label[i].squeeze()
        single_image, single_label = np.swapaxes(single_image, 0, 2), np.swapaxes(single_label, 0, 2)  # (64, 120, 120)
        for j in range(single_image.shape[0]):
            slice, slice_lbl = single_image[j], single_label[j]
            if nt == 2:
                slice = nonlinear_transformation_r2(slice, nonlinear_rate)
            single_image[j], single_label[j] = slice, slice_lbl

        # n_row, n_col = input_size // single_label.shape[1], input_size // single_label.shape[2]
        # pw = round((input_size / n_row - single_label.shape[1])/2)  # (128-120)/2=4
        # ph = round((input_size / n_col - single_label.shape[2])/2)
        # ps = max((n_row * n_col - single_image.shape[0]) // 2, 0)
        n_row, n_col = scale, scale # 8, 8
        n_slc = n_row * n_col       # 64
        start_index = np.random.randint(0, single_image.shape[0] - n_slc + 1)     # 80-64+1=17, [0,17)
        single_image = single_image[start_index:start_index + n_slc]                # n_slc, 112, 112
        single_label = single_label[start_index:start_index + n_slc]
        pw, ph, ps = 4, 4, 0
        image_shape = (1024//scale - pw*2, 1024//scale - pw*2, n_slc)

        image_itk = resampling(sitk.GetImageFromArray(single_image), image_shape, lbl=False)
        label_itk = resampling(sitk.GetImageFromArray(single_label), image_shape, lbl=True)
        single_image = sitk.GetArrayFromImage(image_itk)
        single_label = sitk.GetArrayFromImage(label_itk)                # 64, 120, 120

        single_image = np.pad(single_image, [(ps, ps), (pw, pw), (ph, ph)], mode='constant', constant_values=0)  # (64, 128, 128)
        single_label = np.pad(single_label, [(ps, ps), (pw, pw), (ph, ph)], mode='constant', constant_values=255)  # (64, 128, 128)
        s_l = single_label.shape[1]  # side length, 128/256/512
        s_w = single_label.shape[2]  # side width, 128/256/512
        pww = (input_size - single_image.shape[1] * n_row) // 2
        phh = (input_size - single_image.shape[2] * n_col) // 2
        if single_image.shape[0] < n_row * n_col:
            single_image = np.pad(single_image, [(0, n_row * n_col - single_image.shape[0]), (0, 0), (0, 0)], mode='constant', constant_values=0)
            single_label = np.pad(single_label, [(0, n_row * n_col - single_label.shape[0]), (0, 0), (0, 0)], mode='constant', constant_values=255)
        pww_r, phh_r = pww, phh
        if pww < 0:
            if i == 0:
                spread_image = np.pad(spread_image, [(0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=0)
                spread_label = np.pad(spread_label, [(0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=255)
            pww, phh = 0, 0

        for row in range(n_row):
            for col in range(n_col):
                spread_image[i, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w] = single_image[row * n_col + col]
                spread_label[i, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w] = single_label[row * n_col + col]
    if pww_r < 0:
        spread_image = spread_image[:, abs(pww_r):pww_r, abs(phh_r):phh_r]
        spread_label = spread_label[:, abs(pww_r):pww_r, abs(phh_r):phh_r]
    image, label = spread_image, spread_label  # (B, 1024, 1024)
    return image, label, n_row, n_col, pw, ph, ps, pww_r, phh_r, s_l, s_w


def ToTensor_sam_bs(image, label):     # [B, 1024, 1024]
    volume_batch_inshape, label_batch_inshape = image, label
    volume_batch_inshape = volume_batch_inshape.reshape(volume_batch_inshape.shape[0], 1, volume_batch_inshape.shape[1], volume_batch_inshape.shape[2]).astype(np.float32)  # [B, 1, 1024, 1024]
    label_batch_inshape = label_batch_inshape.reshape(label_batch_inshape.shape[0], label_batch_inshape.shape[1], volume_batch_inshape.shape[2]).astype(np.long)  # [B, 1024, 1024]
    volume_batch_inshape, label_batch_inshape = torch.from_numpy(volume_batch_inshape), torch.from_numpy(label_batch_inshape)  # [B, 1, 1024, 1024], [B, 1024, 1024]
    volume_batch_inshape = torch.cat((volume_batch_inshape, volume_batch_inshape, volume_batch_inshape), 1)  # [B, 3, 1024, 1024]
    volume_batch_inshape, label_batch_inshape = volume_batch_inshape.cuda(), label_batch_inshape.cuda()
    return volume_batch_inshape, label_batch_inshape


def Spread_bs_tensor(image, label, input_size):
    spread_image = torch.zeros((label.shape[0], input_size, input_size))     # (B, 1024, 1024)
    spread_label = torch.zeros((label.shape[0], input_size, input_size)) + 255
    for i in range(label.shape[0]):
        single_image, single_label = image[i].squeeze(), label[i].squeeze()
        single_image, single_label = torch.transpose(single_image, 0, 2), torch.transpose(single_label, 0, 2)  # (64, 120, 120)
        n_row, n_col = input_size // single_label.shape[1], input_size // single_label.shape[2]
        pw = round((input_size / n_row - single_label.shape[1])/2)  # (128-120)/2=4
        ph = round((input_size / n_col - single_label.shape[2])/2)
        ps = max((n_row * n_col - single_image.shape[0]) // 2, 0)

        # single_image = np.pad(single_image, [(ps, ps), (pw, pw), (ph, ph)], mode='constant', constant_values=0)  # (64, 128, 128), (100, 102, 102)
        single_image = F.pad(single_image, (ph, ph, pw, pw, ps, ps), mode='constant', value=0)  # (64, 128, 128), F.pad和np.pad的顺序是反的
        single_label = F.pad(single_label, (ph, ph, pw, pw, ps, ps), mode='constant', value=255)  # (64, 128, 128)
        s_l = single_label.shape[1]  # small square side length, 128/256/512
        s_w = single_label.shape[2]  # side width, 128/256/512
        pww = (input_size - single_image.shape[1] * n_row) // 2
        phh = (input_size - single_image.shape[2] * n_col) // 2
        if single_image.shape[0] < n_row * n_col:
            single_image = F.pad(single_image, (0, 0, 0, 0, 0, n_row * n_col - single_image.shape[0]), mode='constant', value=0)
            single_label = F.pad(single_label, (0, 0, 0, 0, 0, n_row * n_col - single_label.shape[0]), mode='constant', value=255)
        pww_r, phh_r = pww, phh
        if pww < 0:
            if i == 0:
                spread_image = F.pad(spread_image, (abs(phh), abs(phh), abs(pww), abs(pww), 0, 0), mode='constant', value=0)
                spread_label = F.pad(spread_label, (abs(phh), abs(phh), abs(pww), abs(pww), 0, 0), mode='constant', value=255)
            pww, phh = 0, 0

        for row in range(n_row):
            for col in range(n_col):
                spread_image[i, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w] = single_image[row * n_col + col]
                spread_label[i, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w] = single_label[row * n_col + col]
    if pww_r < 0:
        spread_image = spread_image[:, abs(pww_r):pww_r, abs(phh_r):phh_r]
        spread_label = spread_label[:, abs(pww_r):pww_r, abs(phh_r):phh_r]
    image, label = spread_image, spread_label  # (B, 1024, 1024)
    return image, label, n_row, n_col, pw, ph, ps, pww_r, phh_r, s_l, s_w


def sam_test_tensor(model_sam, image, label, input_size, batch_size, labeled_bs, num_classes, multimask_output, nolora=False):
    image_inshape, label_inshape, n_row, n_col, pw, ph, ps, pww, phh, s_l, s_w = Spread_bs_tensor(image, label, input_size)  # (B, 1024, 1024)

    image_inshape = image_inshape.unsqueeze(1)  # [B, 1, 1024, 1024]
    image_inshape = torch.cat((image_inshape, image_inshape, image_inshape), 1)  # [B, 3, 1024, 1024]
    volume_batch_inshape, label_batch_inshape = image_inshape.cuda(), label_inshape.long().cuda()

    pred_single = torch.zeros((batch_size - labeled_bs, image.shape[-1] + ps * 2, image.shape[-2] + pw * 2, image.shape[-3] + ph * 2)).cuda()  # [B, 80, 112, 112]  (1, 112, 114, 86)
    output_soft_single = torch.zeros((batch_size - labeled_bs, num_classes + 1, image.shape[-1] + ps * 2, image.shape[-2] + pw * 2, image.shape[-3] + ph * 2)).cuda()  # [B, C, 64, 128, 128]

    with torch.no_grad():
        if nolora:
            image_embedding = model_sam.image_encoder(volume_batch_inshape)
            sparse_embeddings, dense_embeddings = model_sam.prompt_encoder(points=None, boxes=None, masks=None)
            low_res_masks, _ = model_sam.mask_decoder(  # [bs=1, 1->C, 256, 256]
                image_embeddings=image_embedding,  # [1, 256, 64, 64]
                image_pe=model_sam.prompt_encoder.get_dense_pe(),  # [1, 256, 64, 64]
                sparse_prompt_embeddings=sparse_embeddings,  # [1, 0, 256]
                dense_prompt_embeddings=dense_embeddings,  # [1, 256, 64, 64]
                multimask_output=multimask_output,
            )
            output_masks = model_sam.postprocess_masks(low_res_masks, (input_size, input_size),(input_size, input_size)).cuda()  # [1, C, 1024, 1024]
        else:
            outputs = model_sam(volume_batch_inshape, multimask_output, input_size)
            output_masks = outputs['masks']
        output_soft_inshape = F.softmax(output_masks, dim=1)  # [B, 2, 1024, 1024]
        prediction_inshape = torch.argmax(output_soft_inshape, dim=1)  # (B, 1024, 1024)

    if pww < 0 or phh < 0:
        pww_r, phh_r = pww, phh
        output_soft_inshape = F.pad(output_soft_inshape, (abs(phh), abs(phh), abs(pww), abs(pww), 0, 0, 0, 0), mode='constant', value=255)
        prediction_inshape = F.pad(prediction_inshape, (abs(phh), abs(phh), abs(pww), abs(pww), 0, 0), mode='constant', value=255)
        pww, phh = 0, 0

    for row in range(n_row):
        for col in range(n_col):
            if row * n_col + col < output_soft_single.shape[2]:
                output_soft_single[:, :, row * n_col + col] = output_soft_inshape[:, :, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w]
                pred_single[:, row * n_col + col] = prediction_inshape[:, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w]
    output_soft_single = output_soft_single[:, :, ps:output_soft_single.shape[-3] - ps, pw:output_soft_single.shape[-2] - pw, ph:output_soft_single.shape[-1] - ph]  # (B, C, 64, 128, 128)
    output_soft_single = output_soft_single.permute(0, 1, 4, 3, 2)  # (B, C, 128, 128, 64)
    pred_single = pred_single[:, ps:pred_single.shape[-3] - ps, pw:pred_single.shape[-2] - pw, ph:pred_single.shape[-1] - ph]  # (B, 64, 128, 128)
    pred_single = pred_single.permute(0, 3, 2, 1)  # (B, 128, 128, 64)
    prediction_sam = pred_single

    return output_soft_single, prediction_sam


def sam_test_ours(model_sam, image, label, input_size, batch_size, labeled_bs, num_classes, multimask_output, nolora=False):
    image_inshape, label_inshape, n_row, n_col, pw, ph, ps, pww, phh, s_l, s_w = Spread_bs(image, label, input_size)  # (B, 1024, 1024)
    volume_batch_inshape, label_batch_inshape = ToTensor_sam_bs(image_inshape, label_inshape)

    pred_single = np.zeros((batch_size - labeled_bs, image.shape[-1] + ps * 2, image.shape[-2] + pw * 2, image.shape[-3] + ph * 2))  # [B, 80, 112, 112]  (1, 112, 114, 86)
    output_soft_single = np.zeros((batch_size - labeled_bs, num_classes + 1, image.shape[-1] + ps * 2, image.shape[-2] + pw * 2, image.shape[-3] + ph * 2))  # [B, C, 64, 128, 128]
    # output_soft_single = np.zeros((batch_size - labeled_bs, num_classes + 1, image.shape[-1] + ps * 2, patch_size + pw * 2, patch_size + ph * 2))  # [B, C, 64, 128, 128]

    with torch.no_grad():
        if nolora:
            image_embedding = model_sam.image_encoder(volume_batch_inshape)
            sparse_embeddings, dense_embeddings = model_sam.prompt_encoder(points=None, boxes=None, masks=None)
            low_res_masks, _ = model_sam.mask_decoder(  # [bs=1, 1->C, 256, 256]
                image_embeddings=image_embedding,  # [1, 256, 64, 64]
                image_pe=model_sam.prompt_encoder.get_dense_pe(),  # [1, 256, 64, 64]
                sparse_prompt_embeddings=sparse_embeddings,  # [1, 0, 256]
                dense_prompt_embeddings=dense_embeddings,  # [1, 256, 64, 64]
                multimask_output=multimask_output,
            )
            output_masks = model_sam.postprocess_masks(low_res_masks, (input_size, input_size),(input_size, input_size)).cuda()  # [1, C, 1024, 1024]
        else:
            outputs = model_sam(volume_batch_inshape, multimask_output, input_size)
            output_masks = outputs['masks']
        output_soft = F.softmax(output_masks, dim=1)  # [B, 2, 1024, 1024]
        output_soft_inshape = output_soft.cpu().detach().numpy()  # (B, C, 1024, 1024)
        prediction_inshape = torch.argmax(output_soft, dim=1)  # (B, 1024, 1024)
        prediction_inshape = prediction_inshape.cpu().detach().numpy()  # (B, 1024, 1024)

    if pww < 0 or phh < 0:
        pww_r, phh_r = pww, phh
        output_soft_inshape = np.pad(output_soft_inshape, [(0, 0), (0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=255)
        prediction_inshape = np.pad(prediction_inshape, [(0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=255)
        pww, phh = 0, 0

    for row in range(n_row):
        for col in range(n_col):
            if row * n_col + col < output_soft_single.shape[2]:
                output_soft_single[:, :, row * n_col + col] = output_soft_inshape[:, :, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w]
                pred_single[:, row * n_col + col] = prediction_inshape[:, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_w:phh + (col + 1) * s_w]
    output_soft_single = output_soft_single[:, :, ps:output_soft_single.shape[-3] - ps, pw:output_soft_single.shape[-2] - pw, ph:output_soft_single.shape[-1] - ph]  # (B, C, 64, 128, 128)
    output_soft_single = np.swapaxes(output_soft_single, -3, -1)  # (B, C, 128, 128, 64)
    pred_single = pred_single[:, ps:pred_single.shape[-3] - ps, pw:pred_single.shape[-2] - pw, ph:pred_single.shape[-1] - ph]  # (B, 64, 128, 128)
    pred_single = np.swapaxes(pred_single, -3, -1)  # (B, 128, 128, 64)
    prediction_sam = pred_single

    return output_soft_single, prediction_sam


def sam_test(model_sam, image, label, modelname, patch_size, input_size, num_classes, multimask_output):
    output_soft_vol = np.zeros((image.shape[0], num_classes+1, image.shape[1], image.shape[2], image.shape[3]))
    prediction_vol = np.zeros((image.shape[0], image.shape[1], image.shape[2], image.shape[3]))
    # print(image.shape)
    for i in range(image.shape[-1]):
        slc = image[..., i]
        lbl = label[..., i]
        volume_batch = slc.reshape(slc.shape[0], 1, slc.shape[1], slc.shape[2])
        volume_batch = torch.from_numpy(volume_batch)
        volume_batch = torch.cat((volume_batch, volume_batch, volume_batch), 1)  # [4, 3, 112, 112]
        volume_batch = volume_batch.cuda()

        transform = ResizeLongestSide(input_size)  # 1024
        input_batch = transform.apply_image_torch(volume_batch)  # [4, 3, 1024, 1024]

        with torch.no_grad():
            if modelname == "MedSAM" or modelname == "MedSAM_v2":
                image_embedding = model_sam.image_encoder(input_batch)
                sparse_embeddings, dense_embeddings = model_sam.prompt_encoder(points=None, boxes=None, masks=None)
                low_res_masks, _ = model_sam.mask_decoder(  # [bs=1, 1->C, 256, 256]
                    image_embeddings=image_embedding,  # [1, 256, 64, 64]
                    image_pe=model_sam.prompt_encoder.get_dense_pe(),  # [1, 256, 64, 64]
                    sparse_prompt_embeddings=sparse_embeddings,  # [1, 0, 256]
                    dense_prompt_embeddings=dense_embeddings,  # [1, 256, 64, 64]
                    multimask_output=multimask_output,
                )
                output_masks = model_sam.postprocess_masks(low_res_masks, (patch_size[0], patch_size[1]), (patch_size[0], patch_size[1])).cuda()  # [1, C, 112, 112]
                output_soft_slc = F.softmax(output_masks, dim=1)  # [B, 2, 112, 112]
                prediction_slc = torch.argmax(output_soft_slc, dim=1)  # (B, 1024, 1024)
            elif modelname == "SAM_lora":
                outputs = model_sam(input_batch, multimask_output, input_size)
                output_masks, low_res_masks = outputs['masks'], outputs['low_res_logits']
                output_masks = F.interpolate(low_res_masks, (patch_size[0], patch_size[1]), mode="bilinear", align_corners=False)  # [1, 2, 112, 112]
                output_soft_slc = F.softmax(output_masks, dim=1)  # [B, 2, 112, 112]
                # output_soft_vol[..., i] = output_soft_slc.cpu().detach().numpy()  # (B, C, 1024, 1024)
                prediction_slc = torch.argmax(output_soft_slc, dim=1)  # (B, 1024, 1024)
                # prediction_vol[..., i] = prediction_slc.cpu().detach().numpy()  # (B, 1024, 1024)
            output_soft_vol[..., i] = output_soft_slc.cpu().detach().numpy()  # (B, C, 1024, 1024)
            prediction_vol[..., i] = prediction_slc.cpu().detach().numpy()  # (B, 1024, 1024)

    return output_soft_vol, prediction_vol


def sam_test_scale(model_sam, image, label, input_size, scale, num_classes, multimask_output):
    # image 112*112*80
    image_inshape, label_inshape, n_row, n_col, pw, ph, ps, pww, phh, s_l, s_w = Spread_bs(image, label, input_size)  # (B, 1024, 1024)

    single_image = image.squeeze()
    single_image = np.swapaxes(single_image, 0, 2)   # (64, 120, 120)

    n_row, n_col = scale, scale  # 8, 8
    n_slc = n_row * n_col  # 64
    n_large = single_image.shape[0] // n_slc
    r_large = single_image.shape[0] % n_slc
    if r_large == 0:
        spread_image = np.zeros((n_large, input_size, input_size))  # (n_large, 1024, 1024)
    else:
        spread_image = np.zeros((n_large+1, input_size, input_size))  # (n_large+1, 1024, 1024)

    pw, ph, ps = 4, 4, 0
    image_shape = (1024 // scale - pw * 2, 1024 // scale - pw * 2, single_image.shape[0])

    image_itk = resampling(sitk.GetImageFromArray(single_image), image_shape, lbl=False)
    single_image = sitk.GetArrayFromImage(image_itk)

    single_image = np.pad(single_image, [(ps, ps), (pw, pw), (ph, ph)], mode='constant', constant_values=0)  # (64, 128, 128)
    s_l = single_image.shape[1]  # side length, 128/256/512
    s_w = single_image.shape[2]  # side width, 128/256/512

    for i in range(n_large):
        for row in range(n_row):
            for col in range(n_col):
                spread_image[i, row * s_l:(row + 1) * s_l, col * s_w:(col + 1) * s_w] = single_image[i * n_slc + row * n_col + col]
    if r_large != 0:
        for row in range(n_row):
            for col in range(n_col):
                spread_image[n_large, row * s_l:(row + 1) * s_l, col * s_w:(col + 1) * s_w] = single_image[-1]     # 128*128
                # spread_image[n_large, row * s_l:(row + 1) * s_l, col * s_w:(col + 1) * s_w] = single_image[-1 * n_row * n_col:]
    image_inshape = spread_image  # (B, 1024, 1024)

    volume_batch_inshape, _ = ToTensor_sam_bs(image_inshape, image_inshape)


    output_soft_single = np.zeros((1, num_classes + 1, image_shape[-1] + ps * 2, image_shape[-2] + pw * 2, image_shape[-3] + ph * 2))  # [B, C, 80, 256, 256]
    # output_soft_single = np.zeros((1, num_classes + 1, 1, image_shape[-2] + pw * 2, image_shape[-3] + ph * 2))  # [B, C, 1, 256, 256]
    output_single = np.zeros((1, num_classes + 1, image.shape[-1], image.shape[-2], image.shape[-3]))
    for i in range(volume_batch_inshape.shape[0]):
        with torch.no_grad():
            outputs = model_sam(volume_batch_inshape[i].unsqueeze(0), multimask_output, input_size)
            output_masks = outputs['masks']
            output_soft_inshape = output_masks.cpu().detach().numpy()  # (B, C, 1024, 1024)
        if i < n_large:
            for row in range(n_row):
                for col in range(n_col):
                    # print(i, output_soft_single.shape)
                    output_soft_single[:, :, i * n_slc + row * n_col + col] = output_soft_inshape[:, :, row * s_l:(row + 1) * s_l, col * s_w:(col + 1) * s_w]
        # else:
        #     for row in range(n_row):
        #         for col in range(n_col):
        #             output_soft_single[:, :, -1] = output_soft_inshape[:, :, row * s_l:(row + 1) * s_l, col * s_w:(col + 1) * s_w]

        output_soft_single_p = output_soft_single[:, :, ps:output_soft_single.shape[-3] - ps, pw:output_soft_single.shape[-2] - pw, ph:output_soft_single.shape[-1] - ph]  # (1, 2, 248, 248, 80)
        for c in range(num_classes):
            output_itk = resampling(sitk.GetImageFromArray(output_soft_single_p[0, c]), image.squeeze().shape, lbl=False)
            output_single[0, c] = sitk.GetArrayFromImage(output_itk)


    # output_soft_single = np.zeros((1, num_classes + 1, image_shape[-1] + ps * 2, image_shape[-2] + pw * 2, image_shape[-3] + ph * 2))  # [B, C, 80, 256, 256]
    # with torch.no_grad():
    #     outputs = model_sam(volume_batch_inshape, multimask_output, input_size)
    #     output_masks = outputs['masks']
    #     output_soft_inshape = output_masks.cpu().detach().numpy()  # (B, C, 1024, 1024)
    #     # output_soft = F.softmax(output_masks, dim=1)  # [B, 2, 1024, 1024]
    #
    # for i in range(n_large):
    #     for row in range(n_row):
    #         for col in range(n_col):
    #             output_soft_single[:, :, i * n_slc + row * n_col + col] = output_soft_inshape[i, :, row * s_l:(row + 1) * s_l, col * s_w:(col + 1) * s_w]
    # if r_large != 0:
    #     for row in range(n_row):
    #         for col in range(n_col):
    #             output_soft_single[:, :, -1] = output_soft_inshape[n_large, :, row * s_l:(row + 1) * s_l, col * s_w:(col + 1) * s_w]
    #
    # output_soft_single = output_soft_single[:, :, ps:output_soft_single.shape[-3] - ps, pw:output_soft_single.shape[-2] - pw, ph:output_soft_single.shape[-1] - ph]  # (1, 2, 248, 248, 80)
    # output_single = np.zeros((1, num_classes + 1, image.shape[-1], image.shape[-2], image.shape[-3]))
    # for c in range(num_classes):
    #     output_itk = resampling(sitk.GetImageFromArray(output_soft_single[0, c]), image.squeeze().shape, lbl=False)
    #     output_single[0, c] = sitk.GetArrayFromImage(output_itk)
    output_soft_single = np.swapaxes(output_single, -3, -1)  # (B, C, 128, 128, 64)
    return output_soft_single


def Recover_bs(output_soft, patch_size, bs, num_classes, n_row, n_col, pw, ph, ps, pww, phh, s_l):
    output_soft_inshape = output_soft.cpu().detach().numpy()  # (B, C, 1024, 1024)
    outputs_soft_sam = np.zeros((bs, num_classes, patch_size[2] + ps * 2, patch_size[0] + pw * 2, patch_size[1] + ph * 2))  # [B, C, 64, 128, 128]

    if pww < 0 or phh < 0:
        pww_r, phh_r = pww, phh
        output_soft_inshape = np.pad(output_soft_inshape, [(0, 0), (0, 0), (abs(pww), abs(pww)), (abs(phh), abs(phh))], mode='constant', constant_values=255)
        pww, phh = 0, 0
    for row in range(n_row):
        for col in range(n_col):
            if row * n_col + col < outputs_soft_sam.shape[2]:
                outputs_soft_sam[:, :, row * n_col + col] = output_soft_inshape[:, :, pww + row * s_l:pww + (row + 1) * s_l, phh + col * s_l:phh + (col + 1) * s_l]
    outputs_soft_sam = outputs_soft_sam[:, :, ps:outputs_soft_sam.shape[-3] - ps, pw:outputs_soft_sam.shape[-2] - pw, ph:outputs_soft_sam.shape[-1] - ph]  # (B, C, 64, 128, 128)
    outputs_soft_sam = np.swapaxes(outputs_soft_sam, -3, -1)  # (B, C, 128, 128, 64)
    outputs_soft_sam = torch.from_numpy(outputs_soft_sam).cuda()

    return outputs_soft_sam


def seg_test(model_seg, volume_batch, label_batch):
    volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
    with torch.no_grad():
        outputs, outputs_aux1, outputs_aux2, outputs_aux3 = model_seg(volume_batch)
        outputs_soft = F.softmax(outputs, dim=1)
        prediction_seg = torch.argmax(outputs_soft, dim=1)
        prediction_seg = prediction_seg.cpu().detach().numpy()  # (1, 128, 128, 64)

    return outputs_soft, prediction_seg


def RandomRotFlip_2d(image, k, axis):     # (1024, 1024)
    image = np.rot90(image, k)
    image = np.flip(image, axis=axis).copy()
    return image


def RandomFlipRot_2d(image, k, axis):     # (1024, 1024)
    image = np.flip(image, axis=axis).copy()
    # label = np.flip(label, axis=axis).copy()
    image = np.rot90(image, k)
    # label = np.rot90(label, k)
    return image


def RandomCropResample(image, label, output_size):
    raw_size = image.shape
    # pad the sample if necessary
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1] or label.shape[2] <= output_size[2]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        pd = max((output_size[2] - label.shape[2]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

    (w, h, d) = image.shape
    w1 = np.random.randint(0, w - output_size[0])
    h1 = np.random.randint(0, h - output_size[1])
    d1 = np.random.randint(0, d - output_size[2])

    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]
    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1], d1:d1 + output_size[2]]

    image_itk = resampling(sitk.GetImageFromArray(image), (raw_size[2], raw_size[1], raw_size[0]), lbl=False)
    label_itk = resampling(sitk.GetImageFromArray(label), (raw_size[2], raw_size[1], raw_size[0]), lbl=True)
    image = sitk.GetArrayFromImage(image_itk)
    label = sitk.GetArrayFromImage(label_itk)
    return image, label


def RandomCropResample_2d(image, label, output_size):
    raw_size = image.shape
    # pad the sample if necessary
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

    (w, h) = image.shape
    w1 = np.random.randint(0, w - output_size[0])
    h1 = np.random.randint(0, h - output_size[1])

    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1]]

    image_itk = resampling(sitk.GetImageFromArray(image), raw_size, lbl=False)
    label_itk = resampling(sitk.GetImageFromArray(label), raw_size, lbl=True)
    image = sitk.GetArrayFromImage(image_itk)
    label = sitk.GetArrayFromImage(label_itk)
    return image, label


def postprocess(prediction, post_morph=True):
    if post_morph:
        for i in range(prediction.shape[0]):
            pred_slc = prediction[i]
            pred_slc = pred_slc.astype(np.uint8)
            kernel = np.ones((2, 2), np.uint8)
            pred_slc = cv2.dilate(pred_slc, kernel)
            prediction[i] = pred_slc
    return prediction


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))
    return normalized_sdf

