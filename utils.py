import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from skimage import metrics

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/%06d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            if label.shape[0] > 10:
                label = label.reshape([1,label.shape[0],-1])
            data, label = np.transpose(data, (0, 2, 1)), np.transpose(label, (0, 2, 1))
            data, label = augmentation(data, label)
            data, label = np.transpose(data, (1, 2, 0)), np.transpose(label, (1, 2, 0))
            data = ToTensor()(data.copy())
            label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    dataset_dir = args.testset_dir
    data_list = os.listdir(dataset_dir)
    test_Loaders = []
    length_of_tests = 0

    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name)
        length_of_tests += len(test_Dataset)
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL'):
        super(TestSetDataLoader, self).__init__()
        self.angres = args.angres
        self.dataset_dir = args.testset_dir + data_name
        self.file_list = []
        tmp_list = os.listdir(self.dataset_dir)
        for index, _ in enumerate(tmp_list):
            tmp_list[index] = tmp_list[index]
        self.file_list.extend(tmp_list)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = self.dataset_dir + '/' + self.file_list[index]
        with h5py.File(file_name, 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            if label.shape[0] > 10:
                label = label.reshape([1,label.shape[0],-1])
            data, label = np.transpose(data, (2, 1, 0)), np.transpose(label, (2, 1, 0))
            data, label = ToTensor()(data.copy()), ToTensor()(label.copy())

        return data, label

    def __len__(self):
        return self.item_num


def augmentation(data, label):
    if random.random() < 0.5:
        data = data[:,:, ::-1]
        label = label[:,:, ::-1]
    if random.random() < 0.5:
        data = data[:,::-1, :]
        label = label[:,::-1, :]
    if random.random() < 0.5:
        data = data.transpose(0, 2, 1)
        label = label.transpose(0, 2, 1)
    return data, label


def LFdivide(data, angRes, patch_size, stride):
    c, uh, vw = data.shape
    h0 = uh // angRes
    w0 = vw // angRes
    bdr = (patch_size - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1
    hE = stride * (numU-1) + patch_size
    wE = stride * (numV-1) + patch_size

    dataE = torch.zeros(c, hE*angRes, wE*angRes)
    for u in range(angRes):
        for v in range(angRes):
            Im = data[:,u*h0:(u+1)*h0, v*w0:(v+1)*w0]
            dataE[:, u*hE : u*hE+h, v*wE : v*wE+w] = ImageExtend(Im, bdr)
    subLF = torch.zeros(numU, numV, c, patch_size*angRes, patch_size*angRes)
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u*hE + kh*stride
                    vv = v*wE + kw*stride
                    subLF[kh, kw, :, u*patch_size:(u+1)*patch_size, v*patch_size:(v+1)*patch_size] = dataE[:, uu:uu+patch_size, vv:vv+patch_size]
    return subLF


def ImageExtend(Im, bdr):
    c, h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out

def LFintegrate(subLF, angRes, pz, stride, h0, w0):
    numU, numV, c, pH, pW = subLF.shape
    ph, pw = pH //angRes, pW //angRes
    bdr = (pz - stride) //2
    temp = torch.zeros(c, stride*numU, stride*numV)
    outLF = torch.zeros(angRes, angRes, c, h0, w0)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    temp[:, ku*stride:(ku+1)*stride, kv*stride:(kv+1)*stride] = subLF[ku, kv, :, u*ph+bdr:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride]

            outLF[u, v, :, :, :] = temp[:, 0:h0, 0:w0]

    return outLF


def cal_psnr(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)

def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

def cal_metrics(img1, img2, angRes, indicate):
    if len(img1.size())==3:
        [c, H, W] = img1.size()
        img1 = img1.view(c, angRes, H // angRes, angRes, W // angRes).permute(1,3,0,2,4)
    if len(img2.size())==3:
        [c, H, W] = img2.size()
        img2 = img2.view(c, angRes, H // angRes, angRes, W // angRes).permute(1,3,0,2,4)

    [U, V, c, h, w] = img2.size()
    PSNR = np.zeros(shape=(U, V, c), dtype='float32')
    SSIM = np.zeros(shape=(U, V, c), dtype='float32')

    bd = 2

    for u in range(U):
        for v in range(V):
            k = u*U + v
            for ch in range(c):
                if k not in indicate:
                    continue
                else:                
                    PSNR[u, v, ch] = cal_psnr(img1[u, v, ch, bd:-bd, bd:-bd], img2[u, v, ch, bd:-bd, bd:-bd])
                    SSIM[u, v, ch] = cal_ssim(img1[u, v, ch, bd:-bd, bd:-bd], img2[u, v, ch, bd:-bd, bd:-bd])
            pass
        pass
    pass

    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean

def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st


