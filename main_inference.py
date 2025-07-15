import time
import argparse
import torch.backends.cudnn as cudnn
from utils import *
from s2LFM_Net import s2lfm_Net
import scipy.io as sio
import time
from tifffile import imwrite


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angin", type=int, default=12, help="input angle number")

    parser.add_argument("--out_channels", type=int, default=8, help="output spectrum channels")
    parser.add_argument('--testset_dir', type=str, default='Dataset/testset/simulation_8channels/')

    parser.add_argument("--patchsize", type=int, default=180, help="inference patches size")
    parser.add_argument("--stride", type=int, default=120, help="stride size two inference patches")

    parser.add_argument('--model_path', type=str, default='pretrain_model/simulation_8channels/epoch_150.pth.tar')
    parser.add_argument('--save_path', type=str, default='output/simulation_8channels/')

    return parser.parse_args()


def test(cfg, test_Names, test_loaders):

    net = s2lfm_Net(cfg.angin, cfg.out_channels)
    net.to(cfg.device)
    cudnn.benchmark = True

    if os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
        net.load_state_dict(model['state_dict'])
    else:
        print("=> no model found at '{}'".format(cfg.model_path))
    ind_all = np.arange(cfg.angin*cfg.angin).reshape(cfg.angin, cfg.angin)        
    delt = (cfg.angin-1) // (cfg.angin-1)
    ind_source = ind_all[0:cfg.angin:delt, 0:cfg.angin:delt]
    ind_source = torch.from_numpy(ind_source.reshape(-1))

    with torch.no_grad():

        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            inference(test_loader, test_name, net, ind_source)

            pass
        pass


def inference(test_loader, test_name, net, ind_source):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angin, w*angin
    
        c, uh, vw = data.shape
        h0, w0 = uh // cfg.angin, vw // cfg.angin
        subLFin = LFdivide(data, cfg.angin, cfg.patchsize, cfg.stride)  # numU, numV, h*angin, w*angin
        numU, numV, c, H, W = subLFin.shape
        s = time.time()
        minibatch = 1
        num_inference = numU*numV//minibatch
        tmp_in = subLFin.contiguous().view(numU*numV, c, subLFin.shape[3], subLFin.shape[4])
        
        with torch.no_grad():
            out_lf = []
            for idx_inference in range(num_inference):
                tmp = tmp_in[idx_inference*minibatch:(idx_inference+1)*minibatch,:,:,:]
                out_lf.append(net(tmp.to(cfg.device)))#
            if (numU*numV)%minibatch:
                tmp = tmp_in[(idx_inference+1)*minibatch:,:,:,:]
                out_lf.append(net(tmp.to(cfg.device)))#
        infer_time = time.time()-s
        print(infer_time)
        out_lf = torch.cat(out_lf, 0)
        subLFout = out_lf.view(numU, numV, cfg.out_channels, cfg.angin * cfg.patchsize, cfg.angin * cfg.patchsize)

        outLF = LFintegrate(subLFout, cfg.angin, cfg.patchsize, cfg.stride, h0, w0)

        gt = LFsplit(label, cfg.angin).squeeze().view(cfg.angin,cfg.angin,cfg.out_channels,h0,w0)
        psnr, ssim = cal_metrics(gt, outLF, cfg.angin, ind_source)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        isExists = os.path.exists(cfg.save_path + test_name)
        if not (isExists ):
            os.makedirs(cfg.save_path + test_name)

        outLF = outLF.view(-1,cfg.out_channels,h0, w0)
        sio.savemat(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                        {'LF': outLF.numpy()})
        
        outLF = outLF.cpu()
        isExists = os.path.exists(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3])
        if not (isExists ):
            os.makedirs(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3])

        gt = gt.view(cfg.angout*cfg.angout,-1,h0,w0)
        for cdx in range(cfg.out_channels):
            pth = cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3]
            imwrite(pth + '/channel_' + str(cdx+1) + '.tif', np.single(outLF[:,cdx,:,:] * 1e+4), imagej=False, metadata={'axes': 'ZYX'}, compression ='zlib')
            imwrite(pth +      '/GT_' + str(cdx+1) + '.tif', np.single(   gt[:,cdx,:,:] * 1e+4), imagej=False, metadata={'axes': 'ZYX'}, compression ='zlib')
        pass

    print(np.array(psnr_iter_test))
    print(np.array(ssim_iter_test))


def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    _, psnr_epoch_test, ssim_epoch_test = test(cfg, test_Names, test_Loaders)
    print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_Names, psnr_epoch_test, ssim_epoch_test))

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
