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
    parser.add_argument("--angres", type=int, default=14, help="input angle number")

    parser.add_argument("--out_channels", type=int, default=8, help="output spectrum channels")
    parser.add_argument('--testset_dir', type=str, default='dataset/testset/simulation_8channels/')

    parser.add_argument("--patchsize", type=int, default=144, help="inference patches size")
    parser.add_argument("--stride", type=int, default=72, help="stride size two inference patches")

    parser.add_argument('--model_path', type=str, default='pretrain_model/simulation_8channels/epoch_125.pth.tar')
    parser.add_argument('--save_path', type=str, default='output/simulation_8channels/')

    return parser.parse_args()


def test(cfg, test_Names, test_loaders):
    # Initialize the network with input and output channel parameters
    net = s2lfm_Net(cfg.angres, cfg.out_channels)
    net.to(cfg.device)
    cudnn.benchmark = True  # Enable cudnn auto-tuner for better performance

    # Load the pre-trained model weights if available
    if os.path.isfile(cfg.model_path):
        model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
        net.load_state_dict(model['state_dict'])
    else:
        print("=> no model found at '{}'".format(cfg.model_path))

    # Prepare indices for angular views
    ind_all = np.arange(cfg.angres*cfg.angres).reshape(cfg.angres, cfg.angres)        
    delt = (cfg.angres-1) // (cfg.angres-1)
    ind_source = ind_all[0:cfg.angres:delt, 0:cfg.angres:delt]
    ind_source = torch.from_numpy(ind_source.reshape(-1))

    # inference here
    with torch.no_grad():  
        psnr_testset = []
        ssim_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            psnr_epoch_test, ssim_epoch_test = inference(test_loader, test_name, net, ind_source)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)

    psnr_ = float(np.array(psnr_testset).mean())
    ssim_ = float(np.array(ssim_testset).mean())
    return psnr_, ssim_

def inference(test_loader, test_name, net, ind_source):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        # Prepare input data
        data = data.squeeze().to(cfg.device)  # Shape: numU, numV, h*angres, w*angres
        c, uh, vw = data.shape
        h0, w0 = uh // cfg.angres, vw // cfg.angres

        # Divide the light field into small patches for inference if out of memory
        subLFin = LFdivide(data, cfg.angres, cfg.patchsize, cfg.stride)
        numU, numV, c, H, W = subLFin.shape

        
        minibatch = 1
        num_inference = numU*numV//minibatch
        tmp_in = subLFin.contiguous().view(numU*numV, c, subLFin.shape[3], subLFin.shape[4])

        # Run inference on each patch
        s = time.time()
        with torch.no_grad():
            out_lf = []
            for idx_inference in range(num_inference):
                tmp = tmp_in[idx_inference*minibatch:(idx_inference+1)*minibatch,:,:,:]
                out_lf.append(net(tmp.to(cfg.device)))
            if (numU*numV)%minibatch:
                tmp = tmp_in[(idx_inference+1)*minibatch:,:,:,:]
                out_lf.append(net(tmp.to(cfg.device)))
        infer_time = time.time()-s
        print(infer_time)

        # Merge the output patches back into a full light field
        out_lf = torch.cat(out_lf, 0)
        subLFout = out_lf.view(numU, numV, cfg.out_channels, cfg.angres * cfg.patchsize, cfg.angres * cfg.patchsize)
        outLF = LFintegrate(subLFout, cfg.angres, cfg.patchsize, cfg.stride, h0, w0)

        # Prepare ground truth and compute metrics
        gt = LFsplit(label, cfg.angres).squeeze().view(cfg.angres,cfg.angres,cfg.out_channels,h0,w0)
        psnr, ssim = cal_metrics(gt, outLF, cfg.angres, ind_source)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

        # Save output as .mat file
        isExists = os.path.exists(cfg.save_path + test_name)
        if not (isExists ):
            os.makedirs(cfg.save_path + test_name)
        outLF = outLF.view(-1,cfg.out_channels,h0, w0)
        sio.savemat(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3] + '.mat',
                        {'LF': outLF.numpy()})

        # Save output and ground truth as TIFF images for each channel
        outLF = outLF.cpu()
        isExists = os.path.exists(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3])
        if not (isExists ):
            os.makedirs(cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3])
        gt = gt.view(cfg.angres*cfg.angres,-1,h0,w0)
        for cdx in range(cfg.out_channels):
            pth = cfg.save_path + test_name + '/' + test_loader.dataset.file_list[idx_iter][0:-3]
            imwrite(pth + '/channel_' + str(cdx+1) + '.tif', np.single(outLF[:,cdx,:,:] * 1e+4), imagej=False, metadata={'axes': 'ZYX'}, compression ='zlib')
            imwrite(pth +      '/GT_' + str(cdx+1) + '.tif', np.single(   gt[:,cdx,:,:] * 1e+4), imagej=False, metadata={'axes': 'ZYX'}, compression ='zlib')


    # Print metrics for all test samples
    print(np.array(psnr_iter_test))
    print(np.array(ssim_iter_test))

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test

def main(cfg):
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    psnr_, ssim_ = test(cfg, test_Names, test_Loaders)
    print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_Names, psnr_, ssim_))

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
