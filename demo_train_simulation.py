import time
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils import *
from s2LFM_Net import s2lfm_Net

def parse_args():
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angres", type=int, default=14, help="input angle number")
    parser.add_argument("--out_channels", type=int, default=8, help="output spectrum channels")
    parser.add_argument("--datasize", type=int, default=72, help="spatial pixel size for each angle")

    parser.add_argument('--model_name', type=str, default='s2LFMNet')
    parser.add_argument('--trainset_dir', type=str, default='dataset/trainset/simulation_8channels/Train/')
    parser.add_argument('--testset_dir', type=str, default='dataset/trainset/simulation_8channels/Validation/')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=125, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=25, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument("--smooth", type=float, default=0.001, help="smooth loss")

    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrainmodel_path', type=str, default='Substitude Your Path Here')
    # if you need to continue training based on your latest trained model, please change the lr above as the same as the last lr when you saved your latest trained model.

    return parser.parse_args()

def train(cfg, train_loader, test_Names, test_loaders):
    """
    Main training loop for the s2LFM-Net model.
    Handles training, validation, checkpointing, and learning rate scheduling.
    """
    net = s2lfm_Net(cfg.angres, cfg.out_channels)
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0
       
    # Prepare angular indices for source views
    ind_all = np.arange(cfg.angres*cfg.angres).reshape(cfg.angres, cfg.angres)        
    delt = (cfg.angres-1) // (cfg.angres-1)
    ind_source = ind_all[0:cfg.angres:delt, 0:cfg.angres:delt]
    ind_source = torch.from_numpy(ind_source.reshape(-1))

    # Load pretrained model if specified
    if cfg.load_pretrain:
        if os.path.isfile(cfg.pretrainmodel_path):
            model = torch.load(cfg.pretrainmodel_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            epoch_state = model["epoch"]
            optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
            optimizer._step_count = epoch_state
            scheduler._step_count = epoch_state
            scheduler.last_epoch = epoch_state - 1
            scheduler.step()
            print("Pretrain model successfully loaded")
        else:
            print("Wrong! No model found in '{}'".format(cfg.load_model))
    else:
        optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)

    loss_epoch = []
    loss_list = []

    for idx_epoch in range(epoch_state, cfg.n_epochs):
        # Training loop for one epoch
        for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            out  = net(data)

            loss = criterion_Loss(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        # Logging and checkpointing
        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f, lr---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(optimizer.state_dict()['param_groups'][0]['lr'])))
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'optimier_state_dict': optimizer.state_dict(),
                'schedular': scheduler},
                save_path='pretrain_model/simulation_8channels/', filename= 'epoch_' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []

        ''' evaluation '''
        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []
            # Evaluate on all test sets
            for index, test_name in enumerate(test_Names):
                test_loader = test_loaders[index]
                psnr_epoch_test, ssim_epoch_test = valid(test_loader, net, ind_source)
                psnr_testset.append(psnr_epoch_test)
                ssim_testset.append(ssim_epoch_test)
                print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (test_name, psnr_epoch_test, ssim_epoch_test))

        scheduler.step()


def valid(test_loader, net, ind_source):
    """
    Evaluate the model on the validation/test set.
    Returns average PSNR and SSIM for the dataset.
    """
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device) 
        # label = label.squeeze()

        # Prepare data for network input
        c, uh, vw = data.shape
        h0, w0 = uh // cfg.angres, vw // cfg.angres
        subLFin = LFdivide(data, cfg.angres, cfg.datasize, cfg.datasize)
        numU, numV, c, H, W = subLFin.shape # numU = numV = 1
        minibatch = 1
        num_inference = numU*numV//minibatch
        tmp_in = subLFin.contiguous().view(numU*numV, c, subLFin.shape[3], subLFin.shape[4])
        
        with torch.no_grad():
            out_lf = []
            # Inference in minibatches
            for idx_inference in range(num_inference):
                tmp = tmp_in[idx_inference*minibatch:(idx_inference+1)*minibatch,:,:,:]
                out_lf.append(net(tmp.to(cfg.device)))#
            if (numU*numV)%minibatch:
                tmp = tmp_in[(idx_inference+1)*minibatch:,:,:,:]
                out_lf.append(net(tmp.to(cfg.device)))#
        out_lf = torch.cat(out_lf, 0)
        subLFout = out_lf.view(numU, numV, cfg.out_channels, cfg.angres * cfg.datasize, cfg.angres * cfg.datasize)

        # Reconstruct the full light field
        outLF = LFintegrate(subLFout, cfg.angres, cfg.datasize, cfg.stride, h0, w0) #[ang, ang, channel, H, W]
        
        gt = LFsplit(label, cfg.angres).squeeze().view(cfg.angres,cfg.angres,cfg.out_channels,h0,w0)
        psnr, ssim = cal_metrics(gt, outLF, cfg.angres, ind_source)

        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test

def save_ckpt(state, save_path='../model', filename='check.pth.tar'):
    """
    Save model checkpoint to disk.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"folder '{save_path}' made")
    else:
        print(f"folder '{save_path}' exists")

    torch.save(state, os.path.join(save_path,filename))

def main(cfg):
    """
    Main entry point: prepares data loaders and starts training.
    """
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=cfg.batch_size, shuffle=False)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    train(cfg, train_loader, test_Names, test_Loaders)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)