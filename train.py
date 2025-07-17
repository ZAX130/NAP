import os
import glob
import sys
import random
import time
import torch
import numpy as np
# import scipy.ndimage
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchsummary import summary
from data import datasets, trans
from torchvision import transforms

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import utils


def train(datadir,
          model_dir,
          load_model,
          gpu,
          initial_epoch,
          epochs,
          steps_per_epoch,
          batch_size,
          atlas=False,
          bidir=False):
    # train_vol_names = glob.glob(os.path.join(datadir, '*.nii.gz'))
    # random.shuffle(train_vol_names)  # shuffle volume list
    # assert len(train_vol_names) > 0, 'Could not find any training data'

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = True

    train_composed = transforms.Compose([  # trans.RandomFlip(0),
        trans.NumpyType((np.float32, np.float32)),
    ])
    val_composed = transforms.Compose([trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16))])
    train_set = datasets.LPBABrainDatasetS2S(glob.glob(datadir + '*.pkl'), transforms=train_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_set = datasets.LPBABrainInferDatasetS2S(glob.glob('/hy-tmp/LPBA_data/Val/' + '*.pkl'), transforms=val_composed)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    # extract shape from sampled input
    inshape = (160, 192, 160)

    os.makedirs(model_dir, exist_ok=True)

    # prepare odel folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    torch.backends.cudnn.deterministic = True
    reg_model = utils.register_model(inshape, 'nearest')
    reg_model.cuda()
    # prepare the model
    model = vxm.NATfrp_modet.RPNet(inshape)
    model.to(device)
    # summary(model)
    if load_model != False:
        print('loading', load_model)
        best_model = torch.load(load_model)
        model.load_state_dict(best_model)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # prepare losses
    Losses = [vxm.losses.NCC().loss, vxm.losses.Grad_2('l2').loss]
    Weights = [1.0, 1.0]

    # training/validate loops
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch, epochs, 1e-4)
        # training
        model.train()
        train_losses = []
        train_total_loss = []
        idx = 0
        for data in train_loader:
            idx += 1
            adjust_learning_rate(optimizer, epoch, epochs, 1e-4)
            model.train()
            # generate inputs (and true outputs) and convert them to tensors
            # inputs, labels = data[0]
            # # inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
            # # labels = [torch.from_numpy(d).to(device).float() for d in labels]
            # inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]  # 其实包括了俩
            # labels = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in labels]  # 一个
            source = data[0].cuda()
            target = data[1].cuda()
            # run inputs through the model to produce a warped image and flow field
            pred = model(source, target)

            # calculate total loss
            loss = 0
            loss_list = []
            for i, Loss in enumerate(Losses):
                curr_loss = Loss(pred[i], target) * Weights[i]
                loss_list.append(curr_loss.item())
                loss += curr_loss
            train_losses.append(loss_list)
            train_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(model_dir + ' Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                                loss.item(),
                                                                                                loss_list[0],
                                                                                                loss_list[1]))
        # print epoch info
        epoch_info = model_dir + ' IIRP Epoch %d/%d' % (epoch + 1, epochs)
        time_info = 'Total %.2f sec' % (time.time() - start_time)
        train_losses = ', '.join(['%.4f' % f for f in np.mean(train_losses, axis=0)])
        train_loss_info = 'Train loss: %.4f  (%s)' % (np.mean(train_total_loss), train_losses)
        print(' - '.join((epoch_info, time_info, train_loss_info)), flush=True)

        # save model checkpoint
        if epoch == 0 or (epoch + 1) % 5 == 0:
            eval_dsc = utils.AverageMeter()
            with torch.no_grad():
                for data in val_loader:
                    model.eval()
                    data = [t.cuda() for t in data]
                    x = data[0]
                    y = data[1]
                    x_seg = data[2]
                    y_seg = data[3]
                    # x_in = torch.cat((x, y), dim=1)
                    # grid_img = mk_grid_img(8, 1, img_size)
                    output = model(x, y)
                    def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                    # def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                    dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                    eval_dsc.update(dsc.item(), x.size(0))
                    print(model_dir + ' ', epoch, ':', eval_dsc.avg)
            torch.save(model.state_dict(), os.path.join(model_dir, '%04d_%.4f.pt' % (epoch + 1, eval_dsc.avg)))


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--datadir', default='/hy-tmp/LPBA_data/Train/', help='base data directory')
    parser.add_argument('--model-dir', default='models_NAP', help='model output directory (default: models)')
    parser.add_argument('--load-model', default=False, help='optional model file to initialize with')
    parser.add_argument('--gpu', default='1', help='GPU ID number(s), comma-separated (default: 0)')
    parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs (default: 1500)')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')

    args = parser.parse_args()
    train(**vars(args))
