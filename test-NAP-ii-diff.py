import glob
import os, utils
import time

from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
# from models_change import Im2grid  #改动
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd
import sys

sys.path.append('..')
from config import configs


def plot_grid(gridx, gridy, **kwargs):
    for i in range(gridx.shape[1]):
        plt.plot(gridx[i, :], gridy[i, :], linewidth=0.8, **kwargs)
    for i in range(gridx.shape[0]):
        plt.plot(gridx[:, i], gridy[:, i], linewidth=0.8, **kwargs)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)


def MAE_torch(x, y):
    return torch.mean(torch.abs(x - y))


GPU_iden =1


def main():
    test_dir = configs['test_dir']
    model_idx = -1
    weights = [1, 1]  # loss weights
    lr = 0.0001
    c = 64
    model_folder = 'NAP-diff/'
    delta = 0.0005
    img_size = configs['img_size']
    model = vxm.NATfrp_modet.IIRPNet2_diff(img_size, nccres=delta)
    model.cuda()
    model_dir = configs['experiments'] + model_folder

    if not os.path.exists('/data/whq/pred/' + configs['dataset_dir_name'] + '/II' + model_folder):
        os.makedirs('/data/whq/pred/' + configs['dataset_dir_name'] + '/II' + model_folder)
    f = open(configs['dataset_dir_name']+' II'+model_folder[:-1]+'.txt', "w")
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])#['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model, strict=False)
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    num_images = len(glob.glob(test_dir + '*.pkl'))
    test_set = datasets.LPBABrainInferDatasetS2S(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    data_jac = []
    row_names = []
    eval_dsc_def = AverageMeter()
    eval_det = AverageMeter()
    eval_time = AverageMeter()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    all_ct = []
    with torch.no_grad():
        stdy_idx = 0
        for data in tqdm(test_loader):
            x_index = stdy_idx // (num_images - 1)
            s = stdy_idx % (num_images - 1)
            y_index = s + 1 if s >= x_index else s
            pair_name = '%02d to %02d' % (x_index, y_index)
            row_names.append(pair_name)
            print(pair_name, end=' ')

            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            # 速度
            starter.record()
            x_def, flow, ct = model(x, y)
            all_ct.append(ct)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)

            if stdy_idx > 40:
                eval_time.update(curr_time, x.size(0))

            if stdy_idx<200:
                print('finish! ref time=', curr_time, end=' ')
            else:
                print('avg time=', eval_time.avg, end=' ')

            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            # 保存
            moved = def_out.long().detach().cpu().numpy()[0, 0, :, :, :]
            np.save(
                '/data/whq/pred/' + configs['dataset_dir_name'] + '/II' + model_folder + str(stdy_idx) + ',' + pair_name,
                arr=moved)

            # 雅可比
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            fold_ratio = np.sum(jac_det <= 0) / np.prod(y.shape[2:])
            eval_det.update(fold_ratio, x.size(0))

            #DSC
            dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
            print('Trans dsc: {:.4f}, det: {}'.format(dsc_trans.item(), fold_ratio))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            stdy_idx += 1
        print(configs['dataset_dir_name'], model_folder)
        print('Deformed DSC: {:.6f} +- {:.6f}'.format(eval_dsc_def.avg,eval_dsc_def.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        print('deformed time: {}, std: {}'.format(eval_time.avg, eval_time.std))
        ct= np.array(all_ct)
        print(ct.mean(0))
        print(configs['dataset_dir_name'], model_folder, file=f)
        print('Deformed DSC: {:.6f} +- {:.6f}'.format(eval_dsc_def.avg,eval_dsc_def.std), file=f)
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std), file=f)
        print('deformed time: {}, std: {}'.format(eval_time.avg, eval_time.std), file=f)
        print(ct.mean(0), file=f)
# def csv_writter(line, name):
#     with open(name+'.csv', 'a') as file:
#         file.write(line)
#         file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''

    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()