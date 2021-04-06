'''
Code for the implementation of
"Estimating Nonplanar Flow from 2D Motion-blurred Widefield Microscopy Images via Deep Learning"

Copyright (c) 2021 Idiap Research Institute, https://www.idiap.ch/
Written by Adrian Shajkofci <adrian.shajkofci@idiap.ch>,
All rights reserved.

This file is part of Estimating Nonplanar Flow from 2D Motion-blurred Widefield Microscopy Images via Deep Learning.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of mosquitto nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

import argparse
import logging
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import pandas as pd
from pandas.io.parsers import read_csv
from diffgrad import DiffGrad
from radam import *
from torch.utils.tensorboard import SummaryWriter
from skimage import io
from matplotlib import pyplot as plt
import glob
from fastai.vision import Image, unet_learner, ImageList, ItemList, ItemBase, SegmentationItemList, SegmentationLabelList, get_transforms, contrast, rand_crop, ImageImageList,  partial, FloatList, cnn_learner
from fastai.vision.models import resnet34 as model_resnet34, resnet50 as model_resnet50, darknet as darknet, resnet18 as model_resnet18, squeezenet1_1 as sq, xresnet34
from fastai.vision import children, create_head, flatten_model
from fastai.vision.data import imagenet_stats, normalize, normalize_funcs
from fastai.layers import MSELossFlat, NormType, FlattenedLoss
from fastai.basics import dataclass, Callback, Learner, Path
from fastai.callbacks import SaveModelCallback, TrainingPhase, GeneralScheduler
from fastai.callback import annealing_cos
from fastai.basic_train import requires_grad
from torch.nn import SmoothL1Loss, MSELoss
import gpytorch
from torch.nn.modules.loss import _Loss


def loss_with_flag(outputs, labels):
    zero_or_one = (1.0 - labels[:, -1])
    loss_flag = ((outputs[:, -1] - labels[:, -1]) ** 2).mean()
    loss_parameters = F.smooth_l1_loss(outputs, labels)
    loss = (zero_or_one * loss_parameters).mean() + loss_flag
    return loss


def arch_summary(arch):
    model = arch(False)
    tot = 0
    for i, l in enumerate(model.children()):
        n_layers = len(flatten_model(l))
        tot += n_layers
        print(f'({i}) {l.__class__.__name__:<12}: {n_layers:<4}layers (total: {tot})')


def get_groups(model, layer_groups):
    group_indices = [len(g) for g in layer_groups]
    curr_i = 0
    group = []
    for layer in model:
        group_indices[curr_i] -= len(flatten_model(layer))
        group.append(layer.__class__.__name__)
        if group_indices[curr_i] == 0:
            curr_i += 1
            print(f'Group {curr_i}:', group)
            group = []

dir_checkpoint = 'checkpoints/'

def grayloader(path, onedim=False):
    img = np.asarray(io.imread(path, as_gray=False, plugin='imageio')).astype(np.float32)/65536.
    img = torch.Tensor(img)
    if onedim is True:
        img.unsqueeze_(0)
    else:
        img = img.repeat(3, 1, 1)
    return img

@dataclass
class TensorboardLogger(Callback):
    learn: Learner
    run_name: str
    histogram_freq: int = 2
    path: str = None
    run_type: str = None

    def __post_init__(self):
        self.path = self.path or os.path.join(self.learn.path, "runs")
        self.log_dir = os.path.join(self.path, self.run_name)

    def on_train_begin(self, **kwargs):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_epoch_end(self, **kwargs):
        pass


    def on_batch_end(self, **kwargs):
        iteration = kwargs["iteration"]
        loss = kwargs["last_loss"]

        if iteration % self.histogram_freq == 0:

            self.writer.add_scalar("learning_rate", self.learn.opt.lr, iteration)
            self.writer.add_scalar("momentum", self.learn.opt.mom, iteration)

            self.writer.add_scalar("loss", loss, iteration)

            li = kwargs['last_input'][:,0,:,:]
            lt = kwargs['last_target']
            lo = kwargs['last_output']


            metric = torch.abs(lt-lo).mean()
            self.writer.add_scalar("L1norm", metric, iteration)


            if self.run_type is 'unet':
                self.writer.add_image('images', li[0].unsqueeze(0), iteration)
                self.writer.add_image('masks/true_position', lt[0,0].unsqueeze(0), iteration)
                self.writer.add_image('masks/pred_position', lo[0,0].unsqueeze(0), iteration)
            else:

                if iteration % (self.histogram_freq * 100) == 0:
                # plot the images in the batch, along with predicted and true labels
                    fig = plt.figure()
                    plt.imshow(li[0])
                    plt.title("xy_speed,z_speed,center,sinangle,cosangle,bead_size\ntarget={} \n output={}".format(lt[0].data.cpu(), lo[0].data.cpu()))
                    self.writer.add_figure('predictions',
                                    fig,
                                    global_step=iteration)
                    self.writer.add_text('target',str(lt[0].data.cpu()), iteration)
                    self.writer.add_text('output',str(lo[0].data.cpu()), iteration)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MultiResResnet(nn.Module):
    def __init__(self, pretrained, num_features, multiplier=5):
        super(MultiResResnet, self).__init__()
        self.res1 = nn.Sequential(*list(children(model_resnet34(pretrained=True))[:-2]), nn.AdaptiveMaxPool2d(1), Flatten(),nn.Linear(in_features=512, out_features=multiplier*num_features))
        self.res2 = nn.Sequential(*list(children(model_resnet34(pretrained=True))[:-2]), nn.AdaptiveMaxPool2d(1), Flatten(),nn.Linear(in_features=512, out_features=multiplier*num_features))
        self.res3 = nn.Sequential(*list(children(model_resnet34(pretrained=True))[:-2]), nn.AdaptiveMaxPool2d(1), Flatten(),nn.Linear(in_features=512, out_features=multiplier*num_features))
        requires_grad(self.res1[0:6], False)
        requires_grad(self.res2[0:6], False)
        requires_grad(self.res3[0:6], False)
        self.linear1 = nn.Sequential(nn.Dropout(0.5),
                                     nn.SELU(inplace=True),
                                     nn.Linear(multiplier*num_features*3, 100),
                                      nn.SELU(inplace=True),
                                      nn.Linear(100, num_features))

    def forward(self, x):
        x2 = F.interpolate(x, scale_factor=0.75)
        x3 = F.interpolate(x, scale_factor=0.5)
        x1_size = (x.shape[2]-x3.shape[2])//2
        x2_size = (x2.shape[2]-x3.shape[2])//2
        x1 = x[:, :, x1_size:-x1_size:, x1_size:-x1_size:]
        x2 = x2[:, :, x2_size:-x2_size:, x2_size:-x2_size:]
        x1 = self.res1(x1)
        x2 = self.res2(x2)
        x3 = self.res3(x3)
        x3[:, 0:2] *= 2
        x2[:, 0:2] *= 1.5
        x_cat = torch.cat([x1, x2, x3], dim=1)
        output = self.linear1(x_cat)
        return output

def flattenAnneal(learn:Learner, lr:float, n_epochs:int, start_pct:float):
    n = len(learn.data.train_dl)
    anneal_start = int(n*n_epochs*start_pct)
    anneal_end = int(n*n_epochs) - anneal_start
    phases = [TrainingPhase(anneal_start).schedule_hp('lr', lr),
           TrainingPhase(anneal_end).schedule_hp('lr', lr, anneal=annealing_cos)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(n_epochs)


def train_cnn_multires(epochs=5, batch_size=1, lr=0.1, val_percent=0.1):
    print("Start script")
    if args.isgrid is False:
        filename = "/media/adrian/E2B45A26B459FD8B/movementgenerator_data_realworld_big/"
        batch_size = int(batch_size//1.5)
    else:
        filename = "/idiap/temp/ashajkofci/movementgenerator_data_realworld_big/"

        batch_size = batch_size
    os.environ['TORCH_HOME'] = os.getcwd()+'/data'

    #transform = transforms.Compose([
    #    transforms.ToPILImage(),
    #    transforms.RandomCrop([450, 450]),
    #    transforms.RandomVerticalFlip(),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #                               ])

    all_files_list = glob.glob(filename + "*/*.png")
    print('{} files found in {}'.format(len(all_files_list), filename))

    all_files_list = sorted(all_files_list, key=lambda name: int(name[-13:-4]))
    print('{} files found'.format(len(all_files_list)))

    _file_csv = read_csv(os.path.expanduser(filename + "parameters.txt"), header=None)
    _labels = _file_csv.values.astype(np.float)
    print('{} labels found'.format(len(_labels)))

    print('{} files found'.format(len(all_files_list)))

    print("Convert to Dataframe")
    df = pd.DataFrame(all_files_list)

    print("Create transforms")
    print("Create data")
    num_features = 6


    class MyImageImageList(ImageList):
        _label_cls = FloatList

        def open(self, fn):
            return Image(grayloader(fn))

    def ff(input):
        out =  _labels[int(input[-13:-4])]
        out = out.tolist()
        return out

    def get_data(bs, size):
        data = (src.label_from_func(ff)
                .transform(get_transforms(do_flip = False, max_zoom=1.0, max_warp=0.0, max_rotate=0, max_lighting=0.3), tfm_y=False)
                .transform([rand_crop(), rand_crop()], tfm_y=False, size= size)
                .databunch(bs=bs).normalize(imagenet_stats, do_y=False))

        data.c = num_features
        return data

    src = (MyImageImageList.from_df(df, path='/')
            .split_by_idx(list(range(int(val_percent*len(all_files_list))))))

    print("Creating dataloaders")
    patch_size = 224
    data_gen = get_data(bs=batch_size, size=patch_size)

    #dataset = DatasetFromFolder(filename, loader = grayloader,  transform=transform, target_transform=transform)

    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train, val = rs(dataset, [n_train, n_val])
    #data = ImageDataBunch.create(train, val, bs=batch_size, num_workers=4)
    #data.c = 2
    #data.normalize(imagenet_stats)
    #data_gen.show_batch(2)
    #plt.show()
    print("Creating learner")
    #optar = partial(DiffGrad, version=1, betas=(.95, .999), eps=1e-6)
    optar = partial(Ranger, betas=(0.95, 0.99), eps=1e-6)
    selfattention=False
    modelname='xresnetmultianneal'

    #learn = cnn_learner(data_gen, MultiResResnet, cut=0, split_on=lambda m: (m[0][6], m[1]), loss_func = SmoothL1Loss())
    learn = Learner(data_gen, MultiResResnet(pretrained=True, num_features=num_features), loss_func = loss_with_flag)

    learn.model_dir = os.getcwd()+'/data'
    learn.opt_func = optar
    print("Summary...")
    dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    #writer = SummaryWriter(comment=f'PROJ_{args.nbgrid}_LR_{lr}_BS_{batch_size}_FP_{args.fakepenalty}_N_{args.network}')
    name =f'{dt_string}_PROJ_{args.nbgrid}_LR_{lr}_BS_{batch_size}_N_{args.network}_ATT_{selfattention}_MODEL_{modelname}_PATCH_{patch_size}'
    mycallback = partial(TensorboardLogger, path='runs', run_name=name)
    #learn.callback_fns.append(mycallback)
    #learn.callback_fns.append(partial(SaveModelCallback,every='improvement', name='{}/{}.pth'.format(dir_checkpoint, name)))
    #learn.model.layers = learn.model.layers[:-1]
    print(learn.summary())
    #learn.lr_find(stop_div = False, num_it=200)
    #learn.recorder.plot(suggestion=True)
    #plt.show()

    #flattenAnneal(learn, lr, epochs, 0.7)
    #learn.fit_one_cycle(epochs, max_lr = lr)
    #torch.save(learn.model, 'data/'+name+'_TORCH_INTERMEDIATE.pth')
    learn.unfreeze()
    #learn.fit_one_cycle(epochs, max_lr = lr/10)
    #learn.save(name+'_FINAL')
    flattenAnneal(learn, lr, epochs, 0.7)
    torch.save(learn.model, 'data/'+name+'_TORCH.pth')

    #learn.unfreeze()
    #learn.fit_one_cycle(50, max_lr = 0.05)


def train_cnn(epochs=5, batch_size=1, lr=0.1, val_percent=0.1):
    print("Start script")
    if args.isgrid is False:
        filename = "/media/adrian/E2B45A26B459FD8B/movementgenerator_data_realworld_big/"
        batch_size = int(batch_size//1.5)
    else:
        filename = "/idiap/temp/ashajkofci/movementgenerator_data_realworld_big/"

        batch_size = batch_size
    os.environ['TORCH_HOME'] = os.getcwd()+'/data'

    #transform = transforms.Compose([
    #    transforms.ToPILImage(),
    #    transforms.RandomCrop([450, 450]),
    #    transforms.RandomVerticalFlip(),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #                               ])

    all_files_list = glob.glob(filename + "*/*.png")
    print('{} files found in {}'.format(len(all_files_list), filename))

    all_files_list = sorted(all_files_list, key=lambda name: int(name[-13:-4]))
    print('{} files found'.format(len(all_files_list)))

    _file_csv = read_csv(os.path.expanduser(filename + "parameters.txt"), header=None)
    _labels = _file_csv.values.astype(np.float)
    print('{} labels found'.format(len(_labels)))

    print('{} files found'.format(len(all_files_list)))

    print("Convert to Dataframe")
    df = pd.DataFrame(all_files_list)

    print("Create transforms")
    print("Create data")


    class MyImageImageList(ImageList):
        _label_cls = FloatList

        def open(self, fn):
            return Image(grayloader(fn))

    def ff(input):
        out =  _labels[int(input[-13:-4])]
        out = out.tolist()
        return out

    def get_data(bs, size):
        data = (src.label_from_func(ff)
                .transform(get_transforms(do_flip = False, max_zoom=1.0, max_warp=0.0, max_rotate=0, max_lighting=0.3), tfm_y=False)
                .transform([rand_crop(), rand_crop()], tfm_y=False, size= size)
                .databunch(bs=bs).normalize(imagenet_stats, do_y=False))

        data.c = 6
        return data

    src = (MyImageImageList.from_df(df, path='/')
            .split_by_idx(list(range(int(val_percent*len(all_files_list))))))

    print("Creating dataloaders")
    patch_size = 112
    data_gen = get_data(bs=batch_size, size=patch_size)

    #dataset = DatasetFromFolder(filename, loader = grayloader,  transform=transform, target_transform=transform)

    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train, val = rs(dataset, [n_train, n_val])
    #data = ImageDataBunch.create(train, val, bs=batch_size, num_workers=4)
    #data.c = 2
    #data.normalize(imagenet_stats)
    #data_gen.show_batch(2)
    #plt.show()
    print("Creating learner")
    optar = partial(DiffGrad, version=1, betas=(.95, .999), eps=1e-6)


    selfattention=False
    modelname='resnet34'

    learn = cnn_learner(data_gen, model_resnet34, pretrained=True, loss_func = loss_with_flag)

    learn.model_dir = os.getcwd()+'/data'
    learn.opt_func = optar
    print("Summary...")
    dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    #writer = SummaryWriter(comment=f'PROJ_{args.nbgrid}_LR_{lr}_BS_{batch_size}_FP_{args.fakepenalty}_N_{args.network}')
    name =f'{dt_string}_PROJ_{args.nbgrid}_LR_{lr}_BS_{batch_size}_N_{args.network}_ATT_{selfattention}_MODEL_{modelname}_PATCH_{patch_size}'
    mycallback = partial(TensorboardLogger, path='runs', run_name=name)
    learn.callback_fns.append(mycallback)
    learn.callback_fns.append(partial(SaveModelCallback,every='improvement', name='{}/{}.pth'.format(dir_checkpoint, name)))
    #learn.model.layers = learn.model.layers[:-1]
    print(learn.summary())
    #learn.lr_find(stop_div = False, num_it=200)
    #learn.recorder.plot(suggestion=True)
    #plt.show()

    learn.fit_one_cycle(epochs, max_lr = lr)
    torch.save(learn.model, 'data/'+name+'_TORCH_INTERMEDIATE.pth')
    learn.unfreeze()
    learn.fit_one_cycle(epochs, max_lr = slice(lr/20,lr/5))
    learn.save(name+'_FINAL')
    torch.save(learn.model, 'data/'+name+'_TORCH.pth')

    #learn.unfreeze()
    #learn.fit_one_cycle(50, max_lr = 0.05)



def patch_mean(images, patch_shape):
    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    padding = tuple(side // 2 for side in patch_size)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    # Convolution with these weights will effectively compute the channel-wise means
    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.cuda()

    # Make convolution operate on single channels
    channel_selector = torch.eye(channels).bool()
    weights[~channel_selector] = 0

    result = conv(images, weights, padding=padding, bias=None)

    return result


def patch_std(image, patch_shape):
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2).sqrt()


def channel_normalize(template):
    """
    Z-normalize image channels independently.
    """
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))

    return reshaped_template.view_as(template)




def train_cnn2(epochs=5, batch_size=1, lr=0.1, val_percent=0.1):
    print("Start script")
    if args.isgrid is False:
        filename = "/media/adrian/E2B45A26B459FD8B/movementgenerator_data_multiple/"
        batch_size = int(batch_size//1.5)
    else:
        filename = "/idiap/temp/ashajkofci/movementgenerator_data_multiple/"

        batch_size = batch_size
    os.environ['TORCH_HOME'] = os.getcwd()+'/data'

    #transform = transforms.Compose([
    #    transforms.ToPILImage(),
    #    transforms.RandomCrop([450, 450]),
    #    transforms.RandomVerticalFlip(),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #                               ])

    all_files_list = glob.glob(filename + "*/*.png")
    print('{} files found in {}'.format(len(all_files_list), filename))

    all_files_list = sorted(all_files_list, key=lambda name: int(name[-13:-4]))
    print('{} files found'.format(len(all_files_list)))

    _file_csv = read_csv(os.path.expanduser(filename + "parameters.txt"), header=None)
    _labels = _file_csv.values.astype(np.float)
    print('{} labels found'.format(len(_labels)))

    print('{} files found'.format(len(all_files_list)))

    print("Convert to Dataframe")
    df = pd.DataFrame(all_files_list)

    print("Create transforms")
    print("Create data")


    class MyImageImageList(ImageList):
        _label_cls = FloatList

        def open(self, fn):
            return Image(grayloader(fn))

    def ff(input):
        out =  _labels[int(input[-13:-4])]
        out[0] /= 255.0
        out[1] /= 255.0
        #out[3] /= 5.0
        out = out.tolist()
        return out

    def get_data(bs, size):
        data = (src.label_from_func(ff)
                .transform(get_transforms(do_flip = False, max_zoom=1.0, max_warp=0.0, max_rotate=0, max_lighting=0.3), tfm_y=False)
                .transform([rand_crop(), rand_crop()], tfm_y=False, size= size)
                .databunch(bs=bs).normalize(imagenet_stats, do_y=False))

        data.c = 120
        return data

    src = (MyImageImageList.from_df(df, path='/')
            .split_by_rand_pct(val_percent))

    print("Creating dataloaders")

    data_gen = get_data(bs=batch_size, size=448)

    #dataset = DatasetFromFolder(filename, loader = grayloader,  transform=transform, target_transform=transform)

    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train, val = rs(dataset, [n_train, n_val])
    #data = ImageDataBunch.create(train, val, bs=batch_size, num_workers=4)
    #data.c = 2
    #data.normalize(imagenet_stats)
    #data_gen.show_batch(2)
    #plt.show()
    optar = partial(DiffGrad, version=1, betas=(.95, .999), eps=1e-6)


    def minmse(output, target):
        all_losses = torch.zeros(20)
        for i in range(20):
            loss =  torch.pow((output[:,i*5:i*5+4] - target[:, i*5:i*5+4]), 2).sum()
            all_losses[i] = loss
        return torch.min(all_losses)

    class SmoothL1NoOrderLoss(_Loss):

        __constants__ = ['reduction']

        def __init__(self, size_average=None, reduce=None, reduction='mean'):
            super(SmoothL1NoOrderLoss, self).__init__(size_average, reduce, reduction)

        def forward(self, input, target):
            nb_feat = 8
            nb_points = 15
            input = input.double()
            target = target.double()
            with torch.no_grad():
                target2 = target.view(-1, nb_points, nb_feat)
                #out_total = torch.zeros((target2.shape[0] * 15 * nb_feat))
                final_target_total = torch.zeros((target2.shape[0] * nb_points * nb_feat)).double().cuda()

                for batch_image in range(target2.shape[0]):
                    filter_ori = target2[batch_image].unsqueeze(1)
                    data = input.view(-1, 1, nb_points * nb_feat)[batch_image].unsqueeze(0)
                    filter = filter_ori.clone()
                    m = filter.mean(axis=2).unsqueeze(2)
                    m2 = filter.sum(axis=2).view(-1, 1, 1)
                    filter.sub_(m)
                    filter.div_(m2)
                    filter[filter != filter] = 0.0
                    filter_for_mean = torch.ones((1, 1, filter.shape[2])).double().cuda() / filter.shape[2]
                    data_group_mean2 = F.conv1d(data, filter_for_mean, padding=0, stride=filter.shape[2])
                    data_group_mean2 = data_group_mean2.repeat_interleave(filter.shape[2])
                    d2 = (data - data_group_mean2)
                    filtered_data = (F.conv1d(d2.expand((-1, 1, -1)), filter, stride=nb_feat))
                    index_min = torch.argsort(filtered_data, dim=2, descending=True)
                    res = torch.argsort(index_min)
                    final_indexes = (res[0, :] == 0).nonzero()[:, 1].view(-1)
                    final_target = filter_ori[final_indexes, :, :]
                    #out_total[batch_image*data.view(-1).shape[0]:batch_image*data.view(-1).shape[0]+data.view(-1).shape[0]] = data.view(-1)
                    final_target_total[batch_image*data.view(-1).shape[0]:batch_image*data.view(-1).shape[0]+data.view(-1).shape[0]] = final_target.view(-1)
                    # print(out)
                    # print(final_target)
                    #loss += F.smooth_l1_loss(out, final_target, reduction=self.reduction)
                    #loss += (out - final_target).pow(2).mean()

            return F.mse_loss(input.view(-1), final_target_total, reduction=self.reduction)

    class LossMatern(nn.Module):

        def __init__(self):
            super(LossMatern, self).__init__()

        def forward(self,output, target):
            covar_module = gpytorch.kernels.MaternKernel(batch_shape= torch.Size([output.shape[0]]), nu=2.5).cuda()

            class GaussianKernel():
                def __init__(self):
                    pass
                def forward(self, output, target):
                    return torch.exp(-(output-target)**2)
            #covar_module = GaussianKernel()

            nb_feat = 8
            nb_points = 15
            target = target.view(-1, nb_points, nb_feat)
            output = output.view(-1, nb_points, nb_feat)

            similarity_xx = covar_module(output, output).evaluate().mean(dim=2).mean(dim=1)
            similarity_yy = covar_module(target, target).evaluate().mean(dim=2).mean(dim=1)
            similarity_xy = covar_module(output, target).evaluate().mean(dim=2).mean(dim=1)

            loss = similarity_xx + similarity_yy - 2*similarity_xy
            return loss.mean()

    def loss_matern(output, target):
        covar_module = gpytorch.kernels.MaternKernel(batch_shape= torch.Size([output.shape[0]]), nu=2.5).cuda()

        class GaussianKernel():
            def __init__(self):
                pass
            def forward(self, output, target):
                return torch.exp(-(output-target)**2)
        #covar_module = GaussianKernel()

        nb_feat = 8
        nb_points = 15
        target = target.view(-1, nb_points, nb_feat)
        output = output.view(-1, nb_points, nb_feat)

        similarity_xx = covar_module(output, output).evaluate().mean(dim=2).mean(dim=1)
        similarity_yy = covar_module(target, target).evaluate().mean(dim=2).mean(dim=1)
        similarity_xy = covar_module(output, target).evaluate().mean(dim=2).mean(dim=1)

        loss = similarity_xx + similarity_yy - 2*similarity_xy
        return loss.mean()

        #target_full = target.view(-1, nb_feat*15)
        #loss = 0.0
        #for i in range(target.shape[0]):
        #    for ii in range(target.shape[1]):
        #        patch = target[i, ii, :][None, ...]
        #       ncc = NCC(patch)
        #        input = output[i][None,...][None, ...]
        #        ncc_results = ncc(input)
        #        position = np.unravel_index(ncc_results.argmax(), output.shape)[1]
        #        position = position - position % nb_feat
        #        loss +=((target_full[i,position:position+nb_feat] - output[i,position:position+nb_feat])**2).mean()
        #loss /= target.shape[0]
        #return loss

    selfattention=False
    modelname='resnet34'

    a, b = data_gen.one_batch()
    u = b.clone().double().cuda()
    b = b.double().cuda()
    #a = torch.Tensor([[[1,1,1], [0,0,0], [2,2,2], [0,0,0]]]).double().cuda()
    #b = torch.Tensor([[[2,2,2], [0,0,85], [1,1,1], [0,0,0]]]).double().cuda()

    b.requires_grad=True
    #a.requires_grad=True
    criterion = loss_matern
    loss = criterion(b,u)
    loss.backward()
    #res = torch.autograd.gradcheck(LossMatern(), (b, u), eps=1e-3, atol=1e-3, raise_exception=True)
    #print('Gradient check:{}'.format(res))  # res should be True if the gradients are correct.

    print("Creating learner")

    learn = cnn_learner(data_gen, model_resnet34, pretrained=True, loss_func = loss_matern)

    learn.model_dir = os.getcwd()+'/data'
    learn.opt_func = optar
    print("Summary...")
    dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    #writer = SummaryWriter(comment=f'PROJ_{args.nbgrid}_LR_{lr}_BS_{batch_size}_FP_{args.fakepenalty}_N_{args.network}')
    name =f'{dt_string}_PROJ_{args.nbgrid}_LR_{lr}_BS_{batch_size}_N_{args.network}_ATT_{selfattention}_MODEL_{modelname}'
    mycallback = partial(TensorboardLogger, path='runs', run_name=name)
    learn.callback_fns.append(mycallback)
    #learn.callback_fns.append(partial(SaveModelCallback,every='epoch', name='{}/{}.pth'.format(dir_checkpoint, name)))
    #learn.model.layers = learn.model.layers[:-1]
    print(learn.summary())
    #learn.lr_find(stop_div = False, num_it=200)
    #learn.recorder.plot(suggestion=True)
    #plt.show()

    learn.fit_one_cycle(epochs, max_lr = lr)
    torch.save(learn.model, 'data/'+name+'_TORCH_INTERMEDIATE.pth')
    learn.unfreeze()
    learn.fit_one_cycle(epochs, max_lr = slice(lr/100,lr/10))
    torch.save(learn.model, 'data/'+name+'_TORCH.pth')
    learn.save(name+'_FINAL.pth')
    #learn.unfreeze()
    #learn.fit_one_cycle(50, max_lr = 0.05)

def train_unet(epochs=5, batch_size=1, lr=0.1, val_percent=0.1):
    print("Start script")
    if args.isgrid is False:
        filename = "/media/adrian/OMENDATA/data/movementgenerator_data_multiple2/"
        batch_size = int(batch_size//1.5)
    else:
        filename = "/idiap/temp/ashajkofci/movementgenerator_data_multiple2/"

        batch_size = batch_size
    os.environ['TORCH_HOME'] = os.getcwd()+'/data'


    all_files_list = glob.glob(filename + "*/*.png")
    print('{} files found in {}'.format(len(all_files_list), filename))

    all_files_list = sorted(all_files_list, key=lambda name: int(name[-13:-4]))
    print('{} files found'.format(len(all_files_list)))

    print("Convert to Dataframe")
    df = pd.DataFrame(all_files_list)

    print("Create transforms")
    print("Create data")

    class MyImageList(ImageList):
        def open(self, fn):
            image = np.load(fn)['arr_0']
            image = np.transpose(image, (2, 0, 1))
            image[1] /= 128.0
            image[0] /= 128.0
            image[3] /= 5.0
            image = torch.Tensor(image)
            #print('{} {} {}'.format(image.min(), image.max(), image.mean()))

            image = Image(image)
            return image

    class MyImageImageList(ImageImageList):
        _label_cls = MyImageList

        def open(self, fn):
            return Image(grayloader(fn))

    def get_data(bs, size):
        data = (src.label_from_func(lambda x: str(x).replace('.png', '_mask.npy.npz'))
                .transform(get_transforms(do_flip = False, max_zoom=1.0, max_warp=0.0, max_rotate=0, max_lighting=0.3), tfm_y=False)
                .transform([rand_crop(), rand_crop()], tfm_y=True, size= size)
                .databunch(bs=bs).normalize(imagenet_stats, do_y=False))

        data.c = 4
        return data

    src = (MyImageImageList.from_df(df, path='/')
            .split_by_rand_pct(val_percent))

    print("Creating dataloaders")

    data_gen = get_data(bs=batch_size, size=448)

    #dataset = DatasetFromFolder(filename, loader = grayloader,  transform=transform, target_transform=transform)

    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train, val = rs(dataset, [n_train, n_val])
    #data = ImageDataBunch.create(train, val, bs=batch_size, num_workers=4)
    #data.c = 2
    #data.normalize(imagenet_stats)
    #data_gen.show_batch(2)
    #plt.show()
    print("Creating learner")
    optar = partial(DiffGrad, version=1, betas=(.95, .999), eps=1e-6)




    selfattention=False
    modelname='resnet34'

    learn = unet_learner(data_gen, model_resnet34, pretrained=True, loss_func = MSELossFlat(), self_attention=False)

    learn.model_dir = os.getcwd()+'/data'
    learn.opt_func = optar
    print("Summary...")
    dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    name =f'{dt_string}_PROJ_{args.nbgrid}_LR_{lr}_BS_{batch_size}_N_{args.network}_ATT_{selfattention}_MODEL_{modelname}'
    mycallback = partial(TensorboardLogger, path='runs', run_name=name, run_type='unet')
    learn.callback_fns.append(mycallback)
    learn.callback_fns.append(partial(SaveModelCallback,every='improvement', name='{}/{}.pth'.format(dir_checkpoint, name)))
    #learn.model.layers = learn.model.layers[:-1]
    print(learn.summary())
    #learn.lr_find(stop_div = False, num_it=200)
    #learn.recorder.plot(suggestion=True)
    #plt.show()

    learn.fit_one_cycle(epochs, max_lr = lr)
    torch.save(learn.model, 'data/'+name+'_TORCH_INTERMEDIATE.pth')
    learn.unfreeze()
    learn.fit_one_cycle(epochs, max_lr = slice(lr/50,lr/5))
    learn.save(name+'_FINAL')
    torch.save(learn.model, 'data/'+name+'_TORCH.pth')
    #learn.unfreeze()
    #learn.fit_one_cycle(50, max_lr = 0.05)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=60,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=120,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default="",
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=2.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-i', '--isgrid', dest='isgrid', default=False, type=bool)
    parser.add_argument('-n', '--nbgrid', dest='nbgrid', default=0, type=int)
    parser.add_argument('-w', '--network', dest='network', default="cnnmultires", type=str)

    return parser.parse_args()


def input_to_vector(input):
   cosalpha = input[:,:,3]
   sinalpha = input[:,:,4]
   speedxy = input[:,:,0]
   center = input[:,:,2]
   valid = input[:, :, 5]
   x = sinalpha*speedxy
   y = cosalpha*speedxy
   z = input[:,:,1]
   z_position = center - 0.5
   return torch.stack([x,y,z,z_position, valid], dim=2)[0]


def test_cnn_multires(modelname='data/28-01-2020-20:05:47_PROJ_7004_LR_0.001_BS_96_N_cnn_ATT_False_MODEL_resnet34_PATCH_224_TORCH', picturenames=['000000099.png'], picture=False):
    model = torch.load(modelname)

    model.eval()
    #for child in model.children():
    #   if type(child) == nn.BatchNorm2d:
    #       child.track_running_stats = False

    all_outputs = []
    for picturename in picturenames:
        #print('Image {}'.format(picturename))
        im = io.imread(picturename)/65536.0
        image = torch.Tensor([im, im , im]).cuda()
        t = transforms.Normalize(mean = imagenet_stats[0], std = imagenet_stats[1])
        image = t(image)
        image = image.unsqueeze(0)
        window_size = 224
        crop_x = -(image.shape[2]%(window_size//2))
        if crop_x is 0:
            crop_x = image.shape[2]
        crop_y = -(image.shape[3] % (window_size // 2))
        if crop_y is 0:
            crop_y = image.shape[3]
        image = image[:,:,:crop_x,:crop_y]
        with torch.no_grad():
            windows = F.unfold(image, kernel_size=(window_size,window_size), stride=int(window_size/2))
            windows = windows.view(-1, 3, window_size, window_size, windows.shape[-1])
            batch_size = 32
            nb_features = 6
            num_batches = int(np.ceil(windows.shape[-1]/batch_size))
            output_full = torch.zeros((windows.shape[0], windows.shape[-1], nb_features))
            for i in range(num_batches):
                if i == num_batches-1:
                    batch_size_eff = windows.shape[-1]%batch_size
                    if batch_size_eff == 0:
                        batch_size_eff = batch_size
                else:
                    batch_size_eff = batch_size
                #print("taking images {} to {}".format(i*batch_size,i*batch_size+batch_size_eff))
                input = windows[0, :, :, :, i*batch_size:i*batch_size+batch_size_eff].permute(dims=(3,0,1,2))
                #input = windows[0,i*batch_size:i*batch_size+batch_size_eff].reshape(-1, 3, window_size, window_size)
                #input = input.repeat(1,3,1,1)
                output = model(input)
                output_full[0,i*batch_size:i*batch_size+batch_size_eff] = output


        all_outputs.append(input_to_vector(output_full).numpy())
    if picture:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        x_position, y_position = np.meshgrid(np.arange(window_size//2,image.shape[2]-window_size//2+1,window_size//2),
                              np.arange(window_size//2,image.shape[3]-window_size//2+1,window_size//2))
        x_position = x_position.flatten()
        y_position = y_position.flatten()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # todo: calibrate normalization values
        z_position = all_outputs[-1][:,3]*10
        x = all_outputs[-1][:,0]*100
        y = all_outputs[-1][:,1]*100
        z = all_outputs[-1][:,2]*2
        ax.set_zlim(-10, 10)
        ax.quiver(x_position, y_position, z_position, x, y, z, length=1, arrow_length_ratio =0.2, normalize=False)

        plt.show()
    else:
        return all_outputs


def benchmark_precision(folder_models = "data/*TORCH*", folder_images = '/media/adrian/E2B45A26B459FD8B/movementgenerator_data_realworld_big/'):
    list_models = glob.glob(folder_models)
    list_images = glob.glob(folder_images+"*/*")
    list_images = sorted(list_images, key=lambda name: int(name[-13:-4]))

    _file_csv = read_csv(os.path.expanduser(folder_images + "parameters.txt"), header=None)
    _labels = _file_csv.values.astype(np.float)
    print('{} labels found'.format(len(_labels)))
    labels = input_to_vector(torch.Tensor([_labels])).numpy()

    randomize = np.arange(len(list_images)).astype(int)
    np.random.shuffle(randomize)
    list_images = np.asarray(list_images)
    list_images = list_images[randomize][:1000]
    labels = labels[randomize][:1000]
    feature_names = ['x','y','z','z_position', 'valid']

    all_results = {}
    all_results_list = []

    model_nb = 0
    for model in list_models:
        print('Testing model {}'.format(model))
        results = test_cnn_multires(modelname=model, picturenames=list_images, picture=False)
        all_results[model] = []
        for i in range(len(results)):
            score =[]
            x = np.repeat(labels[i, :][np.newaxis, :], results[i].shape[0], axis=0)
            y = results[i][:, :]
            if labels[i, -1] < 1.0:
                score.append((np.abs(x-y)).mean(axis=0))
                all_results[model].append(score)
        all_results[model] = np.array(all_results[model])[:,0]
        model_nb += 1
        print('Model {} {}/{}, error = {:.5f} +/- {:.5f} '.format(model, model_nb, len(list_models), all_results[model][:, :-1].mean(),  all_results[model][:, :-1].std()))
        temp_list = [model]
        columns = ['model']
        for i in range(all_results[model].shape[1]):
            print('Feature {} = {:.5f} +/- {:.5f}'.format(feature_names[i], all_results[model][:,i].mean(),  all_results[model][:, i].std()))
            columns.append(feature_names[i] + '_mean')
            columns.append(feature_names[i] + '_std')
            temp_list.append(all_results[model][:, i].mean())
            temp_list.append(all_results[model][:, i].std())

        all_results_list.append(temp_list)

    df = pd.DataFrame(all_results_list, columns=columns)
    df.to_csv('{}_benchmark.csv'.format(datetime.now(), index=False))

    from pytoolbox.data.load_save import pickle_save

    pickle_save('{}_benchmark.pkl'.format(datetime.now()), all_results, compressed=False)


def get_experiment_images(folder, folder_models = "data/*TORCH*"):
    from pytoolbox.data.load_save import read_tiff
    folders = glob.glob(folder+"x*")
    data = {}
    filenames = {}
    isfake = False
    for name in folders:
        if '.bsh' in name:
            continue
        array = name.split('/')[-1].split('_')
        x = float(array[1])
        y = float(array[3])
        z = float(array[5])
        if x < 1.0 and y < 1.0 and z < 1.0:
            continue
        z_position = float(array[7])
        exp = float(array[9])

        if exp > 80.0:
            continue

        filename = glob.glob(name+"*/*ome.tif")[0]
        filename2 = filename[:-8] + '_projection.tif'
        print('Reading : {}'.format(filename))
        if not os.path.exists(filename2):
            image = read_tiff(filename)
            print('Size = {}'.format(image.shape))

            if 'fake' in folder:
                isfake = True
                image = image[2:-2]
                image = np.mean(image, axis=0).astype(np.int16)
            else:
                num_images = image.shape[0]
                if num_images is 1:
                    continue

                if num_images is 3 or num_images is 4:
                    idx = 1
                elif num_images is 5:
                    idx = 2
                elif num_images is 8 or num_images is 9:
                    idx = 4
                else:
                    idx = num_images//2
                image = image[idx].astype(np.int16)

            io.imsave(filename2, image)
        if exp not in data.keys():
            data[exp] = []
            filenames[exp]= []

        data[exp].append([x,y,z,z_position])
        filenames[exp].append(filename2)

    from pytoolbox.data.load_save import pickle_save
    #pickle_save('data.pkl', data, compressed=False)


    list_models = glob.glob(folder_models)
    feature_names = ['x','y','z','z_position', 'valid']

    all_results = []
    model_nb = 0
    for model in list_models:
        print('Model {} {}/{}'.format(model, model_nb, len(list_models)))

        for exp, content in data.items():
            results = test_cnn_multires(modelname=model, picturenames=filenames[exp])
            labels = data[exp]
            all_results.append({'model':model, 'exp':exp, 'results':results, 'labels':labels})

        model_nb += 1
    filesave = '{}_fake_{}_get_experiment_images.pkl'.format(datetime.now(), isfake)
    pickle_save(filesave, all_results, compressed=False)
    return filesave

def experiment_stats(filename):
    from pytoolbox.data.load_save import pickle_save, pickle_load
    stats = pickle_load(filename, compressed=False)
    feature_names = ['x','y','z','z_position', 'valid']
    from scipy.stats import gaussian_kde

    for model_stat in stats:
        #if model_stat['exp'] > 2.0:
        #    continue
        print('testing model {} with exp {}'.format(model_stat['model'],model_stat['exp']))
        results = model_stat['results']
        results = np.mean(np.asarray(results)[:,:,:-1], axis=1)
        labels = np.asarray(model_stat['labels'])
        for i in range(results.shape[-1]):
            plt.figure()
            # Calculate the point density
            x = labels[:,i]
            y = np.abs(results[:,i])

            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

            plt.scatter(x, y, c=z, s=100, edgecolors='')
            plt.title('exp {} feature {}'.format(model_stat['exp'],feature_names[i]))
    plt.show()

if __name__ == '__main__':
    print("Start..")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda')
    logging.info(f'Using device {device}')

    if args.network is 'unet':
        train_unet(epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, val_percent=args.val/100)
    elif args.network is 'cnn':
        train_cnn(epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, val_percent=args.val/100)
    elif args.network is 'cnn2':
        train_cnn2(epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, val_percent=args.val / 100)
    elif args.network is 'cnnmultires':
        train_cnn_multires(epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, val_percent=args.val / 100)
