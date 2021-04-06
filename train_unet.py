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
from sklearn.metrics import r2_score
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
from torch.nn import functional as F
from radam import *
from diffgrad import DiffGrad
from torch.utils.tensorboard import SummaryWriter
from skimage import io
from matplotlib import pyplot as plt
import glob
from fastai.vision import Image, unet_learner, ImageList, ItemList, ItemBase, SegmentationItemList, SegmentationLabelList, get_transforms, contrast, rand_crop, ImageImageList,  partial
from fastai.vision.models import resnet34 as model_resnet34, resnet50 as model_resnet50, darknet as darknet, resnet18 as model_resnet18
from fastai.vision.data import imagenet_stats
from fastai.layers import MSELossFlat, NormType
from fastai.basics import dataclass, Callback, Learner, Path, load_learner
from fastai.callbacks import SaveModelCallback
from fastai.callbacks import SaveModelCallback, TrainingPhase, GeneralScheduler
from fastai.callback import annealing_cos
import fastai
from fastai.basic_data import DataBunch, DatasetType, TensorDataset, DataLoader


dir_checkpoint = 'checkpoints/'


def normalize(v):
    norm = v.sum()
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v*len(v) / norm


def grayloader(path, onedim=False, bits=16):
    #r = png.Reader(path).asDirect()

    #img = np.vstack(map(np.uint16, r[2]))/65536.
    img = np.asarray(io.imread(path, as_gray=False, plugin='imageio')).astype(np.float32)/(2**bits - 1)
    #if '_mask' in path:
    #    print('{} {} {}'.format(path, np.min(img), np.max(img)))

    img = torch.Tensor(img)
    #img = normalize(img)
    if onedim is True:
        img.unsqueeze_(0)
    else:
        img = img.repeat(3, 1, 1)

    #img = Image.open(path)
    return img

@dataclass
class TensorboardLogger(Callback):
    learn: Learner
    run_name: str
    histogram_freq: int = 50
    path: str = None
    num_epoch : int = 0
    writer: SummaryWriter = None

    def __post_init__(self):
        self.path = self.path or os.path.join(self.learn.path, "runs")
        self.log_dir = os.path.join(self.path, self.run_name)

    def on_train_begin(self, **kwargs):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_epoch_end(self, **kwargs):
        logging.info("Epoch ended !")
        if self.num_epoch % 3 == 0:
            self.learn.export(os.getcwd()+'/data/' + self.run_name + '_EXPORT_{}.pth'.format(self.num_epoch))
        self.num_epoch += 1

    def on_batch_end(self, **kwargs):
        iteration = kwargs["iteration"]
        loss = kwargs["last_loss"]

        if iteration % self.histogram_freq == 0:

            self.writer.add_scalar("learning_rate", self.learn.opt.lr, iteration)
            self.writer.add_scalar("momentum", self.learn.opt.mom, iteration)

            self.writer.add_scalar("loss", loss, iteration)
        if (iteration % (self.histogram_freq * 5)) == 0:
            li = kwargs['last_input'][:3,0,:,:].unsqueeze(1)
            lt = kwargs['last_target'][:3,:-2,:,:]
            lo = kwargs['last_output'][:3,:-2,:,:]
            #plt.imshow(li[0, 0])
            #plt.show()
            self.writer.add_images('images', li, iteration)
            self.writer.add_images('masks/true', lt, iteration)
            self.writer.add_images('masks/pred', lo, iteration)

            #for name, param in self.learn.model.named_parameters():
            #    self.writer.add_histogram(name, param, iteration)


def loss_with_flag(outputs, labels):
    zero_or_one = (1.0 - labels[:, -1])
    loss_flag = ((outputs[:, -1] - labels[:, -1]) ** 2)
    # je multiplie la loss de l'angle par la taille du vecteur. Si le vecteur est petit alors l'angle a moins d'importance
    #penality_angle = (zero_or_one * F.smooth_l1_loss((outputs[:, 1] * (1-labels[:,0])),(labels[:, 1] * (1-labels[:,0]))))
    #outputs[:, 1] *=  labels[:,0]
    #labels[:, 1] *= labels[:,0]

    loss_parameters = F.smooth_l1_loss(outputs, labels, reduction='none')
    loss_parameters[:, 1, :, :] *= labels[:, 0, :, :]
    loss = (zero_or_one * loss_parameters.mean(dim=1)).mean() + loss_flag.mean()

    return loss


def flattenAnneal(learn:Learner, lr:float, n_epochs:int, start_pct:float):
    n = len(learn.data.train_dl)
    anneal_start = int(n*n_epochs*start_pct)
    anneal_end = int(n*n_epochs) - anneal_start
    phases = [TrainingPhase(anneal_start).schedule_hp('lr', lr),
           TrainingPhase(anneal_end).schedule_hp('lr', lr, anneal=annealing_cos)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(n_epochs)

class MyImageList(ImageList):
    def open(self, fn):
        filename_pickle = fn.replace('_mask.png', '_mask.npy')
        ext_data = torch.Tensor(np.load(filename_pickle))
        image = grayloader(fn, onedim=True, bits=8)[0]
        #image -= 0.5
        #image *= 1.5

        x = image[:,:,0]
        y = image[:,:,1]
        z = image[:,:,2]
        #print("Cartesian:")
        #print('{} {} {}'.format(x.min(), x.max(), x.mean()))
        #print('{} {} {}'.format(y.min(), y.max(), y.mean()))
        #print('{} {} {}'.format(z.min(), z.max(), z.mean()))


        # cylindrical coordinate system, with z always positive
        #rho = torch.sqrt(x**2 + y**2)
        #tau = (torch.asin(y/rho) / np.pi) % np.pi
        #tau[tau != tau] = 0
        ##print("Cylindrical:")
        #print('{} {} {}'.format(rho.min(), rho.max(), rho.mean()))
        #print('{} {} {}'.format(tau.min(), tau.max(), tau.mean()))
        #print('{} {} {}'.format(z.min(), z.max(), z.mean()))
        #image[:, :, 0] = rho
        #image[:, :, 1] = tau
        #image[:, :, 2] = z


        ext_data = torch.ones((image.shape[0], image.shape[1])) * ext_data
        ext_data.unsqueeze_(-1)
        image = torch.cat([image, ext_data], dim = 2)
        image = np.transpose(image, (2, 0, 1))
        #image[:-1,:,:] /= 2.

        #image[1] /= 128.0
        #image[0] /= 128.0
        #image[3] /= 5.0
        #image = torch.Tensor(image)
        #print('{} {} {}'.format(image.min(), image.max(), image.mean()))
        #plt.figure()
        #plt.imshow(image.data.cpu().numpy()[:-2, :, :].transpose((1,2,0)))
        #plt.figure()
        #plt.imshow(i.data.cpu().numpy()[:, :, :-1])
        #plt.show()
        image = Image(image)
        return image

class MyImageImageList(ImageImageList):
    _label_cls = MyImageList

    def open(self, fn):
        try:
            image = Image(grayloader(fn))
        except:
            print('Error {}'.format(fn))
            image = 0.0

        return image

def get_data(bs, size, src):

    data = (src.label_from_func(lambda x: str(x).replace('.png', '_mask.png'))
            .transform(get_transforms(do_flip = False, max_zoom=1.0, max_warp=0.0, max_rotate=0, max_lighting=0.3), tfm_y=False)
            .transform([rand_crop(), rand_crop()], tfm_y=True, size= size)
            .databunch(bs=bs, num_workers=1))

    data.c = 5
    return data

def train_unet(epochs=5, batch_size=1, lr=0.1, val_percent=0.1):
    print("Start script")
    if args.isgrid is False:
        filename = "/media/adrian/E2B45A26B459FD8B/psfmaskmoving_zernike2d_128_n_1_s_0_p_0_b_0__noise_1_2dzernike_test/"
        batch_size = int(batch_size//1.5)
    else:
        filename = "/idiap/temp/ashajkofci/psfmaskmoving_zernike2d_128_n_1_s_0_p_0_b_0__noise_1_2dzernike_train/"

        batch_size = batch_size
    os.environ['TORCH_HOME'] = os.getcwd()+'data'

    #transform = transforms.Compose([
    #    transforms.ToPILImage(),
    #    transforms.RandomCrop([450, 450]),
    #    transforms.RandomVerticalFlip(),
    #    transforms.RandomHorizontalFlip(),
    #    transforms.ToTensor(),
    #                               ])

    all_files_list = glob.glob(filename + "*/*.png")
    print('{} files found in {}'.format(len(all_files_list), filename))

    all_files_list = [x for x in all_files_list if "_mask" not in x]
    print('{} files found'.format(len(all_files_list)))

    all_files_list = sorted(all_files_list, key=lambda name: int(name[-13:-4]))
    print('{} files found'.format(len(all_files_list)))

    #all_files_list = all_files_list[:100000]
    #all_labels_list = lambda x: str(x).replace('.png', '_mask.png')

    num_files = len(all_files_list)
    print('{} files found'.format(len(all_files_list)))
    print("Convert to Dataframe")
    #df = pd.DataFrame({'data':all_files_list, 'label':all_labels_list})
    df = pd.DataFrame(all_files_list)

    print("Create transforms")
    print("Create data")

    #class MyImageList(ImageList):
    #    def open(self, fn):
    #        image = Image(grayloader(fn, onedim=True))

    #       return image

    src = (MyImageImageList.from_df(df, path='/')
            .split_by_rand_pct(val_percent))

    print("Creating dataloaders")

    data_gen = get_data(bs=batch_size, size=224, src=src)

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
    modelname='resnet34unetanneal'
    learn = unet_learner(data_gen, model_resnet34, pretrained=True, self_attention=selfattention, norm_type=NormType.Weight, loss_func = loss_with_flag, y_range = (0., 1.0))

    learn.model_dir = os.getcwd()+'data'
    learn.opt_func = optar
    print("Summary...")
    dt_string = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    #writer = SummaryWriter(comment=f'PROJ_{args.nbgrid}_LR_{lr}_BS_{batch_size}_FP_{args.fakepenalty}_N_{args.network}')
    name =f'{dt_string}_PROJ_{args.nbgrid}_LR_{lr}_BS_{batch_size}_N_{args.network}_ATT_{selfattention}_MODEL_{modelname}'
    mycallback = partial(TensorboardLogger, path='runs', run_name=name)
    learn.callback_fns.append(mycallback)
    learn.model.layers = learn.model.layers[:-1]
    print(learn.summary())
    #learn.lr_find(stop_div = False, num_it=200)
    #learn.recorder.plot(suggestion=True)
    #plt.show()

    flattenAnneal(learn, lr, epochs, 0.7)

    #learn.fit_one_cycle(epochs, max_lr = lr)
    torch.save(learn.model, os.getcwd()+'/data/'+name+'_TORCH_INTERMEDIATE.pth')
    learn.export(os.getcwd()+'/data/'+name+'_INTERMEDIATE_EXPORT.pth')

    #learn.fit_one_cycle(epochs, max_lr=lr/5.0)
    learn.unfreeze()
    flattenAnneal(learn, lr/5, epochs, 0.7)
    mycallback = partial(TensorboardLogger, path='runs', run_name=name+'_UNFREEZE')
    learn.callback_fns[-1] = mycallback

    torch.save(learn.model, os.getcwd()+'/data/'+name+'_TORCH.pth')
    learn.export( os.getcwd()+'/data/'+name+'_EXPORT.pth')
    #learn.fit_one_cycle(50, max_lr = 0.05)


def test_unet_without_gt(learn, picture_input, downsample=8, batch_size=12, picture=False):

    picture_input.unsqueeze_(dim=0)
    #picture_input = picture_input[:,:,0:224,0:224]
    picture_input = F.interpolate(picture_input, size=(224, 224), mode='bilinear', align_corners=True).float()
    #picture_input = torch.cat([picture_input, picture_input, picture_input], dim=1)
    my_dataset = TensorDataset(picture_input, picture_input)  # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size)  # create your dataloader
    my_databunch = DataBunch(train_dl=my_dataloader, test_dl=my_dataloader, valid_dl=my_dataloader)
    learn.data = my_databunch
    output = learn.get_preds(ds_type=DatasetType.Valid)[0]
    output_inter = F.interpolate(output, scale_factor=1.0 / downsample, mode='nearest')

    if picture:
        import matplotlib.pyplot as plt

        idx = 0
        plt.figure()
        plt.subplot(221)
        aa = picture_input[idx, :, :, :].data.numpy()
        im_out = np.transpose(aa, (1, 2, 0))
        plt.imshow(im_out)
        plt.title('input')

        plt.subplot(222)
        aa = output[idx, :-1, :, :].data.numpy()
        im_out = np.transpose(aa, (1, 2, 0))
        plt.imshow(im_out)
        plt.title('output')

        plt.subplot(223)
        aa = output_inter[idx, :-1, :, :].data.numpy()
        im_out = np.transpose(aa, (1, 2, 0))
        plt.imshow(im_out)
        plt.title('output_downsampled')

        plt.show()

    return output_inter.data.cpu().numpy()



def test_unet(modelname, picturenames, withdata = True, picture=False):
    #model = torch.load(modelname)
    list_images_df = pd.DataFrame(picturenames)

    src = (MyImageImageList.from_df(list_images_df, path='/').split_none())

    data_gen = get_data(bs=4, size=224, src=src)
    data_gen.ignore_empty = True


    len_data = len(data_gen.train_ds)
    learn = load_learner(path = '',file=modelname)

    #model.eval()
    # for child in model.children():
    #   if type(child) == nn.BatchNorm2d:
    #       child.track_running_stats = False

    all_outputs = []
    all_labels = []
    for picture_i in range(len_data):
        #print('Image {}'.format(picture_i))
        data = data_gen.train_ds[picture_i]
        image = data[0]
        mask = data[1].data
        #filename_mask = picturename.replace('.png', '_mask.png')
        #filename_pickle = picturename.replace('.png', '_mask.npy')
        #ext_data = torch.Tensor(np.load(filename_pickle))
        #mask = grayloader(filename_mask, onedim=True)[0]
        #mask *= 255
        #ext_data = torch.ones((mask.shape[0], mask.shape[1])) * ext_data
        #ext_data.unsqueeze_(-1)
        #mask = torch.cat([mask, ext_data], dim=2)
        #mask = np.transpose(mask, (2, 0, 1))

        output = learn.predict(image)[1]

        all_outputs.append(output)
        all_labels.append(mask)

        if picture:
            import matplotlib.pyplot as plt
            import matplotlib
            plt.rc('text', usetex=True)
            font = {'family': 'serif',
                    'weight': 'normal',
                    'size': 22}
            matplotlib.rc('font', **font)

            f, axarr = plt.subplots(1, 3,figsize=(12,6))
            image = image.data.numpy()
            image = np.transpose(image[:,:,:], (1,2,0))
            axarr[0].imshow(image)
            axarr[0].set_title('Input image')
            axarr[0].axis('off')

            mask2 = mask.data.numpy().astype(float)[:-2,:,:]
            axarr[1].imshow(np.transpose(mask2, (1,2,0)))
            axarr[1].set_title('Ground truth')
            axarr[1].axis('off')
            aa = output[:-2,:,:].data.numpy()


            axarr[2].imshow(np.transpose(aa, (1,2,0))[:,:, [1, 0, 2]])
            axarr[2].set_title('Prediction')
            axarr[2].axis('off')

            plt.show()
    a=torch.stack(all_outputs)
    b=torch.stack(all_labels)
    return a, b


def benchmark_precision_unet(folder_models,folder_images, picture=False):

    list_models = glob.glob(folder_models)
    list_images = glob.glob(folder_images + "*/*")
    list_images = [x for x in list_images if "_mask" not in x]

    list_images = sorted(list_images, key=lambda name: int(name[-13:-4]))

    randomize = np.arange(len(list_images)).astype(int)
    np.random.shuffle(randomize)
    list_images = np.asarray(list_images)
    list_images = list_images[randomize][:1000]

    #feature_names = ['focus', 'ast', 'astdirection','valid']
    feature_names = ['x', 'y', 'z','z_position', 'valid']

    all_results = {}
    all_results_list = []
    columns = []
    model_nb = 0
    for model in list_models:
        print(f"Testing model : {model}")
        results, labels = test_unet(modelname=model, picturenames=list_images, picture=picture)

        results_flat = results[labels[:,:,-1] < 1.0]
        results_flat = results_flat.reshape((results_flat.size(0), -1))
        labels_flat = labels[labels[:,:,-1] < 1.0]
        labels_flat = labels_flat.reshape((labels_flat.size(0), -1))
        r2 = r2_score(labels_flat, results_flat)
        all_results[model] = []
        for i in range(len(results)):
         if labels[i,-1].mean() < 1.0:
             score = (torch.abs(results[i] - labels[i])).mean(axis=1).mean(axis=1)
             all_results[model].append(score)
        all_results[model] = torch.stack(all_results[model]).cpu().data.numpy()
        model_nb += 1
        print('Model {} {}/{}, error = {:.5f} +/- {:.5f} (R2={})'.format(model, model_nb, len(list_models),
                                                               all_results[model][:, :-1].mean(),
                                                               all_results[model][:, :-1].std(), r2))
        temp_list = [model]
        columns = ['model']
        for i in range(all_results[model].shape[1]):
            print('Feature {} = {:.5f} +/- {:.5f}'.format(feature_names[i], all_results[model][:, i].mean(),
                                                       all_results[model][:, i].std()))
            columns.append(feature_names[i] + '_mean')
            columns.append(feature_names[i] + '_std')
            temp_list.append(all_results[model][:, i].mean())
            temp_list.append(all_results[model][:, i].std())

        all_results_list.append(temp_list)

    df = pd.DataFrame(all_results_list, columns=columns)
    df.to_csv('{}_benchmark.csv'.format(datetime.now(), index=False))

    from pytoolbox.data.load_save import pickle_save

    pickle_save('{}_benchmark.pkl'.format(datetime.now()), all_results, compressed=False)


def test_for_moving_grids(folder_images, folder_models ="data/*TORCH*"):
    from pathvalidate import sanitize_filename

    from pytoolbox.data.load_save import read_tiff
    folders = glob.glob(folder_images + "x*")
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

        filename = glob.glob(name+"/*ome.tif")
        if len(filename) == 0:
            continue
        else:
            filename = filename[0]
        filename2 = filename[:-8] + '_projection.tif'
        print('Reading : {}'.format(filename))
        if not os.path.exists(filename2):
            image = read_tiff(filename)
            print('Size = {}'.format(image.shape))

            if 'fake' in folder_images:
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

    list_models = glob.glob(folder_models)
    feature_names = ['x','y','z','z_position', 'valid']

    all_results = []
    model_nb = 0
    for model in list_models:
        print('Model {} {}/{}'.format(model, model_nb, len(list_models)))
        learn = load_learner(path='', file=model)
        learn.callbacks = []
        learn.callback_fns = []
        for exp, content in data.items():
            results = []
            for image in filenames[exp]:
                img = grayloader(image)
                results.append(test_unet_without_gt(learn=learn, picture_input=img, picture=True)[0].transpose(1,2,0).reshape(784, 4))
            labels = data[exp]
            all_results.append({'model':model, 'exp':exp, 'results':results, 'labels':labels})

        model_nb += 1
    filesave = '{}_fake_{}_get_grid_model_{}_folder_{}.pkl'.format(datetime.now(), isfake, sanitize_filename(model), sanitize_filename(folder_images))
    pickle_save(filesave, all_results, compressed=False)
    return filesave

def test_for_beads(folder_images='/media/adrian/OMENDATA/data/beads/', folder_models ="data/*TORCH*"):
    from pathvalidate import sanitize_filename
    from pytoolbox.data.load_save import pickle_save

    all_results = []
    model_nb = 0
    list_models = glob.glob(folder_models)

    for model in list_models:
        print('Model {} {}/{}'.format(model, model_nb, len(list_models)))
        learn = load_learner(path='', file=model)
        learn.callbacks = []
        learn.callback_fns = []
        folders = glob.glob(folder_images+"*")
        for name in folders:
            if '.bsh' in name:
                continue
            if '115' in name:
                continue
            array = name.split('/')[-1].split('_')
            exp = float(array[1][:-2])
            pos = float(array[2])
            results = []
            input_img = []
            filename = glob.glob(name+"/Default/*.tif")
            for file in filename:
                img = grayloader(file)
                input_img.append(img.data.cpu().numpy())
                ress = test_unet_without_gt(learn=learn, picture_input=img, downsample=1, picture=False)[0].transpose(1, 2, 0)
                results.append(
                    ress.reshape(-1, 4))
            all_results.append({'model':model, 'exp':exp, 'pos': pos, 'results':results})
            input_img = np.asarray(input_img)
            res = np.asarray(results)
            res = res.reshape((res.shape[0], ress.shape[0], ress.shape[1], ress.shape[2]))
            plt.figure()
            valid_mask = (res[:, :,:, -1] > 0.5)[:, :, :, np.newaxis]
            valid_mask = np.repeat(valid_mask, 3, axis=3)
            mean_ = np.ma.array(res[:, :,:, :-1], mask=valid_mask).mean(axis=0)
            ax = plt.subplot(131)
            ax.set_title("Mean {:.5f}".format(mean_.mean()))
            plt.imshow(mean_)

            x_position, y_position = np.meshgrid(
                np.arange(0, ress.shape[0],1),
                np.arange(0, ress.shape[1],1))

            ax.quiver(x_position, y_position, -mean_[:,:,0], mean_[:,:,1])

            #plt.title('mean')

            ax = plt.subplot(132)
            std_ = res[:, :,:, :-1].std(axis=0)

            plt.imshow(std_)
            ax.set_title("std {:.4f}".format(std_.std()))

            ax = plt.subplot(133)
            plt.imshow(img.data.cpu().numpy()[0].transpose((1,2,0)))
            ax.set_title("Input {}".format(name))


            plt.suptitle('model {}'.format(model))

            import pyqtgraph as pg
            pg.image(input_img.transpose((0,2,3,1)))
            pg.image(res[:, :,:, :-1])
            pg.image(res[:, :,:, -1])

            plt.show()

            model_nb += 1

    filesave = '{}_beads_folder_{}.pkl'.format(datetime.now(), sanitize_filename(folder_images))
    pickle_save(filesave, all_results, compressed=False)
    return filesave


def experiment_stats(filename):
    from pytoolbox.data.load_save import pickle_save, pickle_load
    from sklearn.manifold import TSNE
    stats = pickle_load(filename, compressed=False)
    feature_names = ['x','y','z','z_position', 'valid']
    from scipy.stats import gaussian_kde
    np.set_printoptions(precision=3)
    model_1 = 0
    for model_stat in stats:
        #if model_stat['exp'] > 2.0:
        #    continue
        print('testing model {} with exp {}'.format(model_stat['model'],model_stat['exp']))
        resultsx = model_stat['results']
        results = []
        labels = model_stat['labels']
        labels_colors = []
        im_i = 0
        for result in resultsx:
            mean_inter_image = result.mean(axis=0)
            std_inter_image = result.std(axis=0)
            print('Image {} : {} +/- {}'.format(im_i, mean_inter_image, std_inter_image))
            print('Label {} : {}'.format(im_i, labels[im_i]))
            if labels[im_i][-1] == 0.5:
                labels_colors.append((labels[im_i][0]/20+0.5,labels[im_i][1]/20+0.5, labels[im_i][2]/150))
                results.append(result)

            im_i += 1


        results = np.asarray(results)
        results_tsne = results.reshape((results.shape[0], results.shape[1]*results.shape[2]))
        X_embedded = TSNE(n_components=2).fit_transform(results_tsne)

        plt.figure()
        fig, ax = plt.subplots()
        plt.title('tsn for model {}'.format(model_1))
        for i in range(results.shape[0]):
            plt.scatter(X_embedded[i,0], X_embedded[i,1], color=labels_colors[i])
        #for i in range(results.shape[0]):
             #ax.annotate(labels[i], (X_embedded[i,0], X_embedded[i,1]))
        plt.show()

        results = np.mean(np.asarray(resultsx)[:,:,:-1], axis=1)
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
        model_1 += 1
    plt.show()



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default="",
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-i', '--isgrid', dest='isgrid', default=False, type=bool)
    parser.add_argument('-n', '--nbgrid', dest='nbgrid', default=0, type=int)
    parser.add_argument('-u', '--fakepenalty', dest='fakepenalty', default=0.01, type=float)
    parser.add_argument('-w', '--network', dest='network', default="unet_resnet", type=str)
    parser.add_argument('-x', '--normalization', dest='normalization', default=False, type=bool)

    return parser.parse_args()



if __name__ == '__main__':
    print("Start..")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda')
    logging.info(f'Using device {device}')

    if args.network is 'unet_resnet':
      train_unet(epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, val_percent=args.val/100)
