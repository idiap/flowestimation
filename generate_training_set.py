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

from toolbox import *

import pickle
from pandas.io.parsers import read_csv
import glob
import random
from matplotlib import pyplot as plt
import numpy as np
import io as _io
from skimage.draw import circle_perimeter_aa, circle
import numpy as np
from skimage import io, transform
import argparse
import png
from scipy.ndimage import shift, rotate
from scipy.ndimage.interpolation import zoom
from skimage.transform import rescale, resize

def get_parser():
    parser = argparse.ArgumentParser(description='Create dataset for regression of PSFS')
    parser.add_argument('--imprint_size', dest='imprint_size', type=int, default=223)
    parser.add_argument('--stack_size', dest= 'stack_size', type=int, default=64)
    parser.add_argument('--psf_size', dest='psf_size', type=int, default=127)
    parser.add_argument('--number', dest='number', type=int, default=5000)
    parser.add_argument('--output', dest='output_dir', type=string)
    parser.add_argument('--input', dest='input_dir', type=string)

    return parser


def create_one_imprint(xy_speed = 50, bead_size = 0, center = 0.5, z_speed = 1.0):
    focus_z = np.linspace(z_speed*(-2 + 2*center), z_speed*(2 - 2*(1-center)), args.stack_size)
    stack_z = np.linspace(-1.0,1.0, args.stack_size)
    pad = args.imprint_size - args.psf_size
    angle_generation = np.deg2rad(0)

    params = Params()
    params.magnification = 20
    params.n = 1.33
    params.na = 0.45
    params.wavelength = 500
    params.pixelsize = 45
    params.tubelength = 200
    params.sph = 0.0
    params.size = args.psf_size
    params.ast = 0

    psf_3d = []
    ii = 0
    for i in stack_z:
        params.focus = focus_z[ii]
        psf, wavefront, pupil_diameter = get_psf(params)
        #plt.imshow(psf)
        #plt.show()
        psf = np.pad(psf, pad, mode='constant')
        psf = shift(psf, (np.cos(angle_generation) * 50 * xy_speed * i, np.sin(angle_generation) * 50 * xy_speed * i), order=3)
        psf_3d.append(psf)
        ii += 1

    psf_3d = np.asarray(psf_3d)
    psf_3d_projection = np.sum(psf_3d, axis=0)
    img = np.zeros((args.imprint_size, args.imprint_size))
    center = (args.imprint_size//2+1, args.imprint_size//2+1)
    img[center] = 1.0
    bead_size = int(np.floor(bead_size * 5))
    if bead_size > 0:
        rr, cc, val = circle_perimeter_aa(center[0], center[1], bead_size)
        img[rr, cc] = val * 1.0
        rr, cc = circle(center[0], center[1], bead_size)
        img[rr, cc] =  1.0
    img = normalize(img)


    img = convolve(img, psf_3d_projection, padding='reflect')*args.stack_size

    img = resize(img, (args.imprint_size+1, args.imprint_size+1))
    #write_tiff_stack('psf3d.tiff', psf_3d)
    #io.imsave('psf3d_projection.tiff', psf_3d_projection)
    #plt.imshow(img)
    #plt.show()
    return img

def create_one_psf(xy_speed = 50, center = 0.5, z_speed = 1.0):
    focus_z = np.linspace(z_speed*(-2 + 2*center), z_speed*(2 - 2*(1-center)), args.stack_size)
    stack_z = np.linspace(-1.0,1.0, args.stack_size)
    pad = args.imprint_size - args.psf_size
    angle_generation = np.deg2rad(0)

    params = Params()
    params.magnification = 20
    params.n = 1.33
    params.na = 0.45
    params.wavelength = 500
    params.pixelsize = 45
    params.tubelength = 200
    params.sph = 0.0
    params.size = args.psf_size
    params.ast = 0.0

    psf_3d = []
    ii = 0
    for i in stack_z:
        params.focus = focus_z[ii]
        psf, wavefront, pupil_diameter = get_psf(params)
        #plt.imshow(psf)
        #plt.show()
        psf = np.pad(psf, pad, mode='constant')
        psf = shift(psf, (np.cos(angle_generation) * 50 * xy_speed * i, np.sin(angle_generation) * 50 * xy_speed * i), order=3)
        psf_3d.append(psf)
        ii += 1

    psf_3d = np.asarray(psf_3d)
    #write_tiff_stack('psf3d_ast.tiff', psf_3d)
    psf_3d_projection = np.sum(psf_3d, axis=0)
    psf_3d_projection = normalize(psf_3d_projection)
    #plt.imshow(psf_3d_projection)
    #plt.show()
    return psf_3d_projection

def do_generation():

    create_dir(output_dir)
    list_file = open("{}/parameters.txt".format(output_dir), 'w')

    ii = 0
    for i in range(args.number):

        if ii%1000 == 0:
            create_dir(output_dir+'/{}'.format(ii//1000))
        cur_dir = output_dir+'/{}'.format(ii//1000)

        bead_size = rand_float(0,1)[0]
        xy_speed = rand_float(0.01, 0.5)[0]
        center = rand_float(0.3,0.7)[0]
        z_speed = rand_float(0,1.5)[0]


        img_src = create_one_imprint(xy_speed = xy_speed,  bead_size=bead_size, center=center, z_speed=z_speed)
        for a in range(15):
            im = img_src.copy()

            xy_angle = rand_float(-90, 90)[0]
            im = rotate(im, xy_angle, reshape=False)
            im = shift(im, (rand_float(-10,10, 2)))

            if im.min() < 0:
                im -= im.min()
            if im.max() > 1:
                im /= im.max()


            coeff = rand_float(0.4, 0.99)
            im *= coeff

            im = noisy(im, 'poisson', rand_float(0, 20))
            im = noisy(im, 'gauss', rand_float(0.0000001, 0.001))

            if im.min() < 0:
                im -= im.min()
            if im.max() > 1:
                im /= im.max()
            im *= 0.99
            im = to_16_bit(im)

            filename = "{}/{:09d}.png".format(cur_dir, ii)

            with open(filename, 'wb') as f:
                writer = png.Writer(width=im.shape[1], height=im.shape[0], bitdepth=16, greyscale=True, compression=9)
                writer.write(f, im)

            list_file.write('{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(xy_speed, z_speed, center, np.cos(np.deg2rad(xy_angle)), np.sin(np.deg2rad(xy_angle)), bead_size))
            #im -=1
            #plt.imshow(im)
            #plt.title('xy speed {:.2f} xy angle {:.2f} z speed {:.2f} center {:.2f} bead size{:.1f}'.format(xy_speed, xy_angle, z_speed, center, bead_size))
            #plt.show()

            ii += 1
            if ii % 1000 == 0:
                create_dir(output_dir + '/{}'.format(ii // 1000))
            if ii > args.number:
                break
        if ii > args.number:
            break
    list_file.close()


def do_generation_realworld_big():

    create_dir(output_dir)
    list_file = open("{}/parameters.txt".format(output_dir), 'w')

    file_list = glob.glob("/media/adrian/ext4data/data/images_texture/*.tif")
    random.shuffle(file_list)
    print("Images found {}".format(len(file_list)))
    all_images = []
    for file in file_list:
        print('Loading file {}'.format(file))
        original_img = io.imread(file, as_grey=True)
        if np.min(original_img.shape) < 500:
            print('File size minimum : {}'.format(np.min(original_img.shape)))
            continue
        for u in range(5):
            all_images.append(random_crop(original_img, 500, True))

    ii = 0
    for i in range(args.number):

        if ii%1000 == 0:
            create_dir(output_dir+'/{}'.format(ii//1000))

        cur_dir = output_dir+'/{}'.format(ii//1000)

        xy_speed = rand_float(0.01, 0.8)[0]
        center = rand_float(0.3,0.7)[0]
        z_speed = rand_float(0,2.0)[0]

        is_bead = False
        psf = create_one_psf(xy_speed = xy_speed, center=center, z_speed=z_speed)

        for a in range(25):
            if not is_bead:
                img = random.choice(all_images).copy()
                img = convolve(img, psf, padding='reflect') * args.stack_size
            xy_angle = rand_float(-90, 90)[0]
            im = rotate(img, xy_angle, reshape=False)
            if is_bead:
                im = shift(im, (rand_float(-10,10, 2)))
            if not is_bead:
                im = unpad(im, 80)

            if im.min() < 0:
                im -= im.min()
            if im.max() > 1:
                im /= im.max()


            coeff = rand_float(0.4, 0.99)
            im *= coeff

            im = noisy(im, 'poisson', rand_float(0, 20))
            im = noisy(im, 'gauss', rand_float(0.0000001, 0.001))

            if im.min() < 0:
                im -= im.min()
            if im.max() > 1:
                im /= im.max()
            im *= 0.99
            im = to_16_bit(im)

            invalid = 0
            #print('{} {}'.format(im.mean(), im.var()))

            if im.mean() < 10000 or im.var() < 3000000:
                invalid = 1

            filename = "{}/{:09d}.png".format(cur_dir, ii)

            with open(filename, 'wb') as f:
                writer = png.Writer(width=im.shape[1], height=im.shape[0], bitdepth=16, greyscale=True, compression=9)
                writer.write(f, im)

            list_file.write('{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{}\n'.format(xy_speed, z_speed, center, np.cos(np.deg2rad(xy_angle)), np.sin(np.deg2rad(xy_angle)), invalid))
            #im -=1
            #plt.imshow(im)
            #plt.title('xy speed {:.2f} xy angle {:.2f} z speed {:.2f} center {:.2f}'.format(xy_speed, xy_angle, z_speed, center))
            #plt.show()

            ii += 1
            if ii % 1000 == 0:
                create_dir(output_dir + '/{}'.format(ii // 1000))
            if ii > args.number:
                break
        if ii > args.number:
            break
    list_file.close()





def do_generation_realworld(input_images_directory):

    create_dir(output_dir)
    list_file = open("{}/parameters.txt".format(output_dir), 'w')

    file_list = glob.glob(input_images_directory+"/*.jpg")
    random.shuffle(file_list)
    all_images = []
    for file in file_list:
        print('Loading file {}'.format(file))
        original_img = io.imread(file, as_grey=True)
        if np.min(original_img.shape) < 384:
            continue
        for u in range(5):
            all_images.append(random_crop(original_img, 384, True))

    ii = 0
    for i in range(args.number):

        if ii%1000 == 0:
            create_dir(output_dir+'/{}'.format(ii//1000))

        cur_dir = output_dir+'/{}'.format(ii//1000)

        xy_speed = rand_float(0.01, 0.5)[0]
        center = rand_float(0.3,0.7)[0]
        z_speed = rand_float(0,1.5)[0]

        is_bead = False
        if ii%2 == 0:
            psf = create_one_psf(xy_speed = xy_speed, center=center, z_speed=z_speed)
        else:
            img = create_one_imprint(xy_speed = xy_speed,  bead_size=rand_float(0,1)[0], center=center, z_speed=z_speed)
            is_bead = True

        for a in range(25):
            if not is_bead:
                img = random.choice(all_images).copy()
                img = convolve(img, psf, padding='reflect') * args.stack_size
            xy_angle = rand_float(-90, 90)[0]
            im = rotate(img, xy_angle, reshape=False)
            if is_bead:
                im = shift(im, (rand_float(-10,10, 2)))
            if not is_bead:
                im = unpad(im, 50)

            if im.min() < 0:
                im -= im.min()
            if im.max() > 1:
                im /= im.max()


            coeff = rand_float(0.4, 0.99)
            im *= coeff

            im = noisy(im, 'poisson', rand_float(0, 20))
            im = noisy(im, 'gauss', rand_float(0.0000001, 0.001))

            if im.min() < 0:
                im -= im.min()
            if im.max() > 1:
                im /= im.max()
            im *= 0.99
            im = to_16_bit(im)

            filename = "{}/{:09d}.png".format(cur_dir, ii)

            with open(filename, 'wb') as f:
                writer = png.Writer(width=im.shape[1], height=im.shape[0], bitdepth=16, greyscale=True, compression=9)
                writer.write(f, im)

            list_file.write('{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(xy_speed, z_speed, center, np.cos(np.deg2rad(xy_angle)), np.sin(np.deg2rad(xy_angle))))
            #im -=1
            #plt.imshow(im)
            #plt.title('xy speed {:.2f} xy angle {:.2f} z speed {:.2f} center {:.2f}'.format(xy_speed, xy_angle, z_speed, center))
            #plt.show()

            ii += 1
            if ii % 1000 == 0:
                create_dir(output_dir + '/{}'.format(ii // 1000))
            if ii > args.number:
                break
        if ii > args.number:
            break
    list_file.close()


def do_generation_multiple():

    create_dir(output_dir)
    img_final_size = 224*2
    create_dir(output_dir)
    list_file = open("{}/parameters.txt".format(output_dir), 'w')

    ii = 0
    for i in range(args.number):

        if ii%1000 == 0:
            create_dir(output_dir+'/{}'.format(ii//1000))
        cur_dir = output_dir+'/{}'.format(ii//1000)


        data_string = np.zeros(15*8)
        nb_particles = rand_int(0,14)
        img_srcs = []
        for aaa in range(nb_particles[0]):

            xy_speed = rand_float(0.1, 1)[0]
            bead_size = rand_float(0, 1.2)[0]
            center = rand_float(0.2, 0.8)[0]
            z_speed = rand_float(0.5, 1.8)[0]
            img_srcs.append(create_one_imprint(xy_speed=xy_speed, bead_size=bead_size, center=center, z_speed=z_speed))

        for aa in range(50):
            img_final = np.zeros((img_final_size, img_final_size))
            label_final = np.zeros((img_final_size, img_final_size, 7))
            for aaa in range(nb_particles[0]):
                point_blurry = img_srcs[aaa].copy()

                angle = rand_float(-90, 90)[0]
                point_blurry = rotate(point_blurry, angle, reshape=False)
                coeff = rand_float(0.5, 0.99)
                point_blurry *= coeff
                point_blurry = unpad(point_blurry, point_blurry.shape[0]//4)

                coords = rand_int(0, img_final_size-point_blurry.shape[0], 2)
                img_final[coords[0]:coords[0] + point_blurry.shape[0], coords[1]:coords[1] + point_blurry.shape[1]] += point_blurry

                label_final[coords[0] + point_blurry.shape[0] // 2, coords[1] + point_blurry.shape[1] // 2, 0] = 1
                label_final[coords[0] + point_blurry.shape[0] // 2, coords[1] + point_blurry.shape[1] // 2, 1] = xy_speed
                label_final[coords[0] + point_blurry.shape[0] // 2, coords[1] + point_blurry.shape[1] // 2, 2] = z_speed
                label_final[coords[0] + point_blurry.shape[0] // 2, coords[1] + point_blurry.shape[1] // 2, 3] = center
                label_final[coords[0] + point_blurry.shape[0] // 2, coords[1] + point_blurry.shape[1] // 2, 4] = np.cos(np.deg2rad(angle))
                label_final[coords[0] + point_blurry.shape[0] // 2, coords[1] + point_blurry.shape[1] // 2, 5] = np.sin(np.deg2rad(angle))
                label_final[coords[0] + point_blurry.shape[0] // 2, coords[1] + point_blurry.shape[1] // 2, 6] = bead_size

                #print('{:.3f},{:.3f},{:.3f},{} in coords {}\n'.format(xy_speed, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), bead_size, coords))

                data_string[aaa*5] = coords[0] + point_blurry.shape[0] // 2
                data_string[aaa*5+1] = coords[1] + point_blurry.shape[1] // 2
                data_string[aaa * 5 + 2] = xy_speed
                data_string[aaa * 5 + 3] = z_speed
                data_string[aaa * 5 + 4] = center
                data_string[aaa * 5 + 5] = np.cos(np.deg2rad(angle))
                data_string[aaa * 5 + 6] = np.sin(np.deg2rad(angle))
                data_string[aaa * 5 + 7] = bead_size

            if img_final.min() < 0:
                img_final -= img_final.min()
            if img_final.max() > 1:
                img_final /= img_final.max()

            for u in range(2):
                img = img_final.copy()

                coeff = rand_float(0.5, 0.99)
                img *= coeff

                img = noisy(img, 'poisson', rand_float(0, 20))
                img = noisy(img, 'gauss', rand_float(0.0000001, 0.001))

                if img.min() < 0:
                    img -= img.min()
                if img.max() > 1:
                    img /= img.max()
                    img *= 0.99
                img = to_16_bit(img)


                filename = "{}/{:09d}.png".format(cur_dir, ii)
                filename_label = "{}/{:09d}_mask.npy".format(cur_dir, ii)

                with open(filename, 'wb') as f:
                    writer = png.Writer(width=img.shape[1], height=img.shape[0], bitdepth=16, greyscale=True, compression=9)
                    writer.write(f, img)

                #label_final = np.swapaxes(label_final, 2,0)
                #label_final = np.swapaxes(label_final, 1,2)
                np.savez_compressed(filename_label, label_final)

                #im -=1
                #plt.imshow(im)
                #plt.show()
                list_file.write('{}\n'.format(str(data_string.tolist())[1:-1]))
                print("Writing file {}".format(filename))
                ii += 1
                if ii % 1000 == 0:
                    create_dir(output_dir + '/{}'.format(ii // 1000))

    list_file.close()

def poiseuille_flow_vector_field(max_velocity, radius, angle_rotation):
    x = np.linspace(0,10,100)
    y = np.linspace(0,10,100)
    z = np.linspace(0,10,100)

    v = np.asarray([0, 0, max_velocity*(radius-z)**2])
    rotation_matrix = np.asarray([
        [np.cos(angle_rotation), 0, -np.sin(angle_rotation)],
        [0, 1, 0],
        [np.sin(angle_rotation), 0, np.cos(angle_rotation)]])
    projected_vector = rotation_matrix * v

    x_,y_ = np.meshgrid(y, z)
    plt.quiver(x_, y_, projected_vector[1], projected_vector[2])
    plt.show()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    input_dir = args.input_dir
    do_generation_realworld_big(input_dir)
