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

from train import *
if __name__ == '__main__':

   # Getting a graph with the accuracy statistics
   experiment_stats('trained_model.pkl')
   # Testing the Unet model with moving grid or beads
   test_for_beads(folder_models="/models/*")
   test_for_moving_grids(folder_models="data/*9011*EXPORT_37*", folder_images='/media/adrian/OMENDATA/data/move_with_acquisition_all_with_exp3/')
   # Test one Unet Model
   test_unet(learn="model.pkl", picture_input = 'images/*')
   # Benchmark multiple models to select the best CNN model
   benchmark_precision(folder_models = "data/*TORCH*", folder_images = '/media/adrian/E2B45A26B459FD8B/movementgenerator_data_realworld_big_test/')
