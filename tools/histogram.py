"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

from matplotlib import pyplot as plt
import torchvision.transforms as T
from skimage import io


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layers, selected_filter, img_path = "./starry_night.jpg"):
        self.model = model
        self.model.eval()
        self.selected_layers = selected_layers
        self.selected_filter = selected_filter
        # dict for storing the output features
        self.conv_output = {}
        # Create the folder based on the input image name to export images if not exists
        s_a = img_path.split('/')[-1].split('.')[0] 
        s_b = '_results'
        self.output_path = ''.join([s_a, s_b])
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        # transform the input image based on VGG requirement
        transform = T.Compose([T.ToTensor(), 
                               T.Normalize(mean = [0.485, 0.456, 0.406], 
                                           std = [0.229, 0.224, 0.225])])
        self.image = transform(io.imread(img_path)).unsqueeze_(0)

    def hook_layer(self, layer):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output[layer] = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[layer].register_forward_hook(hook_function)
    
    # helper function to create and save histogram
    def create_and_save_histogram(self, data, bin_num, fname):
        plt.figure(figsize=(5, 2.5))
        plt.hist(data, bin_num, density=True)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Histogram of {}'.format(fname.split('.')[0]))
        plt.grid(True)
        plt.savefig(os.path.join(self.output_path, fname))

    def visualise_layer_with_hooks(self, bin_num):
        # Hook the selected layer
        [self.hook_layer(i) for i in self.selected_layers]
        # get the forward stop layer
        final_layer = max(self.selected_layers)

        # Assign create image to a variable to move forward in the model
        x = self.image
        for index, layer in enumerate(self.model):
            # Forward pass layer by layer
            # x is not used after this point because it is only needed to trigger
            # the forward hook function
            x = layer(x)
            # Only need to forward until the selected layer is reached
            if index == final_layer:
                break

        # create the histograms for weights:
        for layer in self.selected_layers:
            layer_obj = self.model[layer]
            # check if the layer has weights
            if not isinstance(layer_obj, nn.Conv2d):
                continue
            # get the weights
            weights = layer_obj.weight.data.numpy().flatten()
            # create and output the histogram
            self.create_and_save_histogram(weights, bin_num, 'layer{}_weights.png'.format(layer))

        # create histograms for features:
        for layer in self.conv_output:
            # get the output of the layer
            features = self.conv_output[layer].data.numpy().flatten()
            # create and output the histogram
            self.create_and_save_histogram(features, bin_num, 'layer{}_features.png'.format(layer))



if __name__ == '__main__':
    # the first 5 Conv2d layer of VGG16
    cnn_layers = [0, 2, 5, 7, 10]
    filter_pos = 5
    # Fully connected layer is not needed
    pretrained_model = models.vgg16(pretrained=True).features
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layers, filter_pos)
    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks(bin_num = 20)
