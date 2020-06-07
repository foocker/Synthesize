"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch

from classifier.benchmark import config as cfg

from .misc_functions import get_example_params, get_example_list
from .misc_functions import get_model_w, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
        https://github.com/pprp/SimpleCVReproduction/blob/master/pytorch-grad-cam-master/Grad_CAM_Pytorch_1_tutorialv2.ipynb
    """
    def __init__(self, model, target_layer):
        # conv1, bn1, relu, conv2, bn2
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        # print("model:",self.model, self.target_layer, type(self.target_layer))

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        # for module_pos, module in enumerate(self.model.children()):    # resnet18(1-7:9)
        #     x = module(x)  # Forward
        #     if module_pos == self.target_layer:    # layer4:index=7
        #         x.register_hook(self.save_gradient)
        #         conv_output = x  # Save the convolution output on that layer
        #     if module_pos == self.target_layer + 1:
        #         x = module(x) 
        #         x = torch.flatten(x, 1)

        for name, module in self.model.named_children():    # resnet18(1-7:9)
            if name == self.target_layer:    # 'layer4'
                for i, block in enumerate(module.children()):
                    if i == 0:    # basicblock has downsample
                        # print( x.shape, "after layer3 shape")
                        x = block(x)
                        # print(x.shape, "after layer4 basicblock")
                    else:    # basicblock has no downsample
                        identity = x
                        # x = block(x)   # in primary forward which add some other op, see below
                        for inner_block in block.children() :
                            x = inner_block(x)
                        x += identity
                        x = block.relu(x)
                        x.register_hook(self.save_gradient)
                        conv_output = x    # Save the relu output on the seconde BasicBlock in layer4
                        # print(conv_output.shape, "conv_output shape")
                        
            elif name == 'avgpool':
                x = module(x)
                x = torch.flatten(x, 1)
                # print(x.shape, name)
            
            else:
                # print(name, x.shape)
                x = module(x)  # Forward

        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        # print('gg', conv_output.shape, model_output.shape)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255

        return cam



def test_gradcam(target_layer, save_dir='cnn_visual/heatmap'):
    # default save_dir is here cnn_visual
    model_w = get_model_w()

    example_list = get_example_list(cfg.heatmap_path, cfg.heatmap_class_name, cfg.heatmap_sample_num)

    # Grad cam
    grad_cam = GradCam(model_w, target_layer=target_layer)    # resnet18
    print('have', len(example_list), 'images')
    for i in range(len(example_list)):

        (original_image, prep_img, target_class, file_name_to_export) =\
            get_example_params(example_list, i)

        # Generate cam mask
        # print(prep_img.shape, target_class, file_name_to_export)
        cam = grad_cam.generate_cam(prep_img, target_class)

        # Save mask
        save_class_activation_images(original_image, cam, file_name_to_export, dir_name=save_dir)
        print('Grad cam completed')
