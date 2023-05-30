#from XAI.Images.xailib.explainers.lime_explainer import LimeXAIImageExplainer
#from XAI.Images.xailib.explainers.gradcam_explainer import GradCAMImageExplainer
#from XAI.Images.xailib.explainers.intgrad_explainer import IntgradImageExplainer
#from XAI.Images.xailib.explainers.lore_explainer import LoreTabularExplainer, LoreTabularExplanation
#from XAI.Images.xailib.explainers.rise_explainer import RiseXAIImageExplainer

import torch
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
import pytorch_lightning as pl

from pytorch_visualizations.src.gradcam import GradCam, save_class_activation_images
from pytorch_visualizations.src.integrated_gradients import IntegratedGradients, save_gradient_images, convert_to_grayscale
from pytorch_visualizations.src.scorecam import ScoreCam
from pytorch_visualizations.src.smooth_grad import VanillaBackprop, generate_smooth_grad




class Label_Explainer:
    def __init__(self, strategy, best_model_path, test_loader, device='cpu', debug=False):

        if not debug:
            strategy.load_state_dict(torch.load(best_model_path))  # device)
        else:
            pass

        self.model = strategy.net
        self.model.eval()
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
        self.test_loader = test_loader
        self.deconv = transforms.Compose([
            invTrans,
            transforms.ToPILImage(mode=None),
        ])

    def explain(self, image, title, label):
        original = self.deconv(image)
        self.model = self.model.to("cpu")
        grad_cam = GradCam(self.model, "model2")
        cam = grad_cam.generate_cam(image.view(1, 3, 299, 299), label)
        save_class_activation_images(original, cam, title)
        print('Grad cam completed')
        original.save("results/original.png")


    def fit(self, method, target_layer='model2'):
        # extract images from dataset
        prep_img, labels, concepts = next(iter(self.test_loader))
        prep_img = prep_img[:10]

        # INVERT TRANSFORM
        original_image = []

        for i in range(10):
            original_image.append(self.deconv(prep_img[i]))

        for i in range(10):

            if method == 'GRADCAM':
                grad_cam = GradCam(self.model, target_layer)
                # Generate cam mask
                cam = grad_cam.generate_cam(prep_img[i].view(1, 3, 299, 299), labels[i])
                # Save mask
                save_class_activation_images(original_image[i], cam, 'Post_gradcam_image%i'%i)
                print('Grad cam completed')

            elif method == 'SCORECAM':
                score_cam = ScoreCam(self.model, target_layer=11)
                # Generate cam mask
                cam = score_cam.generate_cam(prep_img[i], labels[i])
                # Save mask
                save_class_activation_images(original_image[i], cam, 'Post_scorecam_image%i'%i)
                print('Score cam completed')

            elif method == 'INTGRAD':
                IG = IntegratedGradients(self.model)
                # Generate gradients
                integrated_grads = IG.generate_integrated_gradients(prep_img[i], labels[i], 100)
                # Convert to grayscale
                grayscale_integrated_grads = convert_to_grayscale(integrated_grads)
                # Save grayscale gradients
                save_gradient_images(grayscale_integrated_grads, 'Post_intgraf_image%i'%i + '_Integrated_G_gray')
                print('Integrated gradients completed.')

            elif method == 'SMOOTHGRAD':
                VBP = VanillaBackprop(self.model)
                # GBP = GuidedBackprop(pretrained_model)  # if you want to use GBP dont forget to
                # change the parametre in generate_smooth_grad

                param_n = 50
                param_sigma_multiplier = 4
                smooth_grad = generate_smooth_grad(VBP,  # ^This parameter
                                                   prep_img[i],
                                                   labels[i],
                                                   param_n,
                                                   param_sigma_multiplier)

                # Save colored gradients
                save_gradient_images(smooth_grad, 'Post_smoothgrad_image%i'%i + '_SmoothGrad_color')
                # Convert to grayscale
                grayscale_smooth_grad = convert_to_grayscale(smooth_grad)
                # Save grayscale gradients
                save_gradient_images(grayscale_smooth_grad, 'Post_smoothgrad_image%i'%i + '_SmoothGrad_gray')
                print('Smooth grad completed')






class Explainer:
    def __init__(self, best_model_path, test_loader):
        self.model = best_model_path
        if(type == "LIME"):
            self.explainer =LimeXAIImageExplainer()
        elif (type == "INTGRAD"):
            self.explainer = "somethingelse"
        elif (type == 'GRADCAM'):
            self.explainer = 'use xailib'
        else:
            NotImplementedError('Not supported model')

    def run(self,):
        pass


    def save_to_path(self, path):
        pass
