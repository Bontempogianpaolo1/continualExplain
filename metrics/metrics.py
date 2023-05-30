import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from .statistical_test import *


def eval_concepts_activation(concepts, n_concepts=112) -> torch.Tensor:
    return torch.sum(concepts.to(torch.float).view(-1, n_concepts), dim=0)


class class_concepts:
    def __init__(self, cl_model, dataloader, size=(5, 200, 112), device='cpu'):
        '''
        Calculation of drifting concepts for each time step. \n
        For each experience calculates for each seen class the empirical centroid distribution of concepts.\n
        Parameters:
             cl_model: type=CL_Model()
             dataloader: type=DataLoader
             size: type=Tuple (0: nr_experiences, 1: total_classes, 2: total_concepts)
        '''
        # size = #exp * #classes * #concepts
        self.model = cl_model.to(device)
        self.checkpoints = []
        self.dataloader = dataloader
        self.max_t = None
        self.c = torch.zeros(size=size, device=device)
        self.real_c = torch.zeros(size=(200,112), device=device)
        # nr experiences * classes_per_experiences
        self.seen_classes = torch.zeros(size=(4, 50), dtype=torch.long)
        self.device = torch.device(device)

    def which_classes(self, seen_classes, t):
        self.seen_classes[t] = seen_classes

    @torch.no_grad()
    def update_at_time(self, t, dir,seen_classes=None, checkpoint=None, learning_experience=0):
        if checkpoint is not None:
            self.checkpoints.append(checkpoint)
        else:
            checkpoint = self.checkpoints[t]

        if seen_classes is None:
            seen_classes = self.seen_classes[t]
        if os.path.exists(os.path.join(dir, "averages_x_class_learnt_at_"+str(learning_experience)+"_seen_at_experience_"+str(t)+".pt")):
            self.c[t, seen_classes, :]=torch.load(os.path.join(dir, 
                    "averages_x_class_learnt_at_"+str(learning_experience)+"_seen_at_experience_"+str(t)+".pt"))[t, seen_classes, :]
        else:
            self.model.load_state_dict(torch.load(checkpoint))
            self.model.eval()
            images_x_class={key:0 for key in range(200)}
            accuracy_concept=0
            tot=0
            for imgs, ys, attrs, _ in tqdm(self.dataloader):
                
                
                attrs = torch.stack(attrs, dim=1).to(
                    self.model.device, dtype=torch.long)

                for i, y in enumerate(ys):
                    if self.real_c[y].sum() == 0:
                        self.real_c[y] = attrs[i] 
                        print('Updated concepts of class', y)

                for sclass in range(112):
                    
                    mask = (ys == sclass)
                    l = len(ys[mask])

                    if (l != 0) and (sclass in self.seen_classes[t]):
                        f_imgs = imgs[mask]
                        f_attrs= attrs[mask]
                        f_imgs = f_imgs.to(self.device)
                        z=self.model.net.map_to_concepts(f_imgs)
                        #concept_discrete= torch.nn.functional.gumbel_softmax(z[:, :, 1],hard=True)*0
                        #concept_discrete= torch.nn.functional.gumbel_softmax(z[:, :, 1],hard=True)

                        pred_concepts_discretized=z
                        pred_concepts_discretized[z>0.5]=1
                        pred_concepts_discretized[z<0.5]=0
                        accuracy_concept+= (pred_concepts_discretized==f_attrs).sum()
                        tot+=f_attrs.shape[0]*f_attrs.shape[1]
                        #concepts = torch.nn.Softmax(dim=2)(z)[:, :, 1]
                        concepts = torch.nn.Sigmoid()(z)

                        for img in range(f_imgs.shape[0]):
                            images_x_class[sclass]= images_x_class[sclass]+1
                            self.c[t, sclass] += concepts[img]


            for sclass in seen_classes:
                sclass = int(sclass)
                self.c[t, sclass] /= images_x_class[sclass]

            with open('accuracies.txt', 'a') as file:
                file.write("accuracy on concepts for classes learnt at "+str(learning_experience)+"seen at"+str(t)+": "+str(accuracy_concept/tot*100))
                print("accuracy on concepts for classes learnt at ",str(learning_experience),"seen at",str(t),str(accuracy_concept/tot*100))
                file.write("\n")

            torch.save(self.c,os.path.join(dir, "averages_x_class_learnt_at_"+str(learning_experience)+"_seen_at_experience_"+str(t)+".pt"))
            print('Update mean concepts of seen_classes')

    def kl_distance(self, t0, t, classes, prob=None):
        l = len(self.c[t0, classes])
        p = self.c[t0, classes]
        if prob is None:
            q = self.c[t, classes]
        else:
            q = prob
        kl_div = 0

        #assert any(p) > 1 or any(p) < 0,  p
        #assert any(q) > 1 or any(q) < 0,  q

        concept_kl = np.zeros(l)

        for i in range(l):
            if q[i] == 0:
                q[i] += 10 ** -4
            elif q[i] == 1:
                q[i] -= 10 ** -4
            c_kl_div = p[i] * torch.log(p[i] / q[i]) + \
                      (1 - p[i]) * torch.log((1 - p[i]) / (1 - q[i]))
            concept_kl[i] = c_kl_div
            kl_div += c_kl_div
        return kl_div.cpu().numpy(), concept_kl

    def kl_distance_real(self, t0, t, classes, prob=None):
        l = len(self.c[t, classes])
        p = self.c[t, classes]
        if prob is None:
            q = self.real_c[classes]
        else:
            q = prob
        kl_div = 0

        #assert any(p) > 1 or any(p) < 0,  p
        #assert any(q) > 1 or any(q) < 0,  q

        concept_kl = np.zeros(l)

        for i in range(l):
            if q[i] == 0:
                q[i] += 10 ** -6
            elif q[i] == 1:
                q[i] -= 10 ** -6
            c_kl_div = p[i] * torch.log(p[i] / q[i]) + \
                      (1 - p[i]) * torch.log((1 - p[i]) / (1 - q[i]))
            concept_kl[i] = c_kl_div
            kl_div += c_kl_div
        return kl_div.cpu().numpy(), concept_kl

    def sample_concepts(self, t, sclass):
        self.model.load_state_dict(torch.load(self.checkpoints[t]))
        concepts = []
        for img, y, _ in self.dataloader:
            mask = (y == sclass)
            img = img.to(self.model.device)
            img = img[mask]
            l = len(y[mask])
            if l == 0:
                pass
            else:
                results = torch.nn.Softmax(dim=2)(self.model.net.map_to_concepts(img))[:, :, 1].detach().cpu().numpy()
                for result in range(results.shape[0]):
                    concepts.append( results[result])
        return np.asarray(concepts)

    def statistical_test(self, t0, t, sclass, dir, t_type='one_dimensional'):

        ### CREATE FIRST DIR

        if not os.path.exists(dir):
            os.makedirs(dir)
            print("The new directory is created:", dir)


        if not os.path.exists("concept"+str(sclass)+"t0.npy"):
            concepts_t0 = self.sample_concepts(t0, sclass)
            np.save(os.path.join(dir,"concept"+str(sclass)+"t0.npy"),concepts_t0)
        else:
            concepts_t0=np.load(os.path.join(dir,"concept"+str(sclass)+"t0.npy"),allow_pickle=True)

        if not os.path.exists("concept"+str(sclass)+"t1.npy"):
            concepts_t = self.sample_concepts(t, sclass)
            np.save(os.path.join(dir,"concept"+str(sclass)+"t1.npy"),concepts_t)
        else:
            concepts_t=np.load(os.path.join(dir,"concept"+str(sclass)+"t1.npy"),allow_pickle=True)

        if t_type == 'one_dimensional':
            p_val, p_vals, t_vals, ppfs = one_dimensional_test(
                concepts_t0, concepts_t, test_type=0)
            #print('Completed statistical test between times t0 and t1 for class'+str(sclass))
            #print("p-val: "+ str(p_val))
            #print("p_vals: "+ str(p_vals))
            #print("t_vals:"+ str(t_vals))
            #print("ppfs:"+ str(ppfs))

            if not os.path.exists(os.path.join(dir,'figures')):

                os.makedirs(os.path.join(dir,'figures'))
                print("The new directory is created:", os.path.join(dir,'figures'))

            plt.plot(range(112),t_vals)
            plt.savefig(os.path.join(dir,"figures/p_vals"+str(sclass))+".png")
            plt.close()

            return p_val, p_vals, t_vals, ppfs
        else:
            return NotImplementedError('You asked a not implemented test.')
