import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from mammoth.models.utils.continual_model import ContinualModel





class ER_CBM(ContinualModel):
    NAME = 'er_cbm'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ER_CBM, self).__init__(backbone, loss, args, transform)
        #self.buffer = Buffer(self.args.buffer_size, self.device)
        self.cbm_model = args.cbm_model
        self.perc = args.perc
        self.loss_concept_module = nn.BCELoss()

    def loss_fn(self, p_concepts, p_labels, concepts, labels, is_supervised):

        tot_loss = 0
        loss_concepts = 0
        loss_prediction = 0
        if p_concepts is not None:
            # Loss on concepts
            p_concepts = p_concepts[is_supervised == 1]
            concepts = concepts[is_supervised == 1]
            if p_concepts.shape[0] > 0:
                loss_concepts += self.loss_concept_module(p_concepts, concepts.to(
                    dtype=torch.float, device=torch.device(self.device)))
        if p_labels is not None:
            # Loss on Labels
            loss_prediction += self.loss(p_labels, labels)
        tot_loss = loss_prediction + loss_concepts
        return tot_loss, loss_prediction, loss_concepts

    def observe(self, inputs, concepts, labels, is_supervised):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        # if not self.buffer.is_empty():
        # buf_inputs, buf_labels = self.buffer.get_data(
        #     self.args.minibatch_size, transform=self.transform)
        # inputs = torch.cat((inputs, buf_inputs))
        # labels = torch.cat((labels, buf_labels))

        p_labels, p_concepts = self.net(inputs)

        tot_loss, loss_pred, loss_concept = self.loss_fn(
            p_concepts, p_labels, concepts, labels, is_supervised)
        tot_loss.backward()

        self.opt.step()
        pred_concepts_discretized = p_concepts
        pred_concepts_discretized[p_concepts > 0.5] = 1
        pred_concepts_discretized[p_concepts < 0.5] = 0
        acc_concepts = torch.sum(
            pred_concepts_discretized == concepts)/(inputs.shape[0]*112)
        y_pred = torch.argmax(p_labels, dim=1)
        acc_labels = torch.sum(y_pred == labels)/(inputs.shape[0])
        # self.buffer.add_data(examples=not_aug_inputs,
        #                    labels=labels[:real_batch_size])
        try:
            print("tot loss: " + str(tot_loss.item())+" prediction loss: " +
                  str(loss_pred.item()) + " concept loss " + str(loss_concept.item()))
        except:
            print("tot loss: " + str(tot_loss.item())+" prediction loss: " +
                  str(loss_pred.item()))
        print("accuracy concept on train " + str(acc_concepts))
        print("accuracy on label " + str(acc_labels))

        print("-"*30)

        return tot_loss.item()



class ER_DeepProbLog(ContinualModel):
    NAME = 'er_cbm'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ER_DeepProbLog, self).__init__(backbone, loss, args, transform)
        #self.buffer = Buffer(self.args.buffer_size, self.device)


    def observe(self, inputs, labels, is_supervised=False):

        self.opt.zero_grad()
        # if not self.buffer.is_empty():
        # buf_inputs, buf_labels = self.buffer.get_data(
        #     self.args.minibatch_size, transform=self.transform)
        # inputs = torch.cat((inputs, buf_inputs))
        # labels = torch.cat((labels, buf_labels))

        mu, add_prob= self.net(inputs)

        loss, _, _, query_cross_entropy, label_cross_entropy = det_loss_function(x, mu, add_prob, model=self.net, 
                                                                                labels=labels, query=True, sup=is_supervised)
        
        loss.backward()

        self.opt.step()
       
        #print("accuracy concept on train " + str(acc_concepts))
        #print("accuracy on label " + str(acc_labels))

        print("-"*30)

        return loss.item()





def det_loss_function(x, mu, add_prob, model, labels=None, query=True, recon_w=1, kl_w=1, query_w=1,
                  sup_w=1, sup=False, rec_loss='MSE'):
   
    # Cross Entropy on the query
    if query:
        target = torch.ones_like(add_prob)
        query_cross_entropy = torch.nn.BCELoss(reduction='mean')(torch.flatten(add_prob), torch.flatten(target))
    else:
        query_cross_entropy = torch.zeros(size=())

    # Cross Entropy digits supervision
    if sup:
        idxs = labels[labels[:, -1] >= -1][:, -1]  # Index(es) of the labelled image in the current batch
        digit1, digit2 = labels[idxs[0]][:2]  # Correct digit in position 1 and 2, each batch has the same images

        pred_digit1 = model.facts_probs[idxs, 0, digit1]
        pred_digit2 = model.facts_probs[idxs, 1, digit2]
        pred = torch.cat([pred_digit1, pred_digit2])
        target = torch.ones_like(pred, dtype=torch.float32)
        label_cross_entropy = torch.nn.BCELoss(reduction='mean')(torch.flatten(pred), torch.flatten(target))
    else: 
        label_cross_entropy = np.zeros(size=())
  # VAVE Losses
    recon_loss = torch.tensor(0)
    gauss_kl_div = torch.tensor(0)

    # Total loss
    loss = recon_w * recon_loss + kl_w * gauss_kl_div + query_w * query_cross_entropy + sup_w * label_cross_entropy

    return loss, recon_loss, gauss_kl_div, query_cross_entropy, label_cross_entropy
