import torch
import torch.nn as nn
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet50

class base(nn.Module):
    def __init__(self,classes):
        super(base, self).__init__()
        self.backbone = inception_v3(pretrained=True, aux_logits=False)
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, classes)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(self.backbone(x))))


class triLayer(nn.Module):
    def __init__(self, classes):
        super(triLayer, self).__init__()
        self.backbone = inception_v3(pretrained=True, aux_logits=False)
        self.fc1 = nn.Linear(112, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, classes)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))



class baseTransformer(nn.Module):
    def __init__(self, classes):
        super(baseTransformer, self).__init__()
        self.backbone = torch.hub.load(
            'nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True)

    def forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
            top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = self.backbone(
                x)
        return concat_logits



#TODO: create the version with FC and activation


class YandAttributes(nn.Module):
    def __init__(self,n_classes,n_attributes=312,n_hidden=1024,classify=False, device=torch.device('cuda')):
        super(YandAttributes, self).__init__()

        self.backbone = inception_v3(pretrained=True,aux_logits=False,)# device=device)
        self.fc = nn.Linear(1000, n_hidden, device=device).to(device)

        self.concepts=[]
        self.n_attributes= n_attributes
        self.classify= classify

        for attr in range(n_attributes):
            self.concepts.append(nn.Linear(n_hidden, 2, device=device))
        if classify:
            self.model2= nn.Linear(n_attributes,n_classes, bias=False, device=device)

    def forward(self, x):
        z=self.fc(self.backbone(x))
        out = []
        for fc in self.concepts:
            out.append(fc(z))
        attr_preds = torch.stack(out,dim=1)

        # TODO: introduce All feed forward architecture (easier?).
        if( self.classify):
            y = self.model2(attr_preds[:,:,1])
            return y,attr_preds
        else:
            return attr_preds


class YandConcepts(nn.Module):
    def __init__(self,n_classes,n_attributes=312,n_hidden=1024,classify=False, device=torch.device('cuda'), cbm_model='XtoCtoY'):
        super(YandConcepts, self).__init__()

        self.backbone = inception_v3(pretrained=True,aux_logits=False,)# device=device)
        self.fc = nn.Linear(1000, n_hidden, device=device)

        self.n_attributes= n_attributes
        self.classify= classify


        self.concepts=nn.Linear(n_hidden, n_attributes, device=device)
        if classify:
            self.model2= nn.Linear(n_attributes,n_classes, bias=False, device=device)

        self.cbm_model = cbm_model


    def map_to_concepts(self, imgs):
        processed = self.fc(self.backbone(imgs))
        out=self.concepts(processed)
        concept_preds = (torch.tanh(out)+1)/2

        return concept_preds

    def map_to_label(self, concepts):
        y = self.model2(concepts)
        return y

    def forward(self, x=None, concepts = None):

        if self.cbm_model == 'CtoY':
            # C-> Y predict label from true concepts
            y = self.map_to_label(concepts)
            return  y, None

        elif self.cbm_model == 'XtoCtoY':
            # X->C->Y predict concepts from preprocess
            att_preds = self.map_to_concepts(x)

            #predict label from computed concepts
            y = self.map_to_label(att_preds)
            return y,att_preds

        elif self.cbm_model == 'XtoC':
            # X-> C predict only concepts from preprocess
            att_preds = self.map_to_concepts(x)
            return None ,att_preds

        elif self.cbm_model == 'XtoY':
            # X->C->Y predict concepts from preprocess
            att_preds = self.map_to_concepts(x)

            #predict label from computed concepts
            y = self.map_to_label(att_preds)
            return  y, None


        else:
            NotImplementedError('Not the considered architectures.')



class YandAttributesTransformer(nn.Module):
    def __init__(self, n_classes, n_attributes, n_hidden=1024, classify=False):
        super(YandAttributesTransformer, self).__init__()
        self.backbone = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True,
                                       **{'topN': 6, 'device': 'cuda', 'num_classes': n_hidden})
        self.concepts = []
        self.n_attributes = n_attributes
        self.classify = classify
        for attr in range(n_attributes):
            self.concepts.append(nn.Linear(n_hidden, 2).cuda())
        if classify:
            self.model2 = nn.Linear(n_attributes, n_classes).cuda()

    def forward(self, x):
        z = self.backbone(x)
        out = []
        for fc in self.concepts:
            out.append(fc(z))
        attr_preds = torch.stack(out, dim=1)
        if(self.classify):
            y = self.model2(attr_preds[:, :, 1])
            return y, attr_preds
        else:
            return attr_preds


