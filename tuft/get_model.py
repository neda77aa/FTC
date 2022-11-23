import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.wrn import build_WideResNet
import torchvision.models as models
from models.FTC_resnet18 import get_model_res18
from models.FTC_image import get_model_image
import torch
import torchvision 
#from pytorchvideo.models import x3d, resnet, slowfast
    

class wrn(nn.Module):
    def __init__(self, num_classes_diagnosis, num_classes_view):
        super(wrn, self).__init__()
        self.model = models.wide_resnet50_2(pretrained=True)
        # new layer
        self.relu = nn.ReLU(inplace=True)
        self.fc_view = nn.Linear(1000, num_classes_view)
        self.fc_diagnosis = nn.Linear(1000, num_classes_diagnosis)
        self.attention = nn.Linear(1000, 1)

    def forward(self, x):
        out = self.relu(self.model(x))
        pred_view = self.fc_view(out)
        pred_as = self.fc_diagnosis(out)
        att = self.attention(out)
        return pred_view , pred_as
    
    
class wrn_video(nn.Module):
    def __init__(self, num_classes_diagnosis, num_classes_view):
        super(wrn_video, self).__init__()
        self.model = models.wide_resnet50_2(pretrained=True)
        # new layer
        self.relu = nn.ReLU(inplace=True)
        self.fc_view = nn.Linear(1000, num_classes_view)
        self.fc_diagnosis = nn.Linear(1000, num_classes_diagnosis)
        self.attention = nn.Linear(1000, 1)

    def forward(self, x):
        
        nB, nF, nC, nH, nW = x.shape
        
        # Merge batch and frames dimension
        x = x.contiguous().view(nB*nF,nC,nH,nW)
        
        out = self.relu(self.model(x))
        out = out.view(-1, nF, 1000)
        pred_view = self.fc_view(out)
        pred_as = self.fc_diagnosis(out)
        att = self.attention(out)
        
        # View Analysis
        view_prob = torch.nn.functional.softmax(pred_view,dim=2)
        relevance = torch.cumsum(view_prob,dim=2)[:,:,1]
        #print(relevance)
        relevance  = relevance.unsqueeze(2).repeat(1,1,3)
        
        # attention weights B x F x 1
        att_weight = nn.functional.softmax(att, dim=1)
        # B x T x 4   =>  B x 4
        as_prediction = ((relevance*pred_as) * att_weight).sum(1)
            
        return pred_view , as_prediction

        
def get_model(config):
    nc_diagnosis = config['num_classes_diagnosis'] = 3
    nc_view = config['num_classes_view'] = 3
    if config['model'] == "wideresnet":
        # wrn_builder = build_WideResNet()
        # model = wrn_builder.build(nc_diagnosis,nc_view)
        model = wrn(nc_diagnosis,nc_view)
        
                
    if config['model'] == "FTC_res18":
        model = wrn_video(nc_diagnosis,nc_view)
        # Read checkpoint file.
        load_res = torch.load('/AS_clean/tuft_fs/logs/wrn_att/best_model.pth')
        # Loading model.
        model.load_state_dict(load_res["model"])
        print('model loaded')

        # latent_dim=1024
        # ds_max_length = 128
        # num_hidden_layers = 2      # Number of Transformers
        # intermediate_size = 8192    # size of the main MLP inside of the Transformers
        # rm_branch = None            # select branch to not train: None, 'SD', 'EF'
        # use_conv = False            # use convolutions instead of MLP for the regressors - worse results
        # attention_heads = 16    
        # model = get_model_res18(latent_dim, img_per_video=ds_max_length, 
        #                     num_hidden_layers=num_hidden_layers, 
        #                     intermediate_size=intermediate_size, 
        #                     rm_branch=rm_branch, use_conv=use_conv,
        #                     attention_heads=attention_heads)
        
        
    if config['model'] == "FTC_image":
        model = res18(nc_diagnosis,nc_view)
        # latent_dim=1024
        # ds_max_length = 128
        # num_hidden_layers = 2      # Number of Transformers
        # intermediate_size = 8192    # size of the main MLP inside of the Transformers
        # rm_branch = None            # select branch to not train: None, 'SD', 'EF'
        # use_conv = False            # use convolutions instead of MLP for the regressors - worse results
        # attention_heads = 16    
        # model = get_model_image(latent_dim, img_per_video=ds_max_length, 
        #                     num_hidden_layers=num_hidden_layers, 
        #                     intermediate_size=intermediate_size, 
        #                     rm_branch=rm_branch, use_conv=use_conv,
        #                     attention_heads=attention_heads)
    # Calculate number of parameters. 
    num_parameters = sum([x.nelement() for x in model.parameters()])
    print(f"The number of parameters in {config['model']}: {num_parameters/1000:9.2f}k")

    return model

