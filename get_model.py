from torchvision.models.video import r2plus1d_18
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from FTC.FTC import get_model_b
from FTC.FTC_resnet18 import get_model_res18
from FTC.FTC_TAD import get_model_tad
from ResNetAE.ResNetAE import ResNetAE
import torch
#from pytorchvideo.models import x3d, resnet, slowfast

class Two_head(nn.Module):
    def __init__(self, model, num_classes1, num_classes2):
        super(Two_head, self).__init__()
        self.model = model
        self.nc1 = num_classes1
        self.nc2 = num_classes2

    def forward(self, x):
        x = self.model(x)
        out1 = x[:,:self.nc1]
        out2 = x[:,self.nc1:self.nc1+self.nc2]
        return out1, out2
    
class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='r2plus1d_18', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        self.model = r2plus1d_18(pretrained=False, num_classes=2)
        dim_in = self.model.fc.in_features
        self.linear = (head=='linear')
        if head == 'linear':
             self.model.fc = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'mlp':
            self.model.fc = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.model(x)
        feat = F.normalize(feat, dim=1)
        return feat
    
    
class Linear(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='r2plus1d_18', nc=4):
        super(Linear, self).__init__()
        self.model = r2plus1d_18(pretrained=False, num_classes=nc)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc.weight.requires_grad = True
        self.model.fc.bias.requires_grad = True
    def forward(self, x):
        feat = self.model(x)
        return feat

class Reduce(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        #x = nn.functional.normalize(x,dim=2)
        x = x.mean(dim=1)
        return x
    
    
class ResAEend(nn.Module):
    """backbone + projection head"""
    def __init__(self, model, nc=4):
        super(ResAEend, self).__init__()
        self.model = model
        embedding_dim = 1024
        self.aorticstenosispred = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
            nn.LayerNorm(embedding_dim//2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=embedding_dim//2, out_features=4, bias=True),
            # nn.Softmax(dim=2),
            Reduce(),
            )
    def forward(self, x):
        # Video dimension (B x F x C x H x W)
        x = x.permute(0,2,1,3,4)
        
        nB, nF, nC, nH, nW = x.shape
        
        # Merge batch and frames dimension
        x = x.contiguous().view(nB*nF,nC,nH,nW)
        
        with torch.no_grad():
            # (BxF) x C x H x W => (BxF) x Emb
            embeddings = self.model.encode(x).squeeze()
            
        # B x F x Emb => B x 1 x 1
        as_prediction = self.aorticstenosispred(embeddings.view(nB,nF,1024))
        #print(as_prediction.shape)

        return as_prediction
    
    
class Res18end(nn.Module):
    """backbone + projection head"""
    def __init__(self, model, nc=4):
        super(Res18end, self).__init__()
        self.model = model
        embedding_dim = 1024
        self.aorticstenosispred = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
            nn.LayerNorm(embedding_dim//2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=embedding_dim//2, out_features=4, bias=True),
            # nn.Softmax(dim=2),
            Reduce(),
            )
    def forward(self, x):
        # Video dimension (B x F x C x H x W)
        x = x.permute(0,2,1,3,4)
        
        nB, nF, nC, nH, nW = x.shape
        
        # Merge batch and frames dimension
        x = x.contiguous().view(nB*nF,nC,nH,nW)
        
        with torch.no_grad():
            # (BxF) x C x H x W => (BxF) x Emb
            embeddings = self.model(x).squeeze()
            embeddings = F.normalize(embeddings, dim=0)
            
        # B x F x Emb => B x 1 x 1
        as_prediction = self.aorticstenosispred(embeddings.view(nB,nF,1024))
        #print(as_prediction.shape)

        return as_prediction
    

    
    
    
    
    
def get_model(config):
    if config['loss_type'] == 'laplace_cdf':
        nc = 2
    else:
        nc = config['num_classes']
    if config['cotrastive_method'] == "CE":
        if config['model'] == "r2plus1d_18":
            # instantiate the pretrained model
            model = r2plus1d_18(pretrained=config['pretrained'], num_classes=nc)
            #model = Two_head(model, nc, 2)
        if config['model'] == "FTC":
            latent_dim=1024
            ds_max_length = 128
            num_hidden_layers = 16      # Number of Transformers
            intermediate_size = 8192    # size of the main MLP inside of the Transformers
            rm_branch = None            # select branch to not train: None, 'SD', 'EF'
            use_conv = False            # use convolutions instead of MLP for the regressors - worse results
            attention_heads = 16    
            model = get_model_b(latent_dim, img_per_video=ds_max_length, 
                                num_hidden_layers=num_hidden_layers, 
                                intermediate_size=intermediate_size, 
                                rm_branch=rm_branch, use_conv=use_conv,
                                attention_heads=attention_heads)
            
            
        if config['model'] == "FTC_18":
            latent_dim=1024
            ds_max_length = 128
            num_hidden_layers = 16      # Number of Transformers
            intermediate_size = 8192    # size of the main MLP inside of the Transformers
            rm_branch = None            # select branch to not train: None, 'SD', 'EF'
            use_conv = False            # use convolutions instead of MLP for the regressors - worse results
            attention_heads = 16    
            model = get_model_res18(latent_dim, img_per_video=ds_max_length, 
                                num_hidden_layers=num_hidden_layers, 
                                intermediate_size=intermediate_size, 
                                rm_branch=rm_branch, use_conv=use_conv,
                                attention_heads=attention_heads)
            
            
        if config['model'] == "FTC_TAD":
            latent_dim=1024
            ds_max_length = 128
            num_hidden_layers = 16      # Number of Transformers
            intermediate_size = 8192    # size of the main MLP inside of the Transformers
            rm_branch = None            # select branch to not train: None, 'SD', 'EF'
            use_conv = False            # use convolutions instead of MLP for the regressors - worse results
            attention_heads = 16    
            model = get_model_tad(latent_dim, img_per_video=ds_max_length, 
                                num_hidden_layers=num_hidden_layers, 
                                intermediate_size=intermediate_size, 
                                rm_branch=rm_branch, use_conv=use_conv,
                                attention_heads=attention_heads)
            
        if config['model'] == "resae":
            
            model_AE = ResNetAE(input_shape=(224, 224, 3), n_ResidualBlock=8, n_levels=4, bottleneck_dim=1024)
            model_AE.decoder = None
            model_AE.fc2 = None
            from pathlib import Path
            checkpoint = torch.load(Path('/AS_clean/FTC/logs/resnetae/best_model_cont.pth'))
            prefix = 'module.model.'
            n_clip = len(prefix)
            adapted_dict = {k[n_clip:]: v for k, v in checkpoint["model"].items()
                            if k.startswith(prefix)}
            model_AE.load_state_dict(adapted_dict, strict=False)
            print('Stage1 weights Loaded res')
            model = ResAEend(model_AE)
            
        if config['model'] == "res18":
            import torchvision
            import torch
            model = torchvision.models.resnet18()
            dim_in = model.fc.in_features
            model.fc =  nn.Linear(dim_in, 1024)
            from pathlib import Path
            checkpoint = torch.load(Path('/AS_clean/FTC/logs/resnet18/best_model_cont.pth'))
            prefix = 'module.model.'
            n_clip = len(prefix)
            adapted_dict = {k[n_clip:]: v for k, v in checkpoint["model"].items()
                            if k.startswith(prefix)}
            model.load_state_dict(adapted_dict, strict=False)
            print('Stage1 weights Loaded res18')
            model = Res18end(model)
            
        if config['model'] == "res18_linear":
            import torchvision
            import torch
            model = torchvision.models.resnet18()
            dim_in = model.fc.in_features
            model.fc =  nn.Linear(dim_in, 4)
            from pathlib import Path
            checkpoint = torch.load(Path('/AS_clean/FTC/logs/resnet18/best_model.pth'))
            prefix = 'module.model.'
            n_clip = len(prefix)
            adapted_dict = {k[n_clip:]: v for k, v in checkpoint["model"].items()
                            if k.startswith(prefix)}
            model.load_state_dict(adapted_dict, strict=False)
            print('Stage1 weights Loaded res18')

        # Calculate number of parameters. 
        num_parameters = sum([x.nelement() for x in model.parameters()])
        print(f"The number of parameters in {config['model']}: {num_parameters/1000:9.2f}k")

    elif config['cotrastive_method'] == "SupCon" or config['cotrastive_method'] == "SimCLR":
        model = SupConResNet(head = 'mlp',feat_dim = config['feature_dim'])
        # Calculate number of parameters. 
        num_parameters = sum([x.nelement() for x in model.parameters()])
        print(f"The number of parameters in {config['model']}: {num_parameters/1000:9.2f}k")

    elif config['cotrastive_method'] == "Linear":
        model = Linear(nc = 4)

    return model

# # test script
# config = {'model' : 'slowfast', 'num_classes' : 4, 
#           'pretrained':False, 'loss_type':'evidential'}
# model = get_model(config)
# model.cuda()
# print(model)#summary(model, (3,16,224,224))

# import torch # testing 2 heads
# t = torch.ones((5,3,16,224,224))
# a,b = model(t.cuda())