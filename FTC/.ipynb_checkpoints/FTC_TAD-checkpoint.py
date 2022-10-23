from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.dropout import Dropout
from transformers import BertConfig, BertModel
import torch
import torch.nn as nn
import torchvision
from ResNetAE.ResNetAE import ResNetAE
from FTC.posembedding import build_position_encoding
from FTC.deformable_transformer import DeformableTransformer
from FTC.util.misc import NestedTensor


class Reduce(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        
        # sorted_scores,_ = x.sort(descending=True, dim=1)
        # topk_scores = sorted_scores[:, :9, :]
        # x = torch.mean(topk_scores, dim=1)
        x = x.mean(dim=1)
        return x

class Inception(nn.Module):
    def __init__(self, emb_dim, in_channels=1, out_channels_per_conv=16):
        super().__init__()
        
        self.five       = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_conv, kernel_size=(5,emb_dim), stride=(1,1), padding=(2,0), bias=False)
        self.three      = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_conv, kernel_size=(3,emb_dim), stride=(1,1), padding=(1,0), bias=False)
        self.stride2    = nn.Conv2d(in_channels=in_channels, out_channels=out_channels_per_conv, kernel_size=(3,emb_dim), stride=(1,2), padding=(1,0), bias=False)
    
    def forward(self,x):
        out_five    = self.five(x)
        out_three   = self.three(x)
        out_stride2 = self.stride2(x)
        
        output = torch.cat((out_five, out_three, out_stride2),dim=1).permute(0,3,2,1) # Cat on channel dim
        return output
    

        
# New Multi Branch Auto Encoding Transformer
class FTC(nn.Module):
    def __init__(self, AE, pos_embed , transformer, embedding_dim, pretrained = False):
        super(FTC, self).__init__()
        
        last_features = 4
        self.pretrained = pretrained
        
        
        self.AE = AE
        if self.pretrained == True:
            from pathlib import Path
            checkpoint = torch.load(Path('/AS_Neda/FTC/logs/resnet18/best_model_cont.pth'))
            prefix = 'module.model.'
            n_clip = len(prefix)
            adapted_dict = {k[n_clip:]: v for k, v in checkpoint["model"].items()
                            if k.startswith(prefix)}
            self.AE.load_state_dict(adapted_dict, strict=False)
            print('Stage1 weights Loaded res18')

        self.pos_embed = pos_embed
        self.transformer = transformer
        
        self.aorticstenosispred = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
            nn.LayerNorm(embedding_dim//2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=embedding_dim//2, out_features=4, bias=True),
            # nn.Softmax(dim=2),
            # Reduce(),
            )
        
        self.attentionweights = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
            nn.LayerNorm(embedding_dim//2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=embedding_dim//2, out_features=1, bias=True),
            # nn.Softmax(dim=2),
            # Reduce(),
            )

        #self.vf = frames_per_video
        self.em = embedding_dim
        
    def forward(self, x):   
        
        # Video dimension (B x F x C x H x W)
        x = x.permute(0,2,1,3,4)
        
        nB, nF, nC, nH, nW = x.shape
        # Merge batch and frames dimension
        x = x.contiguous().view(nB*nF,nC,nH,nW)
        
        if self.pretrained:
            # with torch.no_grad():
            # (BxF) x C x H x W => (BxF) x Emb
            embeddings = self.AE(x).squeeze()
            #embeddings = torch.nn.functional.normalize(embeddings, dim=0)
        else:
            # (BxF) x C x H x W => (BxF) x Emb
            embeddings = self.AE(x).squeeze()
            
            
        embeddings_reshaped = embeddings.view(nB, nF, self.em )
        embeddings_reshaped = embeddings_reshaped.permute(0,2,1)
            
        # Creatining porsitional embedding and nested tensor
        mask = torch.ones((nB, nF), dtype=torch.bool).cuda()
        #embeddings_reshaped = embeddings_reshaped.cuda()
        samples = NestedTensor(embeddings_reshaped, mask)
        pos = self.pos_embed(samples).cuda()
        
        #  B x F x Emb

        outputs = self.transformer([embeddings_reshaped],[pos],[mask])
        
        method = "attention"   # "average","emb_mag","attention",
        if method == "average":
            # B x F x Emb => B x T x 4
            as_prediction = self.aorticstenosispred(outputs)
            # B x T x 4   =>  B x 4
            as_prediction = as_prediction.mean(dim=1)
            
        elif method == "emb_mag":
            # B x F x Emb => B x T x 4
            as_prediction = self.aorticstenosispred(outputs)
            # B x T x 4   =>  B x K x 4
            idx_act_feat = max_index[:,:9].unsqueeze(2).expand([-1, -1, 4])
            as_prediction = as_prediction.gather(1,idx_act_feat)
            as_prediction = as_prediction.mean(dim=1)
            
        elif method == "attention":
            # B x F x Emb => B x T x 4
            as_prediction = self.aorticstenosispred(outputs)
            
            # attention weights B x F x 1
            att_weight = self.attentionweights(outputs)
            #print(att_weight.shape,outputs.shape)
            att_weight = nn.functional.softmax(att_weight, dim=1)
            # B x T x 4   =>  B x 4
            as_prediction = (as_prediction * att_weight).sum(1)
            # Calculating the entropy for attention
            entropy_attention = torch.sum(-att_weight*torch.log(att_weight), dim=1)
            
        elif method == "attention_resbranch":
            # B x F x Emb => B x T x 4
            as_prediction = self.aorticstenosispred(outputs)
            
            # attention weights B x F x 1
            att_weight = self.attentionweights(embeddings_reshaped.permute(0,2,1))
            att_weight = nn.functional.softmax(att_weight, dim=1)
            # B x T x 4   =>  B x 4
            as_prediction = (as_prediction * att_weight).sum(1)
            
            # Calculating the entropy for attention
            entropy_attention = torch.sum(-att_weight*torch.log(att_weight), dim=1)
        

        return as_prediction,entropy_attention,outputs,att_weight

def get_model_tad(emb_dim, img_per_video, 
              num_hidden_layers = 16,
              intermediate_size = 8192,
              rm_branch = None,
              use_conv = False,
              attention_heads=16
              ):
    
    model_res = torchvision.models.resnet18()
    dim_in = model_res.fc.in_features
    model_res.fc =  nn.Linear(dim_in, 1024)
    
    # Setup model
    pos_embed = build_position_encoding(1024)
    transformer = DeformableTransformer(
            d_model=1024,
            nhead=8,
            num_encoder_layers=2,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            return_intermediate_dec=True,
            num_feature_levels=1,
            dec_n_points=4,
            enc_n_points=4)
        
    model = FTC(model_res,pos_embed,transformer, emb_dim,pretrained = False) 
    
    return model