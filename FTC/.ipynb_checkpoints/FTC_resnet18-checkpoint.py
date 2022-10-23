from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.dropout import Dropout
from transformers import BertConfig, BertModel
import torch
import torchvision
import torch.nn as nn
from ResNetAE.ResNetAE import ResNetAE

class Reduce(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        #x = nn.functional.normalize(x,dim=2)
        sorted_scores,_ = x.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :9, :]
        x = torch.mean(topk_scores, dim=1)
        #x = x.mean(dim=1)
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
    def __init__(self, AE, Transformer, frames_per_video, embedding_dim, rm_branch=None, num_hidden_layers=None,pretrained = False):
        super(FTC, self).__init__()
        
        last_features = 4
        self.pretrained = pretrained

        
        self.AE = AE
        if self.pretrained == True:
            from pathlib import Path
            checkpoint = torch.load(Path('/AS_clean/FTC/logs/resnet18/best_model_cont.pth'))
            prefix = 'module.model.'
            n_clip = len(prefix)
            adapted_dict = {k[n_clip:]: v for k, v in checkpoint["model"].items()
                            if k.startswith(prefix)}
            self.AE.load_state_dict(adapted_dict, strict=False)
            print('Stage1 weights Loaded res18')

        self.T  = Transformer
        self.rm_branch = rm_branch
        
        self.aorticstenosispred = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim//2, bias=True),
            nn.LayerNorm(embedding_dim//2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Linear(in_features=embedding_dim//2, out_features=4, bias=True),
            # nn.Softmax(dim=2),
            Reduce(),
            )
        
        if num_hidden_layers is not None:
            self.averaging = nn.Conv2d(in_channels=(num_hidden_layers+1), out_channels=1, kernel_size=1, bias=False)
            self.use_conv  = True
        else:
            self.use_conv = False
        
        self.vf = frames_per_video
        self.em = embedding_dim
        
    def forward(self, x):   
        
        # Video dimension (B x F x C x H x W)
        x = x.permute(0,2,1,3,4)
        
        nB, nF, nC, nH, nW = x.shape
        
        # Merge batch and frames dimension
        x = x.contiguous().view(nB*nF,nC,nH,nW)
        
        if self.pretrained:
            with torch.no_grad():
                # (BxF) x C x H x W => (BxF) x Emb
                embeddings = self.AE(x).squeeze()
                embeddings = torch.nn.functional.normalize(embeddings, dim=0)
        else:
            # (BxF) x C x H x W => (BxF) x Emb
            embeddings = self.AE(x).squeeze()
            
        
        # B x F x Emb => AttHeads+1 x B x F x Emb
        outputs = self.T(embeddings.view(-1, nF, self.em), output_hidden_states=True)
        
        # AttHeads+1 x B x F x Emb => B x F x Emb
        outputs = torch.stack(outputs.hidden_states).mean(dim=0)
        
        if not self.rm_branch == 'as':
            # B x F x Emb => B x 1 x 1
            as_prediction = self.aorticstenosispred(outputs)
        else:
            as_prediction = None
        return as_prediction

def get_model_res18(emb_dim, img_per_video, 
              num_hidden_layers = 16,
              intermediate_size = 8192,
              rm_branch = None,
              use_conv = False,
              attention_heads=16
              ):
    # https://huggingface.co/transformers/model_doc/bert.html#bertconfig
    # https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification
    # Setup model
    configuration = BertConfig(
        vocab_size=1, # Set to 0/None ?
        hidden_size=emb_dim, # Length of embeddings
        num_hidden_layers=num_hidden_layers, # 16
        num_attention_heads=attention_heads, 
        intermediate_size=intermediate_size, # 8192
        hidden_act='gelu', 
        hidden_dropout_prob=0.1, 
        attention_probs_dropout_prob=0.1, 
        max_position_embeddings=1024, # 64 ?
        type_vocab_size=1, 
        initializer_range=0.02, 
        layer_norm_eps=1e-12, 
        pad_token_id=0, 
        gradient_checkpointing=False, 
        position_embedding_type='absolute', 
        use_cache=True)
    configuration.num_labels = 4 # 4
    
    model_T  = BertModel(configuration).encoder
    

    model_res = torchvision.models.resnet18()
    dim_in = model_res.fc.in_features
    model_res.fc =  nn.Linear(dim_in, 1024)

    num_berts = num_hidden_layers if use_conv else None
        
    model = FTC(model_res, model_T, img_per_video, emb_dim, rm_branch=rm_branch, num_hidden_layers=num_berts,pretrained = True) 
    
    return model