import numpy as np
import cv2
import torch
from sklearn.manifold import TSNE
import seaborn as sb
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib.patheffects as pe
import sys 
sys.path.append('/AS_Neda/AS_thesis/')
from get_config import get_config
from dataloader.as_dataloader_revision import get_as_dataloader
from get_model import get_model
import os


class TransformerFeatureMap:
    def __init__(self, model, layer_name='avgpool'):
        self.model = model
        for name, module in self.model.named_modules():
            if layer_name in name:
                module.register_forward_hook(self.get_feature)
        self.feature = []

    def get_feature(self, module, input, output):
        self.feature.append(output.cpu())

    def __call__(self, input_tensor):
        self.feature = []
        with torch.no_grad():
            output = self.model(input_tensor.cuda())

        return self.feature


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    config = get_config()
    
    model = get_model(config)
    if True:
        pt_file =Path('/AS_Neda/AS_thesis/logs/150_contrastive_best/best_model_cont.pth')
        print(f"restoring {pt_file}")

        # Read checkpoint file.
        load_res = torch.load(pt_file)
        # Loading model.
        model.load_state_dict(load_res["model"])
    
    config['cotrastive_method'] = 'CE'
    dataloader = get_as_dataloader(config, split='train', mode='train')

    (inputs, labels_AS ,label_B) = next(iter(dataloader))
    
    # feature_extractor = TransformerFeatureMap(model.cuda())
    # feature = feature_extractor(inputs)
    feature = model(inputs)
    features = feature.detach().numpy()
    labels_tot = np.array(labels_AS)
    for cur_iter, (inputs, labels_AS,label_B) in enumerate(dataloader):
        feature = model(inputs)
        features = np.concatenate((features,feature.detach().numpy()))
        labels_tot = np.concatenate((labels_tot,np.array(labels_AS)))
        
    print(sum(labels_tot==0),sum(labels_tot==1),sum(labels_tot==2),sum(labels_tot==3))
    print('Done importing features',len(labels_tot))
    tsne = TSNE(n_components=2).fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)


    # make a mapping from category to your favourite colors and labels
    category_to_label = {0: 'Normal', 1:'Mild', 2:'Moderate', 3:'Severe'}
    
    # setup the plot
    fig, ax = plt.subplots(1,1, figsize=(6,6))

    # We choose a color palette with seaborn.
    palette = np.array(sb.color_palette("hls", 4))
    #print(palette[labels_tot.astype(np.int)])
    # make the scatter
    scat = ax.scatter(tx,ty,c=labels_tot.astype(np.int), cmap=plt.cm.get_cmap("jet", 4), marker='.')
    # create the colorbar
    fig.colorbar(scat,ticks=range(4))
    # cb.set_label('Custom cbar')
    ax.set_title('T-SNE')
    txts = []
    for i in range(4):
        # Position of each label.
        xtext, ytext = np.median(tx[labels_tot == i]),np.median(ty[labels_tot == i])
        txt = ax.text(xtext, ytext, category_to_label.get(i), fontsize=10)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    
    plt.show()
    plt.savefig('scatter_train.png')
        


if __name__ == "__main__":
    main()
