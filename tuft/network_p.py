import os
import torch
import wandb
import torch.nn.functional as F
import numpy as np
import pandas as pd

from tqdm import tqdm
#from torchsummary import summary

from losses import laplace_cdf_loss, laplace_cdf
from losses import evidential_loss, evidential_prob_vacuity
from losses import SupConLoss
from visualization.vis import plot_tsne_visualization
import dataloader.utils as utils
from utils import validation_constructive
from pathlib import Path




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
    
    

class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, model, config):
        """Initialize configuration."""
        self.config = config
        self.model = None
        self.model = model
        self.encoder = TransformerFeatureMap(self.model.cuda())
        if self.config['use_cuda']:  
            print(torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        self.num_classes_AS = config['num_classes_diagnosis']
        self.num_classes_view = config['num_classes_view']
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                            self.optimizer,T_max=config['num_epochs'])
        self.loss_type = config['loss_type']
        #self.bicuspid_weight = config['bicuspid_weight']
        # Recording the training losses and validation performance.
        self.train_losses = []
        self.valid_oas = []
        self.idx_steps = []
        self.lamda = 0.3

        # init auxiliary stuff such as log_func
        self._init_aux()

    def _init_aux(self):
        """Intialize aux functions, features."""
        # Define func for logging.
        self.log_func = print

        # Define directory wghere we save states such as trained model.
        if self.config['use_wandb']:
            if self.config['mode'] == 'test':
                # read from the pre-specified test folder
                if self.config['model_load_dir'] is None:
                    raise AttributeError('For test-only mode, please specify the model state_dict folder')
                self.log_dir = os.path.join(self.config['log_dir'], self.config['model_load_dir'])
            else:
                # create a new directory
                self.log_dir = os.path.join(self.config['log_dir'], wandb.run.name)
        else:
            self.log_dir = self.config['log_dir']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # File that we save trained model into.
        self.checkpts_file = os.path.join(self.log_dir, "checkpoint.pth")

        # We save model that achieves the best performance: early stopping strategy.
        self.bestmodel_file = os.path.join(self.log_dir, "best_model.pth")
        self.bestmodel_file_load = Path('/AS_clean/tuft_fs/logs/wrn-50-2/best_model.pth')
        
        # For test, we also save the results dataframe
        self.test_results_file = os.path.join(self.log_dir, "best_model.pth")
 
    
    def _save(self, pt_file):
        """Saving trained model."""

        # Save the trained model.
        torch.save(
            {
                #"model": self.model.module.state_dict(),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            pt_file,
        )

    def _restore(self, pt_file):
        """Restoring trained model."""
        print(f"restoring {pt_file}")

        # Read checkpoint file.
        load_res = torch.load(pt_file)
        # Loading model.
        self.model.load_state_dict(load_res["model"])
        # Loading optimizer.
        self.optimizer.load_state_dict(load_res["optimizer"])
        
        
    
    def _get_loss(self, logits, target, nclasses):
        """ Compute loss function based on configs """
        if self.loss_type == 'cross_entropy':
            # BxC logits into Bx1 ground truth
            loss = F.cross_entropy(logits, target)
        elif self.loss_type == 'evidential':
            # alpha = F.softplus(logits)+1
            # target_oh = F.one_hot(target, nclasses)
            # mse, kl = evidential_mse(alpha, target_oh, alpha.device)
            # loss = torch.mean(mse + 0.1*kl)
            loss = evidential_loss(logits, target, nclasses)
        elif self.loss_type == 'laplace_cdf':
            # logits_categorical = laplace_cdf(F.sigmoid(logits), nclasses, logits.device)
            # target_oh = 0.9*F.one_hot(target, nclasses) + 0.1/nclasses
            # loss = F.binary_cross_entropy(logits_categorical, target_oh)
            loss = laplace_cdf_loss(logits, target, nclasses)
        elif self.loss_type == 'SupCon' or self.loss_type == 'SimCLR':
            criterion = SupConLoss(temperature=self.temperature)
            if torch.cuda.is_available():
                criterion = criterion.cuda()
            if self.contrastive_method == 'SupCon':
                loss = criterion(logits, target)
            elif self.contrastive_method == 'SimCLR':
                loss = criterion(logits)
        else:
            raise NotImplementedError
        return loss
    
    # obtain summary statistics of
    # argmax, max_percentage, entropy, evid.uncertainty for each function
    # expects logits BxC for classification, Bx2 for cdf
    def _get_prediction_stats(self, logits, nclasses):
        # convert logits to probabilities
        if self.loss_type == 'cross_entropy':
            prob = F.softmax(logits, dim=1)
            vacuity = -1
        elif self.loss_type == 'evidential':
            prob, vacuity = evidential_prob_vacuity(logits, nclasses)
            #vacuity = vacuity.squeeze()
        elif self.loss_type == 'laplace_cdf':
            prob = laplace_cdf(F.sigmoid(logits), nclasses, logits.device)
            vacuity = -1
        else:
            raise NotImplementedError
        max_percentage, argm = torch.max(prob, dim=1)
        entropy = torch.sum(-prob*torch.log(prob), dim=1)
        uni = utils.test_unimodality(prob.cpu().numpy())
        return argm, max_percentage, prob ,entropy, vacuity, uni
        
    # def get_lr(self):
    #     return self.scheduler.get_lr()
        
    def train(self, loader_tr, loader_va,loader_te):
        """Training pipeline."""
        # Switch model into train mode.
        self.model.train()
        best_va_acc = 0.0 # Record the best validation metrics.
        best_va_acc_supcon = 0.0
        best_cont_loss = 1000
        #best_B_f1 = 0.0
            
        for epoch in range(self.config['num_epochs']):
            #losses_AS = []
            #losses_B = []
            losses = []
            print('Epoch: ' + str(epoch) + ' LR: ' + str(self.scheduler.get_lr()))
            
            for loader_tr_min in loader_tr: 
                with tqdm(total=len(loader_tr_min)) as pbar:
                    for data in loader_tr_min:
                        cine = data['x']
                        target_view = data['y_view']
                        target_AS = data['y_AS']

                        # Cross Entropy Training

                        # Transfer data from CPU to GPU.
                        if self.config['use_cuda']:
                            cine = cine.cuda()
                            target_AS = target_AS.cuda()
                            target_view = target_view.cuda()
                        pred_view ,pred_AS = self.model(cine) # Bx3xTxHxW
                        loss_AS = self._get_loss(pred_AS, target_AS, self.num_classes_AS)
                        #print(target_view.shape,pred_view.shape)
                        loss_view = self._get_loss(pred_view.view(pred_view.shape[0]*pred_view.shape[1],3), 
                                                   target_view.view(target_view.shape[0]*target_view.shape[1])
                                                   ,self.num_classes_view)
                        loss = loss_AS + self.lamda * loss_view
                        losses += [loss]
                        #class_weights="0.2447,0.7238,0.0316"

                        # Calculate the gradient.
                        loss.backward()
                        # Update the parameters according to the gradient.
                        self.optimizer.step()
                        # Zero the parameter gradients in the optimizer
                        self.optimizer.zero_grad() 
                        pbar.set_postfix_str("loss={:.4f}".format(loss.item()))
                        pbar.update()

            loss_avg = torch.mean(torch.stack(losses)).item()
            
            
            acc_AS, acc_view ,val_loss = self.test(loader_va, mode="val")
            acc_AS_test, acc_view_test ,test_loss = self.test(loader_te, mode="val")
            print('acc:',acc_AS,'acc_view:',acc_view,'acc_test:',acc_AS_test,'acc_view_test:',acc_view_test)
            if self.config['use_wandb']:
                wandb.log({"tr_loss":loss_avg, "val_loss":val_loss, "val_AS_acc":acc_AS , "val_view_acc":acc_view})

            # Save model every epoch.
            #self._save(self.checkpts_file)

            # Early stopping strategy.
            if acc_AS > best_va_acc:
                # Save model with the best accuracy on validation set.
                best_va_acc = acc_AS
                #best_B_f1 = f1_B
                self._save(self.bestmodel_file)

            print(
                "Epoch: %3d, loss: %.5f, val loss: %.5f, acc: %.5f, top AS acc: %.5f"
                % (epoch, loss_avg, val_loss, acc_AS, best_va_acc)
            ) 

            # Recording training losses and validation performance.
            self.train_losses += [loss_avg]
            self.valid_oas += [acc_AS]
            self.idx_steps += [epoch]
            
            # modify the learning rate
            self.scheduler.step()   

    @torch.no_grad()
    def test(self, loader_te, mode="test"):
        """Estimating the performance of model on the given dataset."""
        # Choose which model to evaluate.
        if mode=="test":
            self._restore(self.bestmodel_file)
        # Switch the model into eval mode.
        self.model.eval()

        conf_AS = np.zeros((self.num_classes_AS, self.num_classes_AS))
        conf_view = np.zeros((self.num_classes_view, self.num_classes_view))
        losses = []
        
        for data in tqdm(loader_te):
            cine = data['x']
            target_view = data['y_view']
            target_AS = data['y_AS']
            # Transfer data from CPU to GPU.
            # Cross Entropy Training

            # Transfer data from CPU to GPU.
            if self.config['use_cuda']:
                cine = cine.cuda()
                target_AS = target_AS.cuda()
                target_view = target_view.cuda()
            pred_view ,pred_AS = self.model(cine) # Bx3xTxHxW
            loss_AS = self._get_loss(pred_AS, target_AS, self.num_classes_AS)
            loss_view = self._get_loss(pred_view.view(pred_view.shape[0]*pred_view.shape[1],3), 
                           target_view.view(target_view.shape[0]*target_view.shape[1])
                           ,self.num_classes_view)
            loss = loss_AS + self.lamda * loss_view
            losses += [loss]

            argm_AS, _, _, _, _, _ = self._get_prediction_stats(pred_AS, self.num_classes_AS)
            argm_view, _, _, _, _, _ = self._get_prediction_stats(pred_view.view(pred_view.shape[0]*pred_view.shape[1],3), self.num_classes_view)
            conf_AS = utils.update_confusion_matrix(conf_AS, target_AS.cpu(), argm_AS.cpu())
            conf_view = utils.update_confusion_matrix(conf_view, target_view.view(target_view.shape[0]*target_view.shape[1]).cpu(), argm_view.cpu())
            
        loss_avg = torch.mean(torch.stack(losses)).item()
        print(conf_AS)
        print(conf_view)
        acc_AS = utils.acc_from_confusion_matrix(conf_AS)
        acc_view = utils.acc_from_confusion_matrix(conf_view)

        # Switch the model into training mode
        self.model.train()
        return acc_AS, acc_view ,loss_avg

    
    @torch.no_grad()
    def test_comprehensive(self, loader, mode="test",record_embeddings=False):
        """Logs the network outputs in dataloader
        computes per-patient preds and outputs result to a DataFrame"""
        print('NOTE: test_comprehensive mode uses batch_size=1 to correctly display metadata')
        # Choose which model to evaluate.
        if mode=="test":
            self._restore(self.bestmodel_file_load)
        # Switch the model into eval mode.
        self.model.eval()
        patient, view, as_label = [], [], []
        target_AS_arr, pred_AS_arr = [], []
        normal, mild, moderate, severe = [], [], [], []
        target_view_arr, pred_view_arr = [], []
        max_AS_arr, entropy_AS_arr, vacuity_AS_arr, uni_AS_arr = [], [], [], []
        max_view_arr, entropy_view_arr, vacuity_view_arr, uni_view_arr = [], [], [], []
        relevance_weight = []
        
        
        for data in tqdm(loader):
            cine = data['x']
            target_view = data['y_view']
            target_AS = data['y_AS']
            p_id = data['p_id']
            # Transfer data from CPU to GPU.
            # Cross Entropy Training

            # Transfer data from CPU to GPU.
            if self.config['use_cuda']:
                cine = cine.cuda()
                target_AS = target_AS.cuda()
                target_view = target_view.cuda()
                
            # collect metadata from data_info
            patient.append(p_id[0])
            view.append(target_view.cpu().numpy()[0])
            as_label.append(target_AS.cpu().numpy()[0])
            
            # get the model prediction
            pred_view ,pred_AS = self.model(cine) # Bx3xTxHxW

            argm_view, max_p_view, prob_view,ent_view, vac_view, uni_view = self._get_prediction_stats(pred_view, self.num_classes_view)
            view_relevance_weight = prob_view[0][0]+prob_view[0][1]
            
            argm_AS, max_p_AS, prob_AS,ent_AS, vac_AS, uni_AS = self._get_prediction_stats(pred_AS, self.num_classes_AS)
            pred_AS_arr.append(argm_AS.cpu().numpy()[0])
            normal.append(prob_AS[0][0].cpu().numpy())
            mild.append(prob_AS[0][1].cpu().numpy())
            moderate.append(prob_AS[0][2].cpu().numpy())
            #severe.append(prob_AS[0][3].cpu().numpy())
            max_AS_arr.append(max_p_AS.cpu().numpy()[0])
            entropy_AS_arr.append(ent_AS.cpu().numpy()[0])
            pred_view_arr.append(argm_view.cpu().numpy()[0])
            max_view_arr.append(max_p_view.cpu().numpy()[0])
            entropy_view_arr.append(ent_view.cpu().numpy()[0])
            relevance_weight.append(view_relevance_weight.cpu().numpy())
        
                
        # compile the information into a dictionary
        d = {'id':patient, 'view':view, 'as':as_label, 
             'pred_AS':pred_AS_arr, 'max_AS':max_AS_arr,
             'ent_AS':entropy_AS_arr, 'pred_view':pred_view_arr, 'max_AS':max_view_arr,
             'ent_AS':entropy_view_arr,'ent_view':entropy_view_arr,'relevance_weight':relevance_weight,
             'normal': normal,'mild': mild, 'moderate': moderate, #'severe':severe
             }
        df = pd.DataFrame(data=d)
        # save the dataframe
        test_results_file = os.path.join(self.log_dir, mode+".csv")
        df.to_csv(test_results_file)


# if __name__ == "__main__":
#     """Main for mock testing."""
#     from get_config import get_config
#     from dataloader.as_dataloader_revision import get_as_dataloader
#     from get_model import get_model

#     config = get_config()
    
#     if config['use_wandb']:
#         run = wandb.init(project="as_v2", entity="guangnan", config=config)
    
#     model = get_model(config)
#     net = Network(model, config)
#     dataloader_tr = get_as_dataloader(config, split='train', mode='train')
#     dataloader_va = get_as_dataloader(config, split='val', mode='val')
#     dataloader_te = get_as_dataloader(config, split='test', mode='test')
    
#     if config['mode']=="train":
#         net.train(dataloader_tr, dataloader_va)
#         net.test_comprehensive(dataloader_te, mode="test")
#     if config['mode']=="test":
#         net.test_comprehensive(dataloader_te, mode="test")
#     if config['use_wandb']:
#         wandb.finish()