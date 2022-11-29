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
from sklearn.metrics import confusion_matrix




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
    
def NPairsample():
    labels = torch.arange(32)
    # Reshape labels to compute adjacency matrix.
    labels_reshaped = torch.reshape(labels, (labels.shape[0], 1))
    labels_remapped_p2 = 1*(torch.eq(labels_reshaped, (labels_reshaped + 2).T))
    labels_remapped_p1 = 1*(torch.eq(labels_reshaped, (labels_reshaped + 1).T))
    labels_remapped_0 = 1*(torch.eq(labels_reshaped, (labels_reshaped).T))
    labels_remapped_b1 = 1*(torch.eq(labels_reshaped, (labels_reshaped - 1).T))
    labels_remapped_b2 = 1*(torch.eq(labels_reshaped, (labels_reshaped - 2).T))
    neg = 1 - (labels_remapped_p2 + labels_remapped_p1 + labels_remapped_0 + labels_remapped_b1 + labels_remapped_b2) 
    pos = labels_remapped_p2 + labels_remapped_p1 + labels_remapped_b1 + labels_remapped_b2
    pos = pos.float()/ torch.sum(pos , dim=1, keepdims=True)
    return pos.cuda(),neg.cuda()
    
    

class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, model, config):
        """Initialize configuration."""
        self.config = config
        self.model = None
        self.model = model
        self.encoder = TransformerFeatureMap(self.model.cuda())
        # pos and neg matrix for npair_loss
        self.pos , self.neg = NPairsample()
        if config['cotrastive_method']=='Linear':
            checkpoint = torch.load(Path('/AS_Neda/FTC/logs/tad_1e-4_added_losses_try2/checkpoint.pth'))
            self.model.load_state_dict(checkpoint["model"], strict=False)
            print("Checkpoint_loaded")
        if self.config['use_cuda']:  
            print(torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        self.num_classes_AS = config['num_classes']
        if config['cotrastive_method']=='Linear':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config['lr'])
            
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,T_max=config['num_epochs'])
        self.loss_type = config['loss_type']
        self.contrastive_method = config['cotrastive_method']
        self.temperature = config['temp']
        #self.bicuspid_weight = config['bicuspid_weight']
        # Recording the training losses and validation performance.
        self.train_losses = []
        self.valid_oas = []
        self.idx_steps = []

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
        #self.bestmodel_file = os.path.join(self.log_dir, "best_model.pth")
        self.bestmodel_file = Path('/AS_Neda/FTC/logs/tad_1e-4_tclloss_batch16/best_model.pth')
        self.bestmodel_file_contrastive = os.path.join(self.log_dir, "best_model_cont.pth")
        
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
        return argm, max_percentage, entropy, vacuity, uni
        
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
            
            with tqdm(total=len(loader_tr)) as pbar:
                for data in loader_tr:
                    cine = data[0]
                    target_AS = data[1]
                    target_B = data[2]

                    # Cross Entropy Training
                    if self.config['cotrastive_method'] == 'CE' or self.config['cotrastive_method'] == "Linear":
                        # Transfer data from CPU to GPU.
                        if self.config['use_cuda']:
                            if self.config['model'] == 'slowfast':
                                cine = [c.cuda() for c in cine]
                            else:
                                cine = cine.cuda()
                            target_AS = target_AS.cuda()
                            target_B = target_B.cuda()
                        if self.config['model'] == "FTC_TAD":
                            pred_AS,entropy_attention,outputs = self.model(cine) # Bx3xTxHxW
                            
                            # Calculating temporal coherent npair loss
                            similarity_matrix = (torch.bmm(outputs,outputs.permute((0,2,1)))/1024)
                            self.pos_e = self.pos.repeat(len(pred_AS),1,1)
                            self.neg_e = self.neg.repeat(len(pred_AS),1,1)
                            npair_loss = torch.mean(-torch.sum(self.pos_e*similarity_matrix,dim =2) + 
                               torch.log(torch.exp(torch.sum(self.pos_e*similarity_matrix,dim=2))+
                               torch.sum(self.neg_e*torch.exp(self.neg_e*similarity_matrix),dim=2)), dim = 1)
                            
                            loss = self._get_loss(pred_AS, target_AS, self.num_classes_AS)
                            loss = loss + 0.05*(torch.mean(entropy_attention)) +0.1*torch.mean(npair_loss)
                            losses += [loss] 
                        
                        else:
                            pred_AS = self.model(cine) # Bx3xTxHxW
                            loss = self._get_loss(pred_AS, target_AS, self.num_classes_AS)
                            losses += [loss]
                    # Contrastive Learning
                    else:
                        cines = torch.cat([cine[0], cine[1]], dim=0)
                        if self.config['use_cuda']:
                            cines = cines.cuda()
                            target_AS = target_AS.cuda()
                            target_B = target_B.cuda()
                        bsz = target_AS.shape[0]
                        features = self.model(cines) # Bx3xTxHxW
                        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                        if self.config['cotrastive_method'] == 'SupCon':
                            loss = self._get_loss(features, target_AS, self.num_classes_AS)
                        elif self.config['cotrastive_method'] == 'SimCLR':
                            loss = self._get_loss(features, target_AS, self.num_classes_AS)
                        else:
                            raise ValueError('contrastive method not supported: {}'.
                                             format(self.config['cotrastive_method']))

                        losses += [loss]
                        
                    # Calculate the gradient.
                    loss.backward()
                    # Update the parameters according to the gradient.
                    self.optimizer.step()
                    # Zero the parameter gradients in the optimizer
                    self.optimizer.zero_grad() 
                    pbar.set_postfix_str("loss={:.4f}".format(loss.item()))
                    pbar.update()

            #loss_avg_AS = torch.mean(torch.stack(losses_AS)).item()
            #loss_avg_B = torch.mean(torch.stack(losses_B)).item()
            loss_avg = torch.mean(torch.stack(losses)).item()
            #acc_AS, f1_B, val_loss = self.test(loader_va, mode="val")
            if self.config['cotrastive_method'] == 'CE' or self.config['cotrastive_method'] == 'Linear':
                ## Validation Data Evaluration 
                acc_AS, val_loss = self.test(loader_va, mode="val")
                if self.config['use_wandb']:
                    wandb.log({"tr_loss":loss_avg, "val_loss":val_loss, "val_AS_acc":acc_AS})
                    # wandb.log({"tr_loss_AS":loss_avg_AS, "tr_loss_B":loss_avg_B, "tr_loss":loss_avg,
                    #            "val_loss":val_loss, "val_B_f1":f1_B, "val_AS_acc":acc_AS})
                ## Test Data Evaluration    
                acc_AS_te, te_loss = self.test(loader_te, mode="val")
                if self.config['use_wandb']:
                    wandb.log({ "te_loss":te_loss, "test_AS_acc":acc_AS_te})

                # Save model every epoch.
                self._save(self.checkpts_file)

                # Early stopping strategy.
                if acc_AS > best_va_acc:
                    # Save model with the best accuracy on validation set.
                    best_va_acc = acc_AS
                    #best_B_f1 = f1_B
                    self._save(self.bestmodel_file)
                # print(
                #     "Epoch: %3d, loss: %.5f/%.5f, val loss: %.5f, acc: %.5f/%.5f, top AS acc: %.5f/%.5f"
                #     % (epoch, loss_avg_AS, loss_avg_B, val_loss, acc_AS, f1_B, best_va_acc, best_B_f1)
                # )
                print(
                    "Epoch: %3d, loss: %.5f, val loss: %.5f, acc: %.5f, top AS acc: %.5f"
                    % (epoch, loss_avg, val_loss, acc_AS, best_va_acc)
                ) 

                # Recording training losses and validation performance.
                self.train_losses += [loss_avg]
                self.valid_oas += [acc_AS]
                self.idx_steps += [epoch]
                
            elif self.config['cotrastive_method'] == 'SupCon' or self.config['cotrastive_method'] == 'SimCLR':
                if epoch%5==0:
                    val_acc = validation_constructive(self.model,self.config)
                    if self.config['use_wandb']:
                        wandb.log({"AMI":val_acc['AMI'],"NMI":val_acc['NMI'],
                                   "precision_at_1":val_acc['precision_at_1']})
                if (val_acc['precision_at_1'] > best_va_acc_supcon and self.config['cotrastive_method'] == 'SupCon'):
                    # Save model with the best accuracy on validation set.
                    best_va_acc_supcon = val_acc['precision_at_1']
                    self._save(self.bestmodel_file_contrastive)
                    print('Model Saved')
                
                if self.config['cotrastive_method'] == 'SimCLR':
                    self._save(self.bestmodel_file_contrastive)
                    print('Model Saved')
                    
                if self.config['use_wandb']:
                    wandb.log({"contrastive_loss":loss_avg})
                    
                print(
                    "Epoch: %3d, loss: %.5f,precision_at_1: %.5f"
                    % (epoch, loss_avg,val_acc['precision_at_1'])
                ) 
               
            
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
        #conf_B = np.zeros((2,2))
        losses = []
        for data in tqdm(loader_te):
            cine = data[0]
            target_AS = data[1]
            target_B = data[2]
            # Transfer data from CPU to GPU.
            # Cross Entropy Training
            if self.config['cotrastive_method'] == 'CE' or self.config['cotrastive_method'] == "Linear":
                # Transfer data from CPU to GPU.
                if self.config['use_cuda']:
                    if self.config['model'] == 'slowfast':
                        cine = [c.cuda() for c in cine]
                    else:
                        cine = cine.cuda()
                    target_AS = target_AS.cuda()
                    target_B = target_B.cuda()
                    
                if self.config['model'] == "FTC_TAD":
                    pred_AS,entropy_attention,outputs = self.model(cine) # Bx3xTxHxW
                else:
                    pred_AS = self.model(cine) # Bx3xTxHxW
                loss = self._get_loss(pred_AS, target_AS, self.num_classes_AS)
                losses += [loss]

            # Contrastive Learning
            else:
                cines = torch.cat([cine[0], cine[1]], dim=0)
                if self.config['use_cuda']:
                    cines = cines.cuda()
                    target_AS = target_AS.cuda()
                    target_B = target_B.cuda()
                bsz = target_AS.shape[0]
                features = self.model(cines) # Bx3xTxHxW
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                if self.config['cotrastive_method'] == 'SupCon':
                    loss = self._get_loss(features, target_AS, self.num_classes_AS)
                elif self.config['cotrastive_method'] == 'SimCLR':
                    loss = self._get_loss(features, target_AS, self.num_classes_AS)
                else:
                    raise ValueError('contrastive method not supported: {}'.
                                     format(self.config['cotrastive_method']))
                losses += [loss]
            
            #argmax_pred_AS = torch.argmax(pred_AS, dim=1)
            #argmax_pred_B = torch.argmax(pred_B, dim=1)
            if self.config['cotrastive_method'] == 'CE' or self.config['cotrastive_method'] == 'Linear':
                argm_AS, _, _, _, _ = self._get_prediction_stats(pred_AS, self.num_classes_AS)
                #argm_B, _, _, _, _ = self._get_prediction_stats(pred_B, 2)
                conf_AS = utils.update_confusion_matrix(conf_AS, target_AS.cpu(), argm_AS.cpu())
                #conf_B = utils.update_confusion_matrix(conf_B, target_B.cpu(), argm_B.cpu())
        if self.config['cotrastive_method'] == 'CE' or self.config['cotrastive_method'] == 'Linear':    
            loss_avg = torch.mean(torch.stack(losses)).item()
            acc_AS = utils.acc_from_confusion_matrix(conf_AS)
            print(conf_AS)
            #f1_B = utils.f1_from_confusion_matrix(conf_B)

            # Switch the model into training mode
            self.model.train()
            return acc_AS, loss_avg
        else:
            loss_avg = torch.mean(torch.stack(losses)).item()
            self.model.train()
            return loss_avg
    
    @torch.no_grad()
    def test_comprehensive(self, loader, mode="test",record_embeddings=False):
        """Logs the network outputs in dataloader
        computes per-patient preds and outputs result to a DataFrame"""
        print('NOTE: test_comprehensive mode uses batch_size=1 to correctly display metadata')
        # Choose which model to evaluate.
        if mode=="test":
            self._restore(self.bestmodel_file)
        # Switch the model into eval mode.
        self.model.eval()
        fn, patient, view, age, lv, as_label,bicuspid = [], [], [], [], [], [],[]
        target_AS_arr, target_B_arr, pred_AS_arr, pred_B_arr = [], [], [], []
        max_AS_arr, entropy_AS_arr, vacuity_AS_arr, uni_AS_arr = [], [], [], []
        #max_B_arr, entropy_B_arr, vacuity_B_arr = [], [], []
        predicted_qual = []
        embeddings = []
        
        for cine, target_AS, target_B, data_info, cine_orig in tqdm(loader):
            # collect the label info
            target_AS_arr.append(int(target_AS[0]))
            target_B_arr.append(int(target_B[0]))
            # Transfer data from CPU to GPU.
            if self.config['use_cuda']:
                cine = cine.cuda()
                target_AS = target_AS.cuda()
                target_B = target_B.cuda()
                
            # collect metadata from data_info
            fn.append(data_info['path'][0])
            patient.append(int(data_info['patient_id'][0]))
            view.append(data_info['view'][0])
            age.append(int(data_info['age'][0]))
            #lv.append(float(data_info['LVMass indexed'][0]))
            as_label.append(data_info['as_label'][0])
            bicuspid.append(data_info['Bicuspid'][0])
            # pvq = (data_info['predicted_view_quality'][0] * 
            #        data_info['predicted_view_probability'][0]).cpu().numpy()
            # predicted_qual.append(pvq)
            
            # get the model prediction
            # pred_AS, pred_B = self.model(cine) #1x3xTxHxW
            if self.config['model'] == "FTC_TAD":
                pred_AS,entropy_attention,outputs = self.model(cine) # Bx3xTxHxW
            else:
                pred_AS = self.model(cine) # Bx3xTxHxW
            # collect the model prediction info
            argm, max_p, ent, vac, uni = self._get_prediction_stats(pred_AS, self.num_classes_AS)
            pred_AS_arr.append(argm.cpu().numpy()[0])
            max_AS_arr.append(max_p.cpu().numpy()[0])
            entropy_AS_arr.append(ent.cpu().numpy()[0])
            if self.loss_type == 'evidential':
                vacuity_AS_arr.append(vac.cpu().numpy()[0])
            else:
                vacuity_AS_arr.append(vac)
            uni_AS_arr.append(uni[0])
            
            if record_embeddings:
                embeddings += [embedding[0].squeeze().numpy()]

                
        # compile the information into a dictionary
        d = {'path':fn, 'id':patient, 'view':view, 'age':age, 'as':as_label, 'bicuspid': bicuspid ,
             'GT_AS':target_AS_arr, 'pred_AS':pred_AS_arr, 'max_AS':max_AS_arr,
             'ent_AS':entropy_AS_arr, 'vac_AS':vacuity_AS_arr, 'uni_AS':uni_AS_arr,
             # 'GT_B':target_B_arr, 'pred_B':pred_B_arr, 'max_B':max_B_arr,
             # 'ent_B':entropy_B_arr, 'vac_B':vacuity_B_arr, 
             }
        df = pd.DataFrame(data=d)
        # save the dataframe
        test_results_file = os.path.join(self.log_dir, mode+".csv")
        df.to_csv(test_results_file)
        if record_embeddings:
            embeddings = np.array(embeddings)
            num_batches, b, d = embeddings.shape
            #print(num_batches, b, d)
            embeddings = np.reshape(embeddings, (num_batches*b, d))
            tsne_save_file =  os.path.join(self.log_dir, mode+"_tsne.html")
            plot_tsne_visualization(X=embeddings, y=as_label, info=fn, title=tsne_save_file , b = bicuspid)


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
