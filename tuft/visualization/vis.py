# Helper functions to visualize dataframe results

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import precision_recall_curve, average_precision_score
#from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px

def plot_tsne_visualization(X, y=None, info=None, title=None,b = None):
    # plots the 2D TNSE representation of the input
    # exepcts input X to be a NxD numpy array
    # expects y to be optional N array containing labels
    # expects info to be an optional N array containing string data
    Xe = tsne(X, compress=False)
    if y is None:
        if info is None:
            fig = px.scatter(x=Xe[:,0], y=Xe[:,1])
        else:
            df = pd.DataFrame({'x1':Xe[:,0], 'x2':Xe[:,1], 'info':info})
            fig = px.scatter(df, x='x1', y='x2', click_data=['info'])
    else:
        if info is None:
            fig = px.scatter(x=Xe[:,0], y=Xe[:,1], color=y)
        if b is not None:
            df = pd.DataFrame({'x1':Xe[:,0], 'x2':Xe[:,1], 'y':y, 'info':info,'bicuspid': b})
            fig = px.scatter(df, x='x1', y='x2', color='y', hover_data=['info','bicuspid'])
        else:
            df = pd.DataFrame({'x1':Xe[:,0], 'x2':Xe[:,1], 'y':y, 'info':info})
            fig = px.scatter(df, x='x1', y='x2', color='y', hover_data=['info'])
    if title is None:
        fig.write_html("tsne.html")
    else:
        fig.write_html(title)
def tsne(X, compress=True):
    # returns 2D TNSE representation of the input
    # exepcts input X to be a NxD numpy array
    D_MAX = 50
    N, D = X.shape
    if compress and D > D_MAX:
        X = PCA(n_components=D_MAX).fit_transform(X)
    X_embedded = TSNE(n_components=2, init='random').fit_transform(X)
    return X_embedded

def plot_confusion_matrix(true, pred, run_name, task_name, classes, normalize=False):
    norm_option=None
    num_format='d'
    if normalize:
        norm_option='true'
        num_format='.2f'
    cf = confusion_matrix(true, pred, normalize=norm_option)
    #true_counts = np.sum(cf, axis=1)
    #if normalize:
    #    cf = cf / true_counts
    ax = draw_confusion_matrix(cf, classes, num_format)
    overall_acc = np.mean(np.array(true==pred))
    
    title = 'Run ' + run_name + ', ' + task_name 
    acc_str = ', accuracy={0:0.3f}'.format(overall_acc)
    plt.title(title + acc_str, fontsize=12)
    
    return ax
    
def draw_confusion_matrix(cf, names, num_format=None):
    disp = ConfusionMatrixDisplay(confusion_matrix = cf, display_labels=names)
    fig, ax_cf = plt.subplots(figsize=(12,8))
    plt.rcParams.update({'font.size': 13})
    disp.plot(values_format=num_format, ax=ax_cf, xticks_rotation='45')
    #disp.plot(values_format=num_format, ax=ax_cf,  cmap='gray', xticks_rotation='45')
#    disp.plot(include_values=include_values,
#                     cmap='gray', ax=ax_cf, xticks_rotation='45',
#                     values_format=values_format, colorbar=colorbar)
    return ax_cf

def aggregate_dataframe(df, strategy="majority", include_quality=False):
    # aggregates results from the dataframe by patientID
    # returns another dataframe with patient-level classifications
    # strategy can be consensus, majority, or weighted
    # if include_quality is true, we account for viewquality during weighting
    agg_GT_AS = []
    agg_pred_AS = []
    agg_GT_B = []
    agg_id = []
    unique_ids = df['id'].unique()
    for i in unique_ids:
        dfi = df[df['id']==i]
        GT_AS = list(dfi['GT_AS'].mode())[-1]
        if strategy=="majority":
            if include_quality:
                # weighted majority with view quality
                groupsum = dfi.groupby(['pred_AS'])['pvq'].sum().reset_index()
                # take the class with the highest predicted quality
                ind = groupsum['pvq'].argmax()
                pred_AS = groupsum['pred_AS'][ind]
            else:
                # find mode, if more than 1 mode, then select maximum severity
                pred_AS = list(dfi['pred_AS'].mode())[-1]
        elif strategy=="worst":
            pred_AS = dfi['pred_AS'].max()
        elif strategy=="consensus":
            # only keep the prediction if prediction from all videos agree
            if len(dfi['pred_AS'].unique()) == 1:
                pred_AS = dfi['pred_AS'].max()
            else:
                continue
        elif "weighted" in strategy:
            if strategy=="weighted_AS":
                # find the 1-entropy for each sample
                inv_ent = np.log(4) - dfi['ent_AS']
            else:
                inv_ent = np.log(2) - dfi['ent_B']
            if include_quality:
                inv_ent = inv_ent * dfi['pvq']
            # group by predicted AS, then look at the sum of entropy for each class
            dfi.insert(0, "inv_ent", inv_ent)
            groupsum = dfi.groupby(['pred_AS'])['inv_ent'].sum().reset_index()
            # take the class with the highest inverse entropy
            ind = groupsum['inv_ent'].argmax()
            pred_AS = groupsum['pred_AS'][ind]
        else:
            raise NotImplementedError
        GT_B = list(dfi['GT_B'].mode())[-1]
        agg_id.append(i)
        agg_GT_AS.append(GT_AS)
        agg_pred_AS.append(pred_AS)
        agg_GT_B.append(GT_B)
    d = {"id":agg_id, "GT_AS":agg_GT_AS, "pred_AS":agg_pred_AS, "GT_B":agg_GT_B}
    return pd.DataFrame(data=d)
           
def find_rejection_characteristic(true, pred, uncertainty):
    # finds the accuracy of the model at different steps in the rejection of uncertainty
    # expects 3 numpy arrays, true, pred, and uncertainty
    
    # make a linspace from the maximum and minimum uncertainty values
    unc_range = np.linspace(np.max(uncertainty), np.min(uncertainty), num=100)
    # take the first 99 values
    unc_range = unc_range[:-1]
    
    # find the accuracy for each uncertainty threshold in the range
    accs = []
    count = []
    for u in unc_range:
        remaining = [x<=u for x in uncertainty]
        true_remaining = true[remaining]
        pred_remaining = pred[remaining]
        accs.append(np.mean(true_remaining==pred_remaining))
        count.append(np.sum(remaining))
    return count, accs

def plot_rejection_characteristic(counts, accs, labels, plot_title):
    for i in range(len(counts)):
        plt.plot(counts[i], accs[i])
    plt.xlabel("number of samples remaining")
    plt.ylabel("accuracy of remaining samples")
    plt.title(plot_title)
    plt.legend(labels)
    
# def plot_rejection_characteristic(e_test, e_valid, s_test, s_valid, cutoff=100):
#     # dicts should contain specified keys to arrays
#     # 'unc' = uncertainty cutoff points
#     # 'rej' = rejection ratio of samples at corresponding unc. cutoff pt.
#     # 'acc' = accuracy for remaining samples at corresponding unc. cutoff pt.
#     fig, ax = plt.subplots(figsize=(8,8))
# #    ax.plot(x1, a1, 'r')
# #    ax.plot(x1, r1, 'r--')
# #    ax.plot(x2, a2, 'b')
# #    ax.plot(x2, r2, 'b--')
# #    plt.xlim((0.0, 1.01))
# #    plt.ylim((0.0, 1.01))
# #    plt.legend(['acc of remaining, evid.','remaining pop\'n, evid.',
# #                'acc of remaining, softmax','remaining pop\'n, softmax'])
# #    plt.xlabel('Uncertainty allowed for samples')
# #    plt.ylabel('Percentage')
    
#     # for evidential, find the elbow based on the validation set
#     elbow_idx = find_elbow(e_valid['rej'][:cutoff], e_valid['acc'][:cutoff])
#     # find uncertainty threshold at the elbow
#     elbow_unc = e_valid['unc'][elbow_idx]
#     print(elbow_unc)
#     # find the uncertainty threshold on the test set closest to that one
#     elbow_test_idx = np.argmin(np.abs(e_test['unc'] - elbow_unc))
    
#     # for softmax, find the uncertainty threshold closest to 0.02  (>98% certain)
#     unc_99_idx = np.argmin(np.abs(s_valid['unc'] - 0.02))
#     unc_test_idx = np.argmin(np.abs(s_test['unc'] - s_valid['unc'][unc_99_idx]))
    
#     ax.plot(e_test['rej'][:cutoff], e_test['acc'][:cutoff], 'r')
#     ax.plot(e_valid['rej'][:cutoff], e_valid['acc'][:cutoff], 'r--')
#     ax.plot(s_test['rej'], s_test['acc'], 'b')
#     ax.plot(s_valid['rej'], s_valid['acc'], 'b--')
    
#     ax.plot(e_test['rej'][elbow_test_idx], e_test['acc'][elbow_test_idx], 'ro')
#     ax.plot(e_valid['rej'][elbow_idx], e_valid['acc'][elbow_idx], 'ro')
#     ax.plot(s_test['rej'][unc_test_idx], s_test['acc'][unc_test_idx], 'bo')
#     ax.plot(s_valid['rej'][unc_99_idx], s_valid['acc'][unc_99_idx], 'bo')
#     #plt.xlim((0.0, 1.01))
#     #plt.ylim((0.85, 1.01))
#     ax.legend(['evidential on test','evidential on validation',
#                 'softmax on test', 'softmax on validation'], loc=4, fontsize=15)
#     plt.ylabel('Accuracy of remaining samples', fontsize=15)
#     plt.xlabel('Percentage of samples rejected', fontsize=15)
#     plt.title('Comparison of Efficient EvidNet vs Softmax for varying rejection rates', fontsize=15)
#     plt.savefig('rejection.eps')
    
#     # get the accuracy for each run
#     stats={}
#     stats['sm_base_acc']=s_test['acc'][0]
#     stats['sm_rej_acc']=s_test['acc'][unc_test_idx]
#     stats['sm_rej_ratio']=s_test['rej'][unc_test_idx]
#     stats['sm_val_acc']=s_valid['acc'][unc_99_idx]
#     stats['sm_val_ratio']=s_valid['rej'][unc_99_idx]
    
#     stats['e_base_acc']=e_test['acc'][0]
#     stats['e_rej_acc']=e_test['acc'][elbow_test_idx]
#     stats['e_rej_ratio']=e_test['rej'][elbow_test_idx]
#     stats['e_val_acc']=e_valid['acc'][elbow_idx]
#     stats['e_val_ratio']=e_valid['rej'][elbow_idx]
#     return stats
    
# def find_elbow(list_x, list_y):
#     # knee finding algorithm
#     #points = find_elbow(e_valid['rej'], e_valid['acc'])
#     x = np.array(list_x)
#     y = np.array(list_y)
#     points = np.stack((x,y), axis=-1)
#     points = points - points[0,:]
#     q = points[-1,:] - points[0,:]
#     q_norm = q/np.sqrt(np.sum(q**2))
#     p_dot_q = np.matmul(points, q_norm)
#     proj_p_on_q = q_norm*p_dot_q.reshape(len(p_dot_q),1)
#     dist = np.sum((proj_p_on_q - points)**2, axis=1)
# #    fig = plt.figure()
# #    ax1 = fig.add_subplot(111)
# #    
# #    ax1.scatter(proj_p_on_q[:,0], proj_p_on_q[:,1], s=10, c='b', marker="s", label='first')
# #    ax1.scatter(points[:,0], points[:,1], s=10, c='r', marker="o", label='second')
    
#     return np.argmax(dist)