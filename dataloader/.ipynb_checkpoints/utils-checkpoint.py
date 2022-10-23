import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import os

def update_confusion_matrix(mat, true, pred):
    '''
    updates a confusion matrix in numpy based on target/pred results
    with true results based on the row index, and predicted based on column

    Parameters
    ----------
    mat : int
        NxN confusion matrix
    true : Bx1 List[int]
        list of integer labels
    pred : Bx1 List[int]
        list of integer labels

    Returns
    -------
    NxN confusion matrix

    '''
    for i in range(len(true)):
        mat[true[i],pred[i]] = mat[true[i],pred[i]] + 1
    return mat

def acc_from_confusion_matrix(mat):
    # get accuracy from NxN confusion matrix
    return np.trace(mat)/np.sum(mat)

def f1_from_confusion_matrix(mat):
    # get f1-score from a 2x2 confusion matrix, assuming the "1" class is positive
    tp = mat[1,1]
    fp = mat[0,1] # it's actually negative(0) but we think it's positive(1)
    fn = mat[1,0] # it's actually + but we think it's -
    return tp / (tp + 0.5*fp + 0.5*fn)

# visualizes the augmented as video data vs the original video data
def visualize_as_video(augmented_video, orig_video, 
                       data_info=None, output_path=None):
    '''
    visualizes the augmented as video data vs the original video data
    the output will be the length of the longer video

    Parameters
    ----------
    augmented_video : TxHxW numpy array
    orig_video : T2xHxW numpy array
    data_info : dictionary containing metadata to display in title
        see dataloader for dictionary format
    output_path : string
    
    Returns
    -------
    None.

    '''
    if augmented_video.shape[0] == 3: # if grayscale, select the first channel
        augmented_video = augmented_video[0]
    t1, h, w = augmented_video.shape
    t2, h2, w2 = orig_video.shape
    if t1 < t2:
        num_repeats = int(t2/t1)
        mod = t2%t1
        looped_va = np.tile(augmented_video, (num_repeats,1,1))
        looped_va = np.concatenate((looped_va, augmented_video[:mod]), axis=0)
        looped_vo = orig_video
    elif t1 == t2:
        looped_vo = orig_video
        looped_va = augmented_video
    else:
        num_repeats = int(t1/t2)
        mod = t1%t2
        looped_vo = np.tile(orig_video, (num_repeats,1,1))
        looped_vo = np.concatenate((looped_vo, orig_video[:mod]), axis=0)
        looped_va = augmented_video
    
    # save images to be used during video construction
    n = looped_va.shape[0]
    frames_path = "{i}.jpg"
    
    for i in range(n):
        fig, axes = plt.subplots(1,2, figsize=(18,6))
        axes[0].imshow(looped_va[i])
        axes[0].set_title("Augmented video")
        axes[1].imshow(looped_vo[i])
        axes[1].set_title("Original video")
        if data_info is not None:
            video_path = data_info['path'][0]
            severity = data_info['as_label'][0]
            title = "Frame = {0}/{1}, video: {2}, GT: {3}" \
                .format(i, n, video_path, severity)
        fig.suptitle(title)
        plt.savefig(frames_path.format(i=i))
        plt.close('all')
        
    # create the video
    if output_path is None:
        vid_path = "test.gif"
    else:
        vid_path = output_path
    
    with iio.get_writer(vid_path, mode='I') as writer:
        for i in range(n):
            writer.append_data(iio.imread(frames_path.format(i=i)))
    
    # remove the temporarily created jpegs
    for i in range(n):
        os.remove(frames_path.format(i=i))
    
    return looped_vo, looped_va

def test_unimodality(prob):
    '''
    test to see if the categorical distributions are unimodal
    assuming a sort of "ordinal relationship" between the leftmost
    and rightmost class

    Parameters
    ----------
    prob : ndarray
        BxC collection of categorical distributions with C classes

    Returns
    -------
    B-length list of integers, 1 if distribution is unimodal

    '''
    # we check to see if the indices of the largest and second-largest
    # elements are adjacent, note that this isn't perfect
    # but generally speaking it does the job
    
    # this is fine because the default numpy sort is in-place
    sorted_args = np.argsort(prob, axis=1)
    largest_arg = sorted_args[:,-1]
    second_largest_arg = sorted_args[:,-2]
    uni = [(n <= 1) for n in np.abs(largest_arg - second_largest_arg)]
    return uni
    