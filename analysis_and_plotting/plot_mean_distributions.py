import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import bootstrapped.bootstrap as bs

import bootstrapped.stats_functions as bs_stats
import math
import os
import matplotlib as mpl
import datetime
import itertools
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import pdb
from scipy import ndimage, stats
from sklearn.mixture import GaussianMixture
from sklearn import metrics,preprocessing
from sklearn.linear_model import LinearRegression
from glob import glob
import warnings
import json
import pandas as pd
import seaborn as sns
import pickle
import re
import scipy.stats
import sys
from functools import lru_cache

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 50

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titleo
sns.set_palette('colorblind')


plots_folder='/om/user/francl/{date:%Y-%m-%d}_plots'.format(date=datetime.datetime.now())

try:
    os.makedirs(plots_folder)
except OSError:
    if not os.path.isdir(plots_folder):
        raise


sign = lambda x: (1, -1)[x < 0]
flatten = lambda l: [item for sublist in l for item in sublist]

convert_from_numpy = lambda x: x[0] if len(x.tolist()) ==1 else x.tolist()

val_in_list_fuzzy = lambda val,val_list,fuzzy_mult: any(
    [element*(1-fuzzy_mult) <= val <= element*(1+fuzzy_mult) for
     element in val_list])

val_match_target_fuzzy = lambda val,target,fuzzy_mult:\
    val*(1-fuzzy_mult) <= target <= val*(1+fuzzy_mult)

float_from_string_of_bytes = lambda x : float(x.split('\'')[1]) if 'b' in x else float(x)
invert = lambda x: -x


def tuplizer(x):
   return tuple(x) if isinstance(x, (np.ndarray, list)) else x
    
def group_all_but(dataframe,target_var,grouping_var='arch_index'):
    '''
    Calculates the mean value of a given column by aggragating over all values
    of target_var where every column but the grouping variable are the same.
    Patricularly useful for collapsing over architectures.
    Parameters:
        dataframe (Pandas dataframe): dataframe that will be grouped
        target_var (string) : name of column over which mean will be calculated
        grouping_var (string): name of variable to be collapsed and removed
        from the array
    Returns:
        dataframe_mean (pandas dataframe): dataframe with mean of target varible
        calculated of all values of grouping var
    '''
    excluded = [grouping_var, target_var]
    other_vars = [x for x in dataframe.columns if x not in excluded]
    dataframe_arrays_removed = dataframe.applymap(tuplizer)
    dataframe_mean = (dataframe_arrays_removed.groupby(other_vars)[target_var]
                      .agg([np.mean,'count','std'])
                      .rename(columns={'mean':target_var+'_mean',
                                       'std': target_var+'_std',
                                       'count':target_var+'_count'})
                      .reset_index())
    return dataframe_mean

def aggregate_and_filter(dataframe,target_var,grouping_var_list):
    '''
    Calculates the mean value of a given column by aggragating over all 
    target values where the grouping variable are the same. 
    Useful for collapsing over architectures right before plotting where
    independent variables are in the grouping variable list and dependent
    variables are target varibales.
    Parameters:
        dataframe (Pandas dataframe): dataframe that will be grouped
        target_var (string) : name of column over which mean will be calculated
        grouping_var (string): name of variable to be collapsed and removed
        from the array
    Returns:
        dataframe_mean (pandas dataframe): dataframe only containing the
        columns in the grouping varibles and the agrregate statistics over the
        target variable
    '''
    dataframe_mean = (dataframe.groupby(grouping_var_list)[target_var]
                      .agg([np.mean,'count','std'])
                      .rename(columns={'mean':target_var+'_mean',
                                       'std': target_var+'_std',
                                       'count':target_var+'_count'})
                      .reset_index())
    return dataframe_mean

def bootstrap_pandas_by_group(group):
    '''
    Can be used to calculate mean and 95% CI over groups during group and aggregate operations in
    pandas.
    '''
    CI = bs.bootstrap(np.array(group),stat_func=bs_stats.mean,iteration_batch_size=10000)
    return (CI.value,abs(CI.lower_bound-CI.value),abs(CI.upper_bound-CI.value))

def azim_error_row(row):
    '''
    Calculates distance between labeled azimuth and predicted azimuth. Check
    distance both directions around circle and returns smaller one.
    '''
    azim = row['azim']
    pred = row['predicted_azim']
    return min(72-(abs(azim-pred)),abs(azim-pred))

def azim_offset_row(row):
    '''
    Calculates distance between reference source and target source. Check
    distance both directions around circle and returns smaller one.
    '''
    target = row['target_azim']
    reference = row['reference_azim']
    normal_distance = target - reference
    target_left_of_reference_counterclockwise = True if target > reference else False
    counterclockwise_distance = 360 - abs(target - reference)
    if abs(normal_distance) < counterclockwise_distance:
        return normal_distance
    else:
        if target_left_of_reference_counterclockwise: return -1*counterclockwise_distance
        return counterclockwise_distance
    return min(72-(abs(azim-pred)),abs(azim-pred))


def getDatafilenames(regex):
    '''
    Finds files that match regex, loads them and builds dataframe from laoded
    data.
    '''
    fnames = glob(regex)
    print("FOUND {} FILES:".format(len(fnames)))
    np_data_list = []
    for fname in fnames:
        #Gets key list with same name as numpy array
        keylist_regex = fname.replace("batch_conditional","keys_test").replace(".npy",".json")
        #This is to filter for the very large outputs broken into
        #different numpy arrays. It maps them all to the same key list.
        keylist_regex = re.sub("_count_\d+_","_",keylist_regex)
        keylist = glob(keylist_regex)
        label_order = []
        if len(keylist) == 1:
            key = keylist[0]
            print("FOUND KEY LIST")
            with open(key) as f:
                data = json.load(f)
                label_order = get_order(regex,data)
                if None in label_order:
                    unused_keys = [key for idx,key in
                                   enumerate(data) if label_order[idx] is None]
                    message = ("Not all values in dictionary used! Unused keys:"
                               "{}".format(str(unused_keys)))
                    warnings.warn(message, UserWarning)
        print(fname)
        temp_array = np.load(fname,allow_pickle=True)
        if "multi_source" in regex:
            reshaped_array = multi_source_reshape_and_filter(temp_array,label_order)
        else:
            reshaped_array = reshape_and_filter(temp_array,label_order)
        np_data_list.append(reshaped_array)
    return np_data_list


def add_fold_offset(row):
    if 'azim' in row:
        if (18 < row['azim'] < 54):
            return abs(row['predicted_folded'] + 18)
        elif row['azim'] == 18 or row['azim'] == 54:
            return row['predicted']
        else:
            return (18 -row['predicted_folded'])%72
    elif 'flipped' in row:
        warnings.warn("Assuming -45 and +45 click positions for folding!")
        return (18 -row['predicted_folded'])%72
    else:
        msg = ("Folding only supported with azimuth column or "
               "for presedence effect mode with flipped column")
        raise NotImplementedError(msg)



def swap_folded_vals(folded_vals,row):
    pdb.set_trace()
    if row['azim'] == 18 or row['azim'] == 54:
        return row['predicted']
    else:
        return folded_vals[row.name] 

def make_dataframe(regex,fold_data=False,elevation_predictions=False,recenter_folded_data=True):
    np_data_array = getDatafilenames(regex)
    cols = get_cols(regex)
    arch_indecies,init_indices = get_arch_indicies(regex)
    cols = ['predicted'] + cols + ['arch_index','init_index']
    if fold_data:
        cols = cols + ['predicted_folded']
    for arch_index,init_index,np_data in zip(arch_indecies,init_indices,np_data_array):
        np_arch_val = np.full((np_data.shape[0],1),arch_index)
        np_init_val = np.full((np_data.shape[0],1),init_index)
        pred = np.array([x[:72] for x in np_data[:,0]]).argmax(axis=1)
        folded_vals = np.array([fold_locations_full_dist_5deg(x[:72]) for x in np_data[:,0]]).argmax(axis=1)
        if elevation_predictions:
            full_pred = np.array([x for x in np_data[:,0]]).argmax(axis=1)
            np_data[:,0] = full_pred
        else:
            np_data[:,0] = pred
        if fold_data:
            folded_vals = np.expand_dims(folded_vals,1)
            np_full_data = np.hstack((np_data,np_arch_val,np_init_val,folded_vals))
        else:
            np_full_data = np.hstack((np_data,np_arch_val,np_init_val))
        df = pd.DataFrame(data=np_full_data,columns=cols)
        main_df = df if 'main_df' not in locals() else pd.concat([main_df,df]).reset_index(drop=True)
    if fold_data and recenter_folded_data:
        main_df['predicted_folded'] = main_df.apply(add_fold_offset, axis=1)
    return main_df

def make_dataframe_full_pred_vector(regex):
    np_data_array = getDatafilenames(regex)
    cols = get_cols(regex)
    arch_indecies,init_indices = get_arch_indicies(regex)
    cols = ['predicted'] + cols + ['arch_index','init_index']
    for arch_index,init_index,np_data in zip(arch_indecies,init_indices,np_data_array):
        np_arch_val = np.full((np_data.shape[0],1),arch_index)
        np_init_val = np.full((np_data.shape[0],1),init_index)
        pred = np.array([x for x in np_data[:,0]])
        np_data[:,0] = pred.tolist()
        np_full_data = np.hstack((np_data,np_arch_val,np_init_val))
        df = pd.DataFrame(data=np_full_data,columns=cols)
        main_df = df if 'main_df' not in locals() else pd.concat([main_df,df]).reset_index(drop=True)
    return main_df


def make_dataframe_multi_source(regex):
    np_data_array = getDatafilenames(regex)
    cols = get_cols(regex)
    arch_indecies,init_indices = get_arch_indicies(regex)
    cols = ['predicted'] + cols + ['arch_index','init_index']
    for arch_index,init_index,np_data in zip(arch_indecies,init_indices,np_data_array):
        np_arch_val = np.full((np_data.shape[0],1),arch_index)
        np_init_val = np.full((np_data.shape[0],1),init_index)
        full_pred = np.array([x for x in np_data[:,0]])
        full_pred_list = full_pred.tolist()
        np_data[:,0] = full_pred.argmax(axis=1)
        np_full_data = np.hstack((np_data,np_arch_val,np_init_val))
        df = pd.DataFrame(data=np_full_data,columns=cols)
        df['predicted'] = full_pred_list
        main_df = df if 'main_df' not in locals() else pd.concat([main_df,df]).reset_index(drop=True)
    return main_df

def dataframe_from_pickle(filename):
    with open(filename, "rb") as f:
        unpickled_list = pickle.load(f)
    if "ILD"  in filename:
        cols = ["Azim","ILD_low_freq","ILD_high_freq"]
    elif "ITD" in filename:
        cols = ["Azim","ITD"]
    else:
        raise ValueError("Filename is not an ITD or ILD value pickle")
    df = pd.DataFrame(data=np.array(unpickled_list),columns=cols)
    return df

def get_arch_indicies(regex):
    fnames = glob(regex)
    arch_index_list = []
    init_index_list = []
    for path in fnames:
        arch_string = [x for x in path.split('/') if "arch_number" in x]
        assert len(arch_string) == 1
        arch_string_split = arch_string[0].split('_')
        arch_index_list.append(int(arch_string_split[2]))
        if "init" in arch_string_split:
            init_index_list.append(int(arch_string_split[4]))
        else:
            init_index_list.append(-1)
    return (arch_index_list,init_index_list)


def get_cols(regex):
    if "joint" in regex:
        var_order = ["azim" , "elev",
                     "ITD", "ILD", "freq"]
    elif "man-added" in regex:
        var_order = ["azim","freq"]
        if "ITD_ILD" in regex:
            var_order = ["azim","elev","freq"]
    elif "middlebrooks_wrightman" in regex:
        var_order = ["azim", "elev", "ILD","ITD","low_cutoff","high_cutoff","subject_num", "noise_idx"]
    elif "CIPIC" in regex:
        var_order = ["azim", "elev", "subject_num", "noise_idx"]
    elif "HRIR_direct" in regex:
        var_order = ["azim", "elev", "noise_idx", "smooth_factor"]
    elif "samTone" in regex:
        var_order = ["carrier_freq", "modulation_freq",
                     "carrier_delay", "modulation_delay",
                     "flipped"]
    elif "transposed" in regex:
        var_order = ["carrier_freq", "modulation_freq",
                     "delay", "flipped"]
    elif "precedence" in regex and "multiAzim" in regex:
        var_order = ["delay","azim","start_sample",
                     "lead_level","lag_level", "flipped"]
    elif "precedence" in regex:
        var_order = ["delay","start_sample",
                     "lead_level","lag_level", "flipped"]
    elif "nsynth" in regex:
        var_order = ["azim","elev","room_geometry", "instrument_source_str", "room_materials", "sample_rate", "qualities", "filename", "instrument_family_str", "pitch", "note_str", "velocity", "instrument", "instrument_family", "note", "instrument_str", "instrument_source", "head_location"]
    elif "bandpass" in regex:
        var_order = ["azim","elev", "bandwidth","center_freq"]
    elif ("binaural_recorded" in regex) and ("speech_label" in regex):
        var_order = ["azim", "elev","speech"]
    elif "binaural_recorded" in regex:
        var_order = ["azim", "elev"]
    elif ("testset" in regex) and ("convolved" in regex):
        var_order = ["azim", "elev"]
    elif ("testset" in regex) and ("convolved" in regex or "multi_source" in
                                   regex):
        var_order = ["azim", "elev"]
    elif "zhong_yost" in regex:
        var_order = ["azim", "elev"]
    elif "wood_bizley" in regex:
        var_order = ["azim","elev","stim_idx","snr"]
    elif ("hebrank_wright" in regex) or ("hebank_wright" in regex):
        var_order = ["azim","elev","low_cutoff","high_cutoff","stim_idx"]
    elif "roffler_butler" in regex:
        var_order = ["azim","elev","bandwidth","center_freq","bin_freq"]
    elif "best_carlile" in regex:
        var_order = ["azim","elev"]
    elif "rahul" in regex:
        var_order = ["azim","elev","sound_id"]
    elif "pavao" in regex:
        var_order = ["azim","elev","stim_name"]
    elif "spectral_modulation" in regex:
        var_order = ["azim","elev","modulation_rate","stim_idx"]
    elif "midline_elevation" in regex:
        var_order = ["azim","elev","bandwidtth","center_freq"]
    else:
        var_order = ["azim","elev"]
    return var_order


def get_order(regex,data):
    var_order = get_cols(regex)
    key_swap_indicies = [next((idx for idx,pos in enumerate(var_order) if pos in key.split('/')), None)
                         for key in data]
    return key_swap_indicies

def reshape_and_filter(temp_array,label_order):
    prob = np.vstack(temp_array[:,0])
    if all(isinstance(x,np.ndarray) for x in temp_array[:,1][0]):
        #This block handles datasets still in tuples becuase vstack
        #processes things incorrectly in all elements are arrays
        metadata = np.array(pd.DataFrame({"meta":temp_array[:,1]})['meta']
                            .apply(lambda x:tuple([m[0] for m in x])).tolist())
    else:
        metadata = np.vstack(temp_array[:,1])
    if len(label_order) != 0:
        target_slice_arr = []
        src_slice_arr = []
        for src_idx,target_idx in enumerate(label_order):
            if target_idx is not None:
                target_slice_arr.append(target_idx)
                src_slice_arr.append(src_idx)
        slice_array = [idx for idx in label_order if idx is not None]
        metadata_new = np.empty((metadata.shape[0],len(src_slice_arr)),
                                dtype=metadata.dtype)
        metadata_new[:,target_slice_arr] = metadata[:,src_slice_arr]
        array = np.array([[prob,*labels] for prob,labels in zip(prob,metadata_new)])
    else:
        array = temp_array
    return array

def multi_source_reshape_and_filter(temp_array,label_order):
    prob = np.vstack(temp_array[:,0])
    if all(isinstance(x,np.ndarray) for x in temp_array[:,1][0]):
        #This deals with testsets metadata still in tuples because vstack
        #dealts with things incorrectly if all elements are arrays
        try:
            metadata = np.array(pd.DataFrame({"meta":temp_array[:,1]})['meta']
                                .apply(lambda x:tuple([m for m in x])).tolist())
        except ValueError as e:
            metadata =  np.apply_along_axis(lambda x :x.tolist(),axis=1,
                                            arr=temp_array[:,1])
            if len(metadata.shape) != 3: raise e

    else:
        metadata = np.vstack(temp_array[:,1])
    #metadata_padded = pad_numpy_arrays_from_tuple(temp_array[:,1],fill_val=-1)
    if len(label_order) != 0:
        target_slice_arr = []
        src_slice_arr = []
        for src_idx,target_idx in enumerate(label_order):
            if target_idx is not None:
                target_slice_arr.append(target_idx)
                src_slice_arr.append(src_idx)
        slice_array = [idx for idx in label_order if idx is not None]
        if len(metadata.shape) ==2:
            metadata_new = np.empty((metadata.shape[0],len(src_slice_arr)),
                                    dtype=metadata.dtype)
        else:
            metadata_new_shape = (metadata.shape[0] , len(src_slice_arr)) + metadata.shape[2:]
            metadata_new = np.empty(metadata_new_shape,dtype=metadata.dtype)
            metadata_new_shape = (metadata.shape[0] , len(src_slice_arr)) + metadata.shape[2:]
            metadata_new = np.empty(metadata_new_shape,dtype=metadata.dtype)
        metadata_new[:,target_slice_arr] = metadata[:,src_slice_arr]
        array = np.empty((prob.shape[0],1+len(src_slice_arr)),object)
        array[:] = [[prob,*labels] for prob,labels in zip(prob,metadata_new)]
    else:
        array = temp_array
    return array


def pad_numpy_arrays_from_tuple(data,fill_val):
    '''
    Function takes numpy array fillted with tuples of numpy arrays. The arrays
    within a column in the tuples, may be ragged and this functtion padds them to the max
    shape within the column using fill_val.

    Parameters:
        data (np-array) : Array (n,) object array filled with tuples.
        fill_val (int/float) : Value used to fill padding in output array.
    Return:
        ret_array (np-array): Array (n,m,len) array where len is the max leghts of the list
        and all data are padded in dimension 2 to the same length.
    '''
    #Stack data in numpy arrays
    temp_vstack = [ np.vstack(row) for row in data]
    final_dtype = temp_vstack[0].dtype

    # Get lengths of each row of data
    lens = np.array([row.shape[-1] for row in temp_vstack])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]
    out_mask = np.stack((mask,mask),axis=1)

    output_shape = (len(temp_vstack),temp_vstack[0].shape[0],mask.shape[-1])

    # Setup output array and put elements from data into masked positions
    out = np.full(output_shape, fill_value=fill_val, dtype=final_dtype)
    for  row_idx in range(len(temp_vstack)):
        out[row_idx,out_mask[row_idx]] = temp_vstack[row_idx].flatten()

    return out

def allbut(*names):
    names = set(names)
    return [item for item in levels if item not in names]

def shift(seq, n):
    n = n % len(seq)
    return seq[n:] + seq[:n]

def count_trues(bool_arr):
    return np.sum(bool_arr)

def shift_np(seq,n):
    n = n % seq.shape[1]
    return np.concatenate((seq[:,n:],seq[:,:n]),axis=1)

def get_folded_label_idx(azim_label):
    azim_degrees = azim_label*10
    if azim_degrees > 270 or azim_degrees < 90:
        reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
    else:
        reversal_point =  azim_degrees
    #folded = fold_locations(averages)
    reversal_idx = int((reversal_point - 90)/10)
    return reversal_idx

def get_folded_label_idx_5deg(azim_label):
    azim_degrees = azim_label*5
    if azim_degrees > 270 or azim_degrees < 90:
        reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
    else:
        reversal_point =  azim_degrees
    #folded = fold_locations(averages)
    reversal_idx = int((reversal_point - 90)/5)
    return reversal_idx

def fold_locations(inputs_cond):
    bottom = inputs_cond[10:27]
    top = inputs_cond[:9][::-1] + inputs_cond[28:][::-1]
    folded = [sum(x) for x in zip(bottom,top)]
    return [inputs_cond[9]] + folded + [inputs_cond[27]]

def fold_locations_full_dist_5deg(inputs_cond):
    if len(inputs_cond.shape) == 1:
        arr_input = inputs_cond
        bottom = arr_input[19:54]
        top = np.concatenate((np.flipud(arr_input[:18]),np.flipud(arr_input[55:])))
        folded = np.add(top,bottom)
        return np.array(([arr_input[18]]+folded.tolist()+[arr_input[54]]))
    arr_input = np.array(inputs_cond)
    bottom = arr_input[:,19:54]
    top = np.concatenate((np.fliplr(arr_input[:,:18]),np.fliplr(arr_input[:,55:])),axis=1)
    folded = np.add(top,bottom)
    return np.column_stack((arr_input[:,18],folded,arr_input[:,54]))

def fold_locations_full_dist(inputs_cond):
    if len(inputs_cond.shape) == 1:
        arr_input = inputs_cond
        bottom = arr_input[10:27]
        top = np.concatenate((np.flipud(arr_input[:9]),np.flipud(arr_input[28:])))
        folded = np.add(top,bottom)
        return np.array(([arr_input[9]]+folded.tolist()+[arr_input[27]]))
    arr_input = np.array(inputs_cond)
    bottom = arr_input[:,10:27]
    top = np.concatenate((np.fliplr(arr_input[:,:9]),np.fliplr(arr_input[:,28:])),axis=1)
    folded = np.add(top,bottom)
    return np.column_stack((arr_input[:,9],folded,arr_input[:,27]))

def calc_CI(distributions,single_entry=False,stat_func=bs_stats.mean,iteration_batch_size=None):
    if single_entry:
        CI = bs.bootstrap(distributions,
                          stat_func=stat_func,iteration_batch_size=iteration_batch_size)
        return (CI.value,abs(CI.lower_bound-CI.value),abs(CI.upper_bound-CI.value))
    means = []
    bottom_error = []
    top_error = []
    for pred in distributions.T:
        CI = bs.bootstrap(pred,
                          stat_func=stat_func,iteration_batch_size=iteration_batch_size)
        means.append(CI.value)
        bottom_error.append(abs(CI.lower_bound-CI.value))
        top_error.append(abs(CI.upper_bound-CI.value))
    return (means,bottom_error,top_error)


def plot_cond_prob_azim(batch_conditionals):
    x_axis = [i for i in range(-180,180,10)]
    fig, ax = plt.subplots(nrows = 6, ncols = 6, figsize=(30,30))
    for azim in range(36):
        a = [i[0][:36]/sum(i[0][:36]) for i in batch_conditionals if i[1] == azim] 
        avergaes = [sum(i)/len(i) for i in zip(*a)]
        shift_num = azim - 18
        np_a = np.array(a)
        centered = shift_np(np_a,shift_num)
        means,bottom_error,top_error = calc_CI(centered)
        azim_degrees = azim*10
        reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
        shifted_reversal_point = reversal_point-azim_degrees
        ax[azim//6][azim%6].errorbar(x_axis,means,yerr=[bottom_error,top_error],marker='o',markersize=2,linestyle='-')
        #ax[azim//6][azim%6].plot(x_axis, centered, marker='o',markersize=2,  linestyle='-')
        ax[azim//6][azim%6].set_title("Sounds at {} degrees".format(azim*10))
        ax[azim//6][azim%6].set_ylabel("Conditional Probability")
        ax[azim//6][azim%6].set_xlabel("Degrees from Correct")
        ax[azim//6][azim%6].set_ylim(0.0,1.0)
        ax[azim//6][azim%6].set_xticks([-135,-90,-45,0,45,90,135])
        ax[azim//6][azim%6].axvline(x=shifted_reversal_point, color='k', linestyle='--')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                        wspace=0.65)

def plot_cond_prob_azim_folded(batch_conditionals):
    x_axis = [i for i in range(90,280,10)]
    fig, ax = plt.subplots(nrows = 6, ncols = 6, figsize=(30,30))
    for azim in range(36):
        a = [i[0][:36]/sum(i[0][:36]) for i in batch_conditionals if i[1] == azim] 
        azim_degrees = azim*10
        if azim_degrees > 270 or azim_degrees < 90:
            reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
        else:
            reversal_point =  azim_degrees
        np_a = np.array(a)
        folded = fold_locations_full_dist(np_a)
        folded_means,bottom_error,top_error = calc_CI(folded)
        ax[azim//6][azim%6].errorbar(x_axis,folded_means,yerr=[bottom_error,top_error],marker='o',markersize=2,linestyle='-')
        ax[azim//6][azim%6].set_title("Sounds at {} degrees".format(azim*10))
        ax[azim//6][azim%6].set_ylabel("Conditional Probability")
        ax[azim//6][azim%6].set_xlabel("Degrees")
        ax[azim//6][azim%6].set_ylim(0.0,1.0)
        ax[azim//6][azim%6].set_xticks([90,135,180,225,270])
        ax[azim//6][azim%6].axvline(x=reversal_point, color='k', linestyle='--')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                        wspace=0.65)

def plot_means_squared_error(batch_conditionals_ordered,freqs,azim_lim=36,bins_5deg=False):
    mse_by_freq = []
    for batch_conditionals in batch_conditionals_ordered:
        mse_by_azim = []
        if type(azim_lim) == list:
            azim_lim_range = max(azim_lim)+1
            azim_idx = azim_lim
        else:
            azim_idx = list(range(azim_lim))
            azim_lim_range = azim_lim
        for azim in range(azim_lim_range):
            if bins_5deg:
                a = [i[0][:71:2] for i in batch_conditionals if i[1] == azim]
            else:
                a = [i[0][:36] for i in batch_conditionals if i[1] == azim] 
            averages = [sum(i)/len(i) for i in zip(*a)]
            azim_degrees = azim*10
            if azim_degrees > 270 or azim_degrees < 90:
                reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
            else:
                reversal_point =  azim_degrees
            #folded = fold_locations(averages)
            np_a = np.array(a)
            folded = fold_locations_full_dist(np_a)
            max_idxs = np.argmax(folded,axis=1)
            folded_map_est  = np.bincount(max_idxs,minlength=19)/len(folded)
            #folded_map_est = fold_locations(map_est.tolist())
            reversal_idx = int((reversal_point - 90)/10)
            value_mult = np.array([abs(value-reversal_idx) for value,prob in enumerate(folded.T)])
            ex_error = np.array([10*value_mult[map_idx] for map_idx in max_idxs])
            val_error = 10*np.sum(folded*value_mult,axis=1)
            mean_error,low_ci,high_ci = calc_CI(ex_error,single_entry=True)
            #expected_error = 10*sum([prob*abs(value-reversal_idx) for value,prob in enumerate(folded)])
            expected_map_error = 10*sum([prob*abs(value-reversal_idx) for value,prob in enumerate(folded_map_est)])
            mse_by_azim.append((mean_error,low_ci,high_ci))
            #mse_by_azim.append(expected_map_error)
        mse_by_freq.append(mse_by_azim)
    transposed = list(map(list,zip(*mse_by_freq)))

    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)
    for line_azim in azim_idx:
        mean = [x[0] for x in transposed[line_azim]]
        bottom_error = [x[1] for x in transposed[line_azim]]
        top_error = [x[2] for x in transposed[line_azim]]
        ax1.set_xscale('log')
        ax1.set_xlim(99,12000)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Mean error (Degrees)")
        #ax1.set_ylim(0,100)
        ax1.errorbar(freqs,mean,yerr=[bottom_error,top_error],marker='o',markersize=3, label = "{} Deg.".format(10*line_azim))
        #ax1.plot(freqs,transposed[line_azim],marker='o',markersize=3, label = line_azim)
    ax1.legend(loc=2)


def make_bandwidth_vs_error_humabn_plot(return_pd=False):
    pd_yost = pd.read_csv("/home/francl/Bandwidth_Data.csv",header=[0])
    #conver bandwidths in fractions to numeric values
    pd_yost['bandwidth'] = pd_yost['bandwidth'].apply(eval)
    #Align Pure tone data with other pure tone points
    #Difference was caused by error in original CSV write
    pd_yost.iloc[7,1] = .001
    pd_yost.iloc[16,1] = .001
    #dummy column needed to add point makers with seaborn
    pd_yost['same_val'] = 1
    pd_yost.columns = ['frequency', 'Bandwidth (Octaves)', 'RMS Error (Degrees)',
                       'standard deviation', 'same_val']
    fig = plt.figure(figsize=(13,11))
    plt.clf()
    sns.lineplot(x='Bandwidth (Octaves)',y='RMS Error (Degrees)',hue="same_val",
                 lw=4,ms=10,legend=False,data=pd_yost,style="same_val",
                 markers=[""],err_style='bars',
                 err_kws={'mew':0,'elinewidth':4,'capthick':0},
                 dashes=False,palette=['k'])
    plt.xticks(rotation=90,size=30)
    plt.yticks(size=30)
    plt.ylim(5,25)
    plt.ylabel("RMS Error (Degrees)",size=40)
    plt.xlabel("Bandwidth (Octaves)",size=40)
    [x.set_color("black") for x in plt.gca().get_lines()]
    plt.tight_layout()
    pd_yost.to_csv(plots_folder+"/"+"yost_human_data_collapsed_bandwidth.csv")
    plt.savefig(plots_folder+"/"+"yost_human_data_collapsed_bandwidth.svg")
    if return_pd:
        return pd_yost



def plot_means_squared_error_by_freq_old(batch_conditionals_ordered,azim_lim=36,labels=None,no_fold=False,bins_5deg=False,collapse_conditions=False):
    mse_by_freq = []
    for batch_conditionals in batch_conditionals_ordered:
        mse_by_azim = []
        for azim in range(azim_lim):
            if bins_5deg:
                a = [i[0][:72] for i in batch_conditionals if i[1] == azim]
            else:
                a = [i[0][:36] for i in batch_conditionals if i[1] == azim]
            averages = [sum(i)/len(i) for i in zip(*a)]
            max_idxs = np.argmax(a,axis=1)
            if bins_5deg:
                map_est  = np.bincount(max_idxs,minlength=72)/len(a)
            else:
                map_est  = np.bincount(max_idxs,minlength=36)/len(a)
            if bins_5deg:
                azim_degrees = azim*5
            else:
                azim_degrees = azim*10
            if azim_degrees > 270 or azim_degrees < 90:
                reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
            else:
                reversal_point =  azim_degrees
            #folded = fold_locations(averages)
            if bins_5deg:
                reversal_idx = int((reversal_point - 90)/5)
            else:
                reversal_idx = int((reversal_point - 90)/10)
            if no_fold:
                folded = np.array(a)
                value_mult = np.array([min(abs(value-azim),abs(36+azim-value))
                                       for value,prob in enumerate(folded.T)])
            else:
                np_a = np.array(a)
                if bins_5deg:
                    folded = fold_locations_full_dist_5deg(np_a)
                else:
                    folded = fold_locations_full_dist(np_a)
                value_mult = np.array([abs(value-reversal_idx) for value,prob in enumerate(folded.T)])
            max_idxs = np.argmax(folded,axis=1)
            #folded_map_est = fold_locations(map_est.tolist())
            if bins_5deg:
                val_error = 5*np.sum(folded*value_mult,axis=1)
                ex_error = 5*value_mult[max_idxs]
            else:
                val_error = 10*np.sum(folded*value_mult,axis=1)
                ex_error = 10*value_mult[map_idx]
            mean_error,low_ci,high_ci = calc_CI(ex_error,single_entry=True)
            #expected_error = 10*sum([prob*abs(value-reversal_idx) for value,prob in enumerate(folded)])
            #expected_map_error = 10*sum([prob*abs(value-reversal_idx) for value,prob in enumerate(folded_map_est)])
            mse_by_azim.append((mean_error,low_ci,high_ci))
        mse_by_freq.append(mse_by_azim)
    fig = plt.figure(figsize=(13,11))
    ax1 = fig.add_subplot(111)
    if bins_5deg:
        #azim_range = [5*x for x in range(-18,19)]
        azim_range = [5*x for x in range(72)]
    else:
        azim_range = [10*x for x in range(36)]
    SEM_mean_array = []
    for line_azim in range(len(batch_conditionals_ordered)):
        mean = [x[0] for x in mse_by_freq[line_azim]]
        bottom_error = [x[1] for x in mse_by_freq[line_azim]]
        top_error = [x[2] for x in mse_by_freq[line_azim]]
        mean2 = [mean[x] for x in range(72)]
        top_error2 = [top_error[x] for x in range(72)]
        bottom_error2 = [bottom_error[x] for x in range(72)]
        pd_data = pd.DataFrame(np_SEM_mean_array.T)
        pd_data["azim"] = azim_range
        pd_out = pd.melt(pd_data,id_vars="azim")
        pd_out.columns = ['azim', 'index', 'error']
        pd_out.to_csv(plots_folder+"/azimuth_vs_error_wood_graph_network.csv")
        #mean2 = [mean[x] for x in range(-18,19)]
        #top_error2 = [top_error[x] for x in range(-18,19)]
        #bottom_error2 = [bottom_error[x] for x in range(-18,19)]
        if collapse_conditions:
            SEM_mean_array.append(mean2)
        else:
            if labels != None:
                ax1.errorbar(azim_range,mean,yerr=[bottom_error,top_error],marker='o',markersize=3, label = labels[line_azim])
            else:
                ax1.errorbar(azim_range,mean,yerr=[bottom_error,top_error],marker='o',markersize=3, label = line_azim)
    if collapse_conditions:
        np_SEM_mean_array = np.array(SEM_mean_array)
        mean = np.mean(np_SEM_mean_array,axis=0)
        SEM = np.std(np_SEM_mean_array,axis=0)/np.sqrt(len(SEM_mean_array))
        top_error = SEM*1.96
        bottom_error = SEM*1.96
        if labels != None:
            ax1.errorbar(azim_range,mean,yerr=[bottom_error,top_error],marker='o',markersize=3, label = labels[0])
        else:
            ax1.errorbar(azim_range,mean,yerr=[bottom_error,top_error],marker='o',markersize=3,color='k')
    #ax1.set_ylabel("Mean error (Degrees,Flipped)",fontsize=40)
    ax1.set_ylabel("Mean error (Degrees)",fontsize=40)
    ax1.set_xlabel("Azimuth (Degrees)",fontsize=40)
    #ax1.set_ylim(0,50)
    #ax1.set_xlim(-90,90)
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    ax1.legend(loc=2,fontsize=20)

def plot_means_squared_error_by_freq(batch_conditionals_ordered,azim_lim=36,
                                     labels=None,no_fold=False,bins_5deg=False,collapse_conditions=False):
    mse_by_freq = []
    for batch_conditionals in batch_conditionals_ordered:
        mse_by_azim = []
        for azim in range(azim_lim):
            if bins_5deg:
                a = [i[0][:72] for i in batch_conditionals if i[1] == azim]
            else:
                a = [i[0][:36] for i in batch_conditionals if i[1] == azim]
            averages = [sum(i)/len(i) for i in zip(*a)]
            max_idxs = np.argmax(a,axis=1)
            if bins_5deg:
                map_est  = np.bincount(max_idxs,minlength=72)/len(a)
            else:
                map_est  = np.bincount(max_idxs,minlength=36)/len(a)
            if bins_5deg:
                azim_degrees = azim*5
            else:
                azim_degrees = azim*10
            if azim_degrees > 270 or azim_degrees < 90:
                reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
            else:
                reversal_point =  azim_degrees
            #folded = fold_locations(averages)
            if bins_5deg:
                reversal_idx = int((reversal_point - 90)/5)
            else:
                reversal_idx = int((reversal_point - 90)/10)
            if no_fold:
                folded = np.array(a)
                value_mult = np.array([min(abs(value-azim),abs(36+azim-value))
                                       for value,prob in enumerate(folded.T)])
            else:
                np_a = np.array(a)
                if bins_5deg:
                    folded = fold_locations_full_dist_5deg(np_a)
                else:
                    folded = fold_locations_full_dist(np_a)
                value_mult = np.array([abs(value-reversal_idx) for value,prob in enumerate(folded.T)])
            max_idxs = np.argmax(folded,axis=1)
            #folded_map_est = fold_locations(map_est.tolist())
            if bins_5deg:
                val_error = 5*np.sum(folded*value_mult,axis=1)
                ex_error = 5*value_mult[max_idxs]
            else:
                val_error = 10*np.sum(folded*value_mult,axis=1)
                ex_error = 10*value_mult[map_idx]
            mean_error,low_ci,high_ci = calc_CI(ex_error,single_entry=True)
            #expected_error = 10*sum([prob*abs(value-reversal_idx) for value,prob in enumerate(folded)])
            #expected_map_error = 10*sum([prob*abs(value-reversal_idx) for value,prob in enumerate(folded_map_est)])
            mse_by_azim.append((mean_error,low_ci,high_ci))
        mse_by_freq.append(mse_by_azim)
    fig = plt.figure(figsize=(13,11))
    ax1 = fig.add_subplot(111)
    if bins_5deg:
        azim_range = [5*x for x in range(-18,19)]
        #azim_range = [5*x for x in range(72)]
    else:
        azim_range = [10*x for x in range(36)]
    SEM_mean_array = []
    for line_azim in range(len(batch_conditionals_ordered)):
        mean = [x[0] for x in mse_by_freq[line_azim]]
        bottom_error = [x[1] for x in mse_by_freq[line_azim]]
        top_error = [x[2] for x in mse_by_freq[line_azim]]
        #mean2 = [mean[x] for x in range(72)]
        #top_error2 = [top_error[x] for x in range(72)]
        #bottom_error2 = [bottom_error[x] for x in range(72)]
        mean2 = [mean[x] for x in range(-18,19)]
        top_error2 = [top_error[x] for x in range(-18,19)]
        bottom_error2 = [bottom_error[x] for x in range(-18,19)]
        if collapse_conditions:
            SEM_mean_array.append(mean2)
        else:
            if labels != None:
                ax1.errorbar(azim_range,mean,yerr=[bottom_error,top_error],marker='o',markersize=3, label = labels[line_azim])
            else:
                ax1.errorbar(azim_range,mean,yerr=[bottom_error,top_error],marker='o',markersize=3, label = line_azim)
    if collapse_conditions:
        np_SEM_mean_array = np.array(SEM_mean_array)
        mean = np.mean(np_SEM_mean_array,axis=0)
        SEM = np.std(np_SEM_mean_array,axis=0)/np.sqrt(len(SEM_mean_array))
        top_error = SEM*1.96
        bottom_error = SEM*1.96
        pd_data = pd.DataFrame(np_SEM_mean_array.T)
        pd_data["azim"] = azim_range
        pd_out = pd.melt(pd_data,id_vars="azim")
        pd_out.columns = ['azim', 'index', 'error']
        pd_out.to_csv(plots_folder+"/azimuth_vs_error_wood_graph_network.csv")
        if labels != None:
            ax1.errorbar(azim_range,mean,yerr=[bottom_error,top_error],marker='o',lw=4,markersize=5,
                         elinewidth=4,color='k',label=labels[0])
        else:
            #ax1.errorbar(azim_range,mean,yerr=[bottom_error,top_error],marker='o',markersize=3)
            ax1.errorbar(azim_range,mean,yerr=[bottom_error,top_error],marker='o',lw=3.5,color='k',
                         markersize=4,elinewidth=3.5)
    ax1.set_ylabel("Mean error (Degrees)",color='k',fontsize=40)
    #ax1.set_ylabel("Mean error (Degrees)",fontsize=40)
    ax1.set_xlabel("Azimuth (Degrees)",fontsize=40)
    ax1.set_ylim(0,50)
    ax1.set_xlim(-90,90)
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    ax1.legend(loc=2,fontsize=20)
    #colormap = plt.cm.gist_ncar
    ax1.legend(loc=2,fontsize=20)
    #colormap = plt.cm.gist_ncar
    #colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
    #for i,j in enumerate(ax1.lines):
    #    j.set_color(colors[i])


def plot_means_squared_error_by_bandwidth(batch_conditionals_ordered,freqs,jitter_distance_octave_fraction,azim_lim=36,labels=None,no_fold=False,bins_5deg=False,collapse_conditions=False):
    if bins_5deg and azim_lim == 36:
        warnings.warn("Only calculating right half of space. Change azim_lim.",
                      RuntimeWarning)
    bandwidth_set = sorted(list(set(np.concatenate(batch_conditionals_ordered)[:,3])))
    fuzzy_mult = 2**(jitter_distance_octave_fraction)-1
    mse_by_arch =[]
    for batch_conditionals in batch_conditionals_ordered:
        mse_by_freq = []
        for freq in freqs:
            print(freq,datetime.datetime.now().time())
            mse_by_bandwidth = []
            for bandwidth in bandwidth_set:
                mse_by_azim = []
                for azim in range(azim_lim):
                    if bins_5deg:
                        a = [i[0][:72] for i in batch_conditionals if i[1] == azim
                             and i[3] == bandwidth and
                             val_match_target_fuzzy(i[4],freq,fuzzy_mult)]
                    else:
                        a = [i[0][:36] for i in batch_conditionals if i[1] == azim
                             and i[3] == bandwidth and 
                             val_match_target_fuzzy(i[4],freq,fuzzy_mult)]
                    averages = [sum(i)/len(i) for i in zip(*a)]
                    max_idxs = np.argmax(a,axis=1)
                    if bins_5deg:
                        map_est  = np.bincount(max_idxs,minlength=72)/len(a)
                    else:
                        map_est  = np.bincount(max_idxs,minlength=36)/len(a)
                    if bins_5deg:
                        azim_degrees = azim*5
                    else:
                        azim_degrees = azim*10
                    if azim_degrees > 270 or azim_degrees < 90:
                        reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
                    else:
                        reversal_point =  azim_degrees
                    #folded = fold_locations(averages)
                    if bins_5deg:
                        reversal_idx = int((reversal_point - 90)/5)
                    else:
                        reversal_idx = int((reversal_point - 90)/10)
                    if no_fold:
                        folded = np.array(a)
                        value_mult = np.array([min(abs(value-azim),abs(36+azim-value))
                                               for value,prob in enumerate(folded.T)])
                    else:
                        np_a = np.array(a)
                        if bins_5deg:
                            folded = fold_locations_full_dist_5deg(np_a)
                        else:
                            folded = fold_locations_full_dist(np_a)
                        value_mult = np.array([abs(value-reversal_idx) for value,prob in enumerate(folded.T)])

                    max_idxs = np.argmax(folded,axis=1)
                    #folded_map_est = fold_locations(map_est.tolist())
                    if bins_5deg:
                        val_error = 5*np.sum(folded*value_mult,axis=1)
                        ex_error = 5*value_mult[max_idxs]
                    else:
                        val_error = 10*np.sum(folded*value_mult,axis=1)
                        ex_error = 10*value_mult[map_idxs]
                    #mean_error,low_ci,high_ci = calc_CI(ex_error,single_entry=True)
                    mean_error = sum(ex_error)/len(ex_error)
                    #expected_error = 10*sum([prob*abs(value-reversal_idx) for value,prob in enumerate(folded)])
                    #expected_map_error = 10*sum([prob*abs(value-reversal_idx) for value,prob in enumerate(folded_map_est)])
                    #mse_by_azim.append((mean_error,low_ci,high_ci))
                    mse_by_azim.append(mean_error)
                mse_by_bandwidth.append(sum(mse_by_azim)/len(mse_by_azim))
            mse_by_freq.append(mse_by_bandwidth)
        mse_by_arch.append(mse_by_freq)
    np_SEM_mean_array= np.array(mse_by_arch).astype(np.float32)
    #Not sure this is still necessary after the sum/len above
    #np_SEM_mean_array = np_mse_by_arch.mean(axis=3)
    pd_data = pd.DataFrame(np_SEM_mean_array.squeeze().T)
    pd_data["bandwidth"] = bandwidth_set
    pd_out = pd.melt(pd_data,id_vars="bandwidth")
    pd_out.columns = ['bandwidth', 'index', 'error']
    pd_out.to_csv(plots_folder+"/bandwidth_vs_error_network.csv")
    mean = np.mean(np_SEM_mean_array,axis=0)
    SEM = np.std(np_SEM_mean_array,axis=0)/np.sqrt(np_SEM_mean_array.shape[0])
    top_error = SEM*1
    bottom_error = SEM*1
    plt.clf()
    fig = plt.figure(figsize=(13,11))
    ax1 = fig.add_subplot(111)
    for f_counter, freq_label in enumerate(freqs):
                ax1.errorbar(bandwidth_set,mean[f_counter],
                             yerr=[bottom_error[f_counter],top_error[f_counter]],
                             marker='', label = freq_label,
                             lw=4,ms=10,capsize=0,elinewidth=4,
                             capthick=0,color='k')
    ax1.set_ylabel("Mean error (Degrees)",fontsize=40)
    ax1.set_xlabel("Bandwidth (Octaves)",fontsize=40)
    ax1.legend(loc=1,fontsize=25)
    plt.xticks(np.arange(0,2.25,0.25), rotation=90,size = 30)
    plt.yticks(size = 30)
    plt.ylim(0,25)
    plt.xlim(-0.1,2.1)
    plt.gca().get_legend().remove()
    plt.tight_layout()
    plt.savefig(plots_folder+"/"+"bandwidth_vs_error_network_plot.png")

def compare_ITDs(batch_condtionals_tone,min_dist=0,max_dist=math.inf,manually_added=True,midline_refernece_version=False):
    #compares all pairs of stimuli and determines if network can identify wich
    #is on the right and on the left(2AFC task simulation)
    batch_results = []
    for batch in batch_condtionals_tone:
        results = []
        for pair in itertools.combinations(batch,2):
            correct = False
            x,y = pair[0],pair[1]
            x_label,y_label = x[1],y[1]
            if not manually_added:
                x_idx,y_idx = get_folded_label_idx(x_label),get_folded_label_idx(y_label)
            else:
                x_idx,y_idx = x_label,y_label
            if midline_refernece_version:
                #Check if either image is on midline. If not, skip pair.
                if x_idx != 0 and y_idx != 0:
                    continue
            label_diff = x_idx - y_idx
            #meant to allow for splitting 2AFC task into different distance
            #ranges to test system sensitivity
            if abs(label_diff) > max_dist or abs(label_diff) < min_dist:
                continue
            x_freq,y_freq = x[2],y[2]
            x_dist_folded,y_dist_folded = fold_locations_full_dist(x[0][:36]),\
                fold_locations_full_dist(y[0][:36])
            x_center,y_center = ndimage.center_of_mass(x_dist_folded)[0],\
                ndimage.center_of_mass(y_dist_folded)[0]
            predicted_diff = x_center-y_center
            if sign(predicted_diff) == sign(label_diff):
                correct = True
            results.append([(x_label,y_label),(x_freq,y_freq),predicted_diff,label_diff,correct])
        batch_results.append(results)
        batch_results = sorted(batch_results, key= lambda x: x[1][1])
    return batch_results

def plot_precednece_effect(np_data_array):
    delays = sorted(list(set(np_data_array[:,1])))  
    #hold levels constant
    c1 = np_data_array[:,3] == -10
    c4 = np_data_array[:,4] == -10
    flipped_list = []
    for flipped in [0,1]:
        c2 = np_data_array[:,5] == flipped
        delays_list = []
        for delay in delays:
            c3 = np_data_array[:,1] == delay
            locs = np.array([x[:72].argmax() for x in np_data_array[c1&c2&c3&c4][:,0]])
            delays_list.append(locs.mean())
        flipped_list.append(delays_list)
    return delays,flipped_list


def plot_precedence_clicks_by_delay(afc_results_array):

    #assert all(plotkey in keys for plotkey in plotted_keys), \
    #        ("Plotted Key not Available! \n Available Keys: {} \
    #         \n Requested Keys: {}".format(keys,plotted_keys))

    fig = plt.figure(figsize=(13,11))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("Delay (ms)",size=40)
    ax1.set_ylabel("Mean Location (Degrees)",size=40)
    ax1.set_ylim(0,360)
    ax1.set_xlim(0,20)
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    ax1.axhline(50,color='k',linestyle='dashed',alpha=0.5)
    #iterates through keys and skips if not in plotted_keys set
    mean_array = []
    for afc_results in afc_results_array:
        #calculates 2AFC results
        delays,mean_choices= \
                plot_precednece_effect(afc_results)
        mean_array.append(mean_choices)
    #calculates SEM over networks
    np_mean_array = np.array(mean_array)
    grand_mean = 5*np.mean(np_mean_array,axis=0)
    SEM = 5*np.std(np_mean_array,axis=0)/np.sqrt(len(mean_array))
    top_error = SEM*1.96
    bottom_error = SEM*1.96
    x_labels = [x for x in delays]
    #ax1.plot(x_labels,mean,marker='o',markersize=3,
    #         label="{} Modulation Freq.".format(float(key)))
    #plots line over networks for single modulation frequency
    for i in range(len(grand_mean)):
        ax1.errorbar(x_labels,grand_mean[i],yerr=[bottom_error[i],top_error[i]],marker='o',markersize=3,
                     label= "{} Flipped".format(i))
    ax1.legend(loc='best')
            
def plot_relative_comparison_acc(results_ILD,sing_freq=False):
    #freqs = [100,200,400,800,1000,1250,1500,1750,2000,3000,6000]
    freqs = [100,200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,11000]
    #freqs = [100,250,500,750,1000,1250,1500,1750,2000,3000,4000,6000]
    if sing_freq:
        freqs = []
    acc_arr = []
    ##change to correct resutls variable
    print("Bootstrapping data...")
    for i,res in enumerate(results_ILD):
        res_np = np.array(res)
        bool_arr = res_np[:,4].astype(np.bool)
        total = res_np.shape[0]
        mean_error,low_ci,high_ci = calc_CI(bool_arr,single_entry=True,
                                        stat_func=bs_stats.sum,iteration_batch_size=100)
        acc = mean_error/total
        low_acc = low_ci/total
        high_acc = high_ci/total
        acc_arr.append([acc,low_acc,high_acc])
        if sing_freq:
            freqs.append(res_np[0,1][0])
    transposed = list(zip(*acc_arr))
    mean = [100*x for x in transposed[0]]
    bottom_error = [100*x for x in transposed[1]]
    top_error = [100*x for x in transposed[2]]
    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)
    ax1.set_xscale('log')
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("2AFC % correct")
    ax1.set_ylim(0,100)
    ax1.errorbar(freqs,mean,yerr=[bottom_error,top_error],marker='o',markersize=3)
    ax1.axhline(50,color='k',linestyle='dashed',alpha=0.5)
    #ax1.plot(freqs,transposed[line_azim],marker='o',markersize=3, label = line_azim)
    ax1.legend(loc=1)

def plot_relative_comparison_acc_multiple_lines(results_ILD_array,freqs=None,collapse_across_conditions=False,sing_freq=False):
    fig = plt.figure(figsize=(13,11))
    ax1 = fig.add_subplot(111)
    ax1.set_xscale('log')
    ax1.set_xlabel("Frequency (Hz)",size=40)
    ax1.set_ylabel("2AFC % correct",size=40)
    ax1.set_ylim(0,100)
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    ax1.axhline(50,color='k',linestyle='dashed',alpha=0.5)
    ax1.legend(loc=1)
    mean_array = []
    top_error_per_net = []
    bottom_error_per_net = []
    #freqs = [100,200,400,800,1000,1250,1500,1750,2000,3000,6000]
    if collapse_across_conditions:
        results_ILD_array = [[sum(x,[]) for x in zip(*results_ILD_array)]]
    for results_ILD in results_ILD_array: 
        freqs = freqs if freqs != None else [100,200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,11000]
        #freqs = [100,250,500,750,1000,1250,1500,1750,2000,3000,4000,6000]
        if sing_freq:
            freqs = []
        acc_arr = []
        ##change to correct resutls variable
        print("Bootstrapping data...")
        for i,res in enumerate(results_ILD):
            res_np = np.array(res)
            bool_arr = res_np[:,4].astype(np.bool)
            total = res_np.shape[0]
            mean_error,low_ci,high_ci = calc_CI(bool_arr,single_entry=True,
                                            stat_func=bs_stats.sum,iteration_batch_size=100)
            acc = mean_error/total
            low_acc = low_ci/total
            high_acc = high_ci/total
            acc_arr.append([acc,low_acc,high_acc])
            if sing_freq:
                freqs.append(res_np[0,1][0])
        transposed = list(zip(*acc_arr))
        mean = [100*x for x in transposed[0]]
        bottom_error = [100*x for x in transposed[1]]
        top_error = [100*x for x in transposed[2]]
        mean_array.append(mean)
        top_error_per_net.append(top_error)
        bottom_error_per_net.append(bottom_error)

    np_mean_array = np.array(mean_array)
    mean = np.mean(np_mean_array,axis=0)
    SEM = np.std(np_mean_array,axis=0)/np.sqrt(len(mean_array))
    top_error = SEM*1.96
    bottom_error = SEM*1.96
    out_dict = {'top_error_per_net':top_error_per_net,
                'bottom_error_per_net':bottom_error_per_net,
                'mean_array':mean_array,'all_nets_mean_array':np_mean_array,
                'all_nets_SEM':SEM}
    ax1.errorbar(freqs,mean,yerr=[bottom_error,top_error],marker='o',markersize=3,alpha=0.6)
    #ax1.plot(freqs,transposed[line_azim],marker='o',markersize=3, label = line_azim)
    return out_dict


def make_yost_localization_by_azim():
    pd_yost =  pd.read_csv("/om/user/francl/Data_Fig_7.csv",header=[0])
    pd_yost_pivoted = pd_yost.set_index('Band-Pass').stack().reset_index()
    pd_yost_pivoted.columns = ["Actual Position (Degrees)",
                               "Predicted Position (Degrees)","Percentage"]
    pd_yost_pivoted = pd_yost_pivoted.convert_objects(convert_numeric=True)
    pd_yost_pivoted["Percentage"] = 100*pd_yost_pivoted["Percentage"]
    pd_yost_pivoted["Predicted Position (Degrees)"] = (pd_yost_pivoted["Predicted Position (Degrees)"]
                                                       .apply(lambda x: (x*15)-105))
    pd_yost_pivoted["Actual Position (Degrees)"] = (pd_yost_pivoted["Actual Position (Degrees)"]
                                                       .apply(lambda x: (x*15)-105))
    pd_yost_distribution = (pd_yost_pivoted.loc[np.repeat(pd_yost_pivoted.index.values,
                                                         pd_yost_pivoted.Percentage)]
                                           .drop("Percentage",axis=1)
                                           .reset_index(drop=True))
    rgb_cmap = get_grayscale_colormap()
    fig = plt.figure(figsize=(14,12.5))
    sns.catplot(kind="violin",x="Actual Position (Degrees)",
                y="Predicted Position (Degrees)", height=8,
                aspect=1.45,inner='quartile',palette=rgb_cmap,
                data=pd_yost_distribution.astype('int32'))
    plt.xlim(-1,11)
    plt.ylim(-105,105)
    plt.yticks([-90,-60,-30,0,30,60,90])
    plt.xticks(list(range(-1,12)),list(range(-90,105,15)))
    plt.savefig(plots_folder+"/yost_frontal_localization_human.svg")
    
    

def format_kulkarni_human_data():
    pd_kulkarni = pd.read_csv("/om/user/francl/kulkarni_colburn_0_azim.csv",header=[0])   
    pd_x_value = [x for x in [1024,512,256,128,64,32,16,8] for i in range(4)]
    pd_kulkarni["Smooth Factor"] = pd_x_value
    plt.clf()
    plt.figure(figsize=(10,10),dpi=200)
    order=[1024,512,256,128,64,32,16,8,4,2,1]
    sns.pointplot(x="Smooth Factor", y="Y",markers=[""],color="k",order=order,data=pd_kulkarni)
    plt.ylim(45,100)
    plt.ylabel("Correct (%)",fontsize=30)
    plt.xlabel("Smooth Factor",fontsize=30)
    plt.yticks(fontsize=30)
    plt.xticks(range(len(order)),order,fontsize=30,rotation=45)
    #plt.text(0.25,0.90,"0 Degrees",fontsize=35,transform=plt.gca().transAxes)
    #plt.axhline(5,color='k',linestyle='dashed',alpha=0.5)
    plt.tight_layout()
    return pd_kulkarni

def format_middlebrooks_human_data():
    pd_middlebrooks = pd.read_csv("/om/user/francl/middlebrooks_datasets.csv",header=[0,1])   
    pd_x_value_ITD = [-600,-300,0,300,600]
    pd_x_value_ILD = [-20,-10,0,10,20]
    cols = ['ILD High', 'ILD Low', 'ITD High', 'ITD Low']
    pd_middlebrooks.loc[:,("ITD Low","X")] = pd_x_value_ITD
    pd_middlebrooks.loc[:,("ITD High","X")] = pd_x_value_ITD
    pd_middlebrooks.loc[:,("ILD High","X")] = pd_x_value_ILD
    pd_middlebrooks.loc[:,("ILD Low","X")] = pd_x_value_ILD
    pd_middlebrooks = pd_middlebrooks[cols]
    return pd_middlebrooks

def format_data_wood_human_data(human_data_csv=None,add_errorbars=False,
                                return_pd=False):
    plt.clf()
    if human_data_csv is None:
        human_data_csv = "/om/user/francl/wood_localization_data.csv"
    pd_wood = pd.read_csv(human_data_csv,header=[0,1])   
    pd_wood_broadband = pd_wood["Broadband Noise"]
    pd_wood_broadband = pd_wood_broadband.sort_values(["X"],ascending=True)
    pd_x_value = [-75,-60,-45,-30,-15,0,15,30,45,60,75]
    pd_wood_broadband["X"] = pd_x_value
    pd_y_col = pd_wood_broadband["Y"]
    if add_errorbars:
        pd_wood_broadband =pd_wood_broadband.reset_index()
        pd_wood_broadband["YErr_top"] = pd_top.sort_values(["X"],ascending=True)["Y"]
        pd_bottom = pd_wood.loc[slice(None),("Broadband Noise Bottom Error",slice(None))]
        pd_bottom.columns = pd_bottom.columns.droplevel(0)
        pd_wood_broadband["YErr_bottom"] = pd_bottom.sort_values(["X"],ascending=True)["Y"]
        errorbar_vector =np.array([abs(pd_wood_broadband["YErr_top"]-pd_wood_broadband["Y"]),
                                  abs(pd_wood_broadband["YErr_bottom"]-pd_wood_broadband["Y"])])
        plt.errorbar(x=pd_wood_broadband["X"],y=pd_wood_broadband["Y"],
                     yerr=errorbar_vector,marker='o',lw=4.4,color='k',
                     markersize=5,elinewidth=4.5)
    else:
        sns.lineplot(x="X",y="Y",data=pd_wood_broadband)
    plt.ylabel("d'",fontsize=20)
    plt.ylim(3.4,0.6)
    plt.xlim(-95,95)
    plt.xlabel("Source Azimuth (Degrees)",fontsize=20)
    plt.yticks([1,2,3],fontsize=15)
    plt.xticks([-90,-45,0,45,90],fontsize=15,rotation=45)
    plt.tight_layout()
    if return_pd:
        return pd_wood_broadband
    #pd_wood_broadband["Y"]=(pd_y_col-pd_y_col.min())/(pd_y_col.max()-pd_y_col.min())
    
def make_wood_network_plot(regex=None):
    if regex is None:
        regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/arch_number_*init_0/"
                 "batch_conditional_broadbandNoiseRecords_wood_convolved_anechoic_"
                 "oldHRIRdist140_stackedCH_upsampled_iter100000*")
    np_data_array_dictionary = getDatafilenames(regex)
    #filter for sounds rendered only at 0 elevation
    np_data_array_dictionary_filtered = [net_preds[net_preds[:,2] == 0] for net_preds in
                                np_data_array_dictionary]
    plt.clf()
    plot_means_squared_error_by_freq(np_data_array_dictionary_filtered,azim_lim=72,
                                     bins_5deg=True,collapse_conditions=True)
    plt.ylim(0,12)
    plt.savefig(plots_folder+"/azimuth_vs_error_wood_graph_network.png")

def get_wood_bizley_snr_threshold_per_architecture(pd_wood_bizley):
    pd_wood_bizley_by_arch = pd_wood_bizley.groupby("arch_index")
    query_trial_count = "azim_folded == -90 or azim_folded == 90"
    query_L_errors = "azim_folded == -90 and predicted_azim_folded >= 0"
    query_R_errors = "azim_folded == 90 and predicted_azim_folded <= 0"
    snr_threshold = []
    for arch_index,df_arch in pd_wood_bizley_by_arch:
        pd_wood_bizley_one_arch_by_snr = df_arch.groupby("snr")
        max_sorted_keys = sorted(pd_wood_bizley_one_arch_by_snr.groups.keys(),reverse=True)
        for snr in max_sorted_keys:
            df_snr = pd_wood_bizley_one_arch_by_snr.get_group(snr)
            L_error_count = df_snr.query(query_L_errors).shape[0]
            R_error_count = df_snr.query(query_R_errors).shape[0]
            trial_count = df_snr.query(query_trial_count).shape[0]
            if float(L_error_count + R_error_count)/trial_count >= .05:
                snr_threshold.append((arch_index,snr))
                break
    return snr_threshold

def get_snr_threshold_trials(pd_wood_bizley,snr_threshold):
    pd_wood_bizley["calibrated_SNR_trial"] = False
    for arch_index,snr in snr_threshold:
        idx = (pd_wood_bizley["arch_index"] == arch_index) & (pd_wood_bizley["snr"] == snr)
        pd_wood_bizley.loc[idx,"calibrated_SNR_trial"] = True
    return pd_wood_bizley


def make_wood_bizley_network_plot(regex=None):
    if regex is None:
        regex = ("/om5/user/francl/grp-om2/gahlm/dataset_pipeline_test/arch_number_*init_0/"
                 "batch_conditional_*_wood_bizley_03092022_iter100000*")
    pd_wood_bizley = make_dataframe(regex,elevation_predictions=True)
    pd_wood_bizley = format_wood_bizley_dataframe(pd_wood_bizley)
    #query breaks with negative numbers in this version of pandas
    snr_threshold = get_wood_bizley_snr_threshold_per_architecture(pd_wood_bizley)
    pd_wood_bizley_filtered = get_snr_threshold_trials(pd_wood_bizley,snr_threshold)
    #snr_val = -8
    #pd_wood_bizley_arch_mean = (pd_wood_bizley.query("snr == @snr_val")
    #                            .groupby(["azim_folded","arch_index","snr"])
    #                            ["azim_error_folded_abs"].mean().reset_index())
    pd_wood_bizley_arch_mean = (pd_wood_bizley_filtered
                                .query("calibrated_SNR_trial == True")
                                .groupby(["azim_folded","arch_index","snr"])
                                ["azim_error_folded_abs"].mean().reset_index())
    plt.clf()
    sns.lineplot(x="azim_folded",y="azim_error_folded_abs",data=pd_wood_bizley_arch_mean,
              ci=68,err_style="bars",color='black',lw=3.1,
              err_kws={'capsize':0.1,'elinewidth':3,'capthick':0.1})
    plt.xticks([-90,-45,0,45,90])
    plt.xlim(-95,95)
    plt.ylim(0,30)
    #sns.lineplot(x="azim_folded",y="azim_error_folded_abs",data=pd_wood_bizley_arch_mean,
    #          units="arch_index",estimator=None)
    pd_wood_bizley = pd_wood_bizley_arch_mean.groupby("azim_folded")["azim_error_folded_abs"].mean().reset_index()
    pd_wood_bizley.columns = ['X', 'Y']
    plt.savefig(plots_folder+"/azimuth_vs_error_wood_graph_network.png")
    return pd_wood_bizley

def format_wood_bizley_dataframe(pd_wood_bizley):
    if pd_wood_bizley['azim'].dtype != 'int64':
        pd_wood_bizley['azim'] = pd_wood_bizley['azim'].apply(convert_from_numpy)
        pd_wood_bizley['elev'] = pd_wood_bizley['elev'].apply(convert_from_numpy)
        pd_wood_bizley = pd_wood_bizley.convert_objects(convert_numeric=True)
    pd_wood_bizley = pd_wood_bizley.query("azim <=18 or azim >= 54")
    pd_wood_bizley['predicted_elev'] = None
    pd_wood_bizley['predicted_elev'] = None
    pd_wood_bizley['predicted_elev'] = pd_wood_bizley['predicted'].apply(lambda x: x//72)
    pd_wood_bizley['predicted_azim'] = pd_wood_bizley['predicted'].apply(lambda x: x%72)
    pd_wood_bizley['predicted_azim_folded'] = pd_wood_bizley['predicted_azim'].apply(CIPIC_azim_folding)
    pd_wood_bizley['azim_folded'] = pd_wood_bizley['azim'].apply(CIPIC_azim_folding)
    pd_wood_bizley['azim_error_folded'] = pd_wood_bizley['azim_folded'] - pd_wood_bizley['predicted_azim_folded']
    pd_wood_bizley['azim_error_folded_abs'] = pd_wood_bizley['azim_error_folded'].abs()
    return pd_wood_bizley

def get_data_from_graph(idx=None,use_points=False):
    lines = plt.gca().get_lines()
    points = plt.gca().findobj(match=matplotlib.collections.PathCollection)
    if use_points and len(points) == 1:
       pd_graph_data = pd.DataFrame(data=points[0].get_offsets(),columns=["X","Y"]) 
       return pd_graph_data
    elif len(lines) == 1 and idx == None:
        x_data = lines[0].get_xdata()
        y_data = lines[0].get_ydata()
    elif isinstance(idx,list):
        idx_list = idx
        x_data = [lines[idx].get_xdata() for idx in idx_list]
        y_data = [lines[idx].get_ydata() for idx in idx_list]
    elif isinstance(idx,int):
        x_data = lines[idx].get_xdata()
        y_data = lines[idx].get_ydata()
    pd_graph_data = pd.DataFrame(data={"X":x_data,"Y":y_data})
    return pd_graph_data

def pd_normalize_column(pd_data,columns=["Y"],columns_out=None,
                        normalizer=preprocessing.MinMaxScaler(),
                        preprocess=lambda x:x, norm_across_cols=False):
    if isinstance(pd_data,np.ndarray):
        return normalizer.fit_transform(pd_data.reshape(-1,1))
    cols_to_normalize = pd_data[columns].apply(preprocess).values
    if norm_across_cols:
        normalizer.fit(cols_to_normalize.reshape(-1,1))
        normalized_columns = normalizer.transform(cols_to_normalize)
    else:
        normalized_columns = normalizer.fit_transform(cols_to_normalize)
    columns_list = columns if isinstance(columns,list) else [columns]
    columns_out = columns_list if columns_out is None else columns_out
    pd_data[columns_out] = pd.DataFrame(normalized_columns,columns=columns_out,
                                    index = pd_data.index)
    return pd_data


def get_stats(pd_human,pd_network,col="Y"):
    pd_human_rank = (pd_human.rank()[col].tolist() if
                     isinstance(pd_human,pd.DataFrame)
                     else stats.rankdata(pd_human))
    pd_network_rank = (pd_network.rank()[col].tolist() if
                     isinstance(pd_network,pd.DataFrame)
                     else stats.rankdata(pd_network))
    pd_human_raw = (pd_human[col] if
                     isinstance(pd_human,pd.DataFrame)
                     else pd_human)
    pd_network_raw = (pd_network[col] if
                     isinstance(pd_network,pd.DataFrame)
                     else pd_network)
    kendall_tau = stats.kendalltau(pd_human_rank,pd_network_rank)
    spearman_r = stats.spearmanr(pd_human_rank,pd_network_rank)
    rmse = math.sqrt(metrics.mean_squared_error(pd_human_raw, pd_network_raw))
    return (kendall_tau,spearman_r,rmse)

def get_wood_correlations(network_regex=None,bootstrap_mode=False):
    format_data_wood_human_data()
    pd_wood_human = get_data_from_graph()
    make_wood_network_plot(network_regex)
    pd_wood_network = get_data_from_graph()
    #Chnage from coordinates (0,355) to (-175,180)
    pd_wood_network["X"] = pd_wood_network["X"].apply(lambda x: x if x <= 180
                                                      else -(360-x))
    pd_network_intersection = pd_wood_network["X"].isin(pd_wood_human["X"])
    pd_wood_network_subset = pd_wood_network[pd_network_intersection]
    if bootstrap_mode:
        return pd_wood_human,pd_wood_network_subset
    #flip network data and normalize
    pd_wood_network_subset["Y"] = pd_wood_network_subset["Y"].apply(lambda x:x*-1)
    pd_wood_network_subset_norm = pd_normalize_column(pd_wood_network_subset)
    pd_wood_human_norm = pd_normalize_column(pd_wood_human)
    (kendall_tau,spearman_r,rmse) = get_stats(pd_wood_human_norm,
                                              pd_wood_network_subset_norm)
    return (kendall_tau,spearman_r,rmse)

def get_wood_bizley_correlations_updated(network_regex=None,bootstrap_mode=False):
    format_data_wood_human_data()
    pd_wood_human = get_data_from_graph()
    pd_wood_network = make_wood_bizley_network_plot(network_regex)
    pd_network_intersection = pd_wood_network["X"].isin(pd_wood_human["X"])
    pd_wood_network_subset = pd_wood_network[pd_network_intersection]
    if bootstrap_mode:
        return pd_wood_human,pd_wood_network_subset
    #flip network data and normalize
    pd_wood_network_subset["Y"] = pd_wood_network_subset["Y"].apply(lambda x:x*-1)
    pd_wood_network_subset_norm = pd_normalize_column(pd_wood_network_subset)
    pd_wood_human_norm = pd_normalize_column(pd_wood_human)
    (kendall_tau,spearman_r,rmse) = get_stats(pd_wood_human_norm,
                                              pd_wood_network_subset_norm)
    return (kendall_tau,spearman_r,rmse)

def make_yost_network_plot(regex=None):
    if regex is None:
        regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/arch_number_*_init_0/"
                 "batch_conditional*bandpass*HRIR*iter100000.npy")
    np_data_array_dictionary = getDatafilenames(regex)
    plot_means_squared_error_by_bandwidth(np_data_array_dictionary,[800],6,bins_5deg=True,azim_lim=72)
    pd_yost_network = get_data_from_graph(idx=0)
    return pd_yost_network


def get_wood_correlations_bootstrapped(network_regex=None,model_choices=None):
    assert model_choices is not None
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_wood_bizley_correlations_updated(network_regex=model,bootstrap_mode=True)
        model_data.append(pd_model)
    pd_wood_human_norm = pd_normalize_column(pd_human)
    rmse_list = []
    for model_idx in model_choices:
        pd_model_sample = pd.concat([model_data[i] for i in model_idx])
        pd_model_mean = pd_model_sample.groupby(pd_model_sample.index).mean()
        pd_model_mean["Y"] = pd_model_mean["Y"].apply(lambda x:x*-1)
        try:
            pd_wood_network_subset_norm = pd_normalize_column(pd_model_mean)
        except:
            pdb.set_trace()
        (kendall_tau,spearman_r,rmse) = get_stats(pd_wood_human_norm,
                                                  pd_wood_network_subset_norm)
        rmse_list.append(rmse)
    return rmse_list

def model_wood(network_regex=None,model_choices=None):
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_wood_correlations(network_regex=model,bootstrap_mode=True)
        model_data.append(pd_model)
    pdb.set_trace()


def get_yost_correlations_bootstrapped(network_regex=None,model_choices=None):
    assert model_choices is not None
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_yost_correlations(network_regex=model,bootstrap_mode=True)
        model_data.append(pd_model)
    rmse_list = []
    for model_idx in model_choices:
        pd_model_sample = pd.concat([model_data[i] for i in model_idx])
        pd_model_mean = pd_model_sample.groupby(pd_model_sample.index).mean()
        pd_model_mean["human_Y"] = pd_human["Y"].values
        pd_data_norm = pd_normalize_column(pd_model_mean,
                                           columns=["Y","human_Y"],
                                           norm_across_cols=True,
                                           columns_out=["Y_norm","human_norm"])
        (kendall_tau,spearman_r,rmse) = get_stats(pd_data_norm["human_norm"],
                                                  pd_data_norm["Y_norm"])
        rmse_list.append(rmse)
    return rmse_list

def model_middlebrooks(network_regex=None,model_choices=None):
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_middlebrooks_correlations(regex=model,bootstrap_mode=True)
        model_data.append((pd_model[0].reset_index(),pd_model[1].reset_index()))
    pdb.set_trace()


def get_middlebrooks_correlations_bootstrapped(network_regex=None,
                                              model_choices=None):
    assert model_choices is not None
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_middlebrooks_correlations(regex=model,bootstrap_mode=True)
        model_data.append((pd_model[0].reset_index(),pd_model[1].reset_index()))
    pd_human_ITD,pd_human_ILD = pd_human
    rmse_list_ITD = []
    rmse_list_ILD = []
    for model_idx in model_choices:
        pd_model_sample_ITD = pd.concat([model_data[i][0] for i in model_idx])
        pd_model_sample_ILD = pd.concat([model_data[i][1] for i in model_idx])
        pd_model_mean_ITD = pd_model_sample_ITD.groupby(pd_model_sample_ITD.index).mean()
        pd_model_mean_ILD = pd_model_sample_ILD.groupby(pd_model_sample_ILD.index).mean()
        try:
            pd_model_mean_ITD['human'] = pd_human_ITD
        except:
            pd_model_sample_ITD = pd.concat([model_data[i][0] for i in range(10)])
            pd_model_mean_ITD = pd_model_sample_ITD.groupby(pd_model_sample_ITD.index).mean()
            pd_model_mean_ITD['human'] = pd_human_ITD
        try:
            pd_model_mean_ILD['human'] = pd_human_ILD
        except:
            pd_model_sample_ILD = pd.concat([model_data[i][1] for i in range(10)])
            pd_model_mean_ILD = pd_model_sample_ILD.groupby(pd_model_sample_ILD.index).mean()
            pd_model_mean_ILD['human'] = pd_human_ILD
        pd_model_ITD_norm = pd_normalize_column(pd_model_mean_ITD,
                                                columns=['Y','human'],
                                                norm_across_cols=True,
                                                columns_out=['Y_norm','human_norm'])
        pd_model_ILD_norm = pd_normalize_column(pd_model_mean_ILD,
                                                columns=['Y','human'],
                                                norm_across_cols=True,
                                                columns_out=['Y_norm','human_norm'])

        (kendall_tau_ITD,spearman_r_ITD,rmse_ITD) = get_stats(pd_model_ITD_norm['human_norm'],
                                                              pd_model_mean_ITD['Y_norm'])
        (kendall_tau_ILD,spearman_r_ILD,rmse_ILD) = get_stats(pd_model_ILD_norm['human_norm'],
                                                              pd_model_ILD_norm['Y_norm'])
        rmse_list_ITD.append(rmse_ITD)
        rmse_list_ILD.append(rmse_ILD)
    return rmse_list_ITD,rmse_list_ILD

def get_kulkarni_correlations_bootstrapped(network_regex=None,
                                           model_choices=None):
    assert model_choices is not None
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_kulkarni_correlations(network_regex=model,
                                                      bootstrap_mode=True)
        model_data.append(pd_model)
    rmse_list = []
    for model_idx in model_choices:
        pd_model_sample = pd.concat([model_data[i] for i in model_idx])
        pd_model_mean = pd_model_sample.groupby(pd_model_sample.index).mean()
        pd_model_norm = pd_normalize_column(pd_model_mean)
        pd_human_norm = pd_normalize_column(pd_human)
        (kendall_tau,spearman_r,rmse) = get_stats(pd_human_norm,
                                                  pd_model_norm)
        rmse_list.append(rmse)
    return rmse_list

def get_van_opstal_correlations_bootstrapped(network_regex=None,
                                             model_choices=None):
    assert model_choices is not None
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_van_opstal_correlations(regex=model,
                                                        bootstrap_mode=True)
        #pd_model[['x','y']] = pd.DataFrame(pd_model[0].tolist(),
        #                                   index=pd_model.index)
        model_data.append(pd_model)
    rmse_list_x = []
    rmse_list_y = []
    for model_idx in model_choices:
        pd_model_sample = pd.concat([model_data[i].reset_index() for i in model_idx])
        pd_model_mean = pd_model_sample.groupby(pd_model_sample.index).mean()
        pd_humans_norm = pd_normalize_column(pd_human,columns=[0,1],
                                             columns_out=['x','y'],
                                            norm_across_cols=True)
        pd_networks_norm = pd_normalize_column(pd_model_mean,
                                               columns=[0,1],
                                               columns_out=['x','y'],
                                              norm_across_cols=True)
        (kendall_tau_x,spearman_r_x,rmse_x) = get_stats(pd_humans_norm,
                                                        pd_networks_norm,
                                                        col="x")
        (kendall_tau_y,spearman_r_y,rmse_y) = get_stats(pd_humans_norm,
                                                        pd_networks_norm,
                                                        col="y")
        rmse_list_x.append(rmse_x)
        rmse_list_y.append(rmse_y)
    return rmse_list_x,rmse_list_y


def get_litovsky_correlations_bootstrapped(network_regex=None,
                                           model_choices=None):
    assert model_choices is not None
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_litovsky_correlations(network_regex=model,
                                                        bootstrap_mode=True)
        model_data.append(pd_model)
    rmse_list = []
    pdb.set_trace()
    for model_idx in model_choices:
        pd_model_sample = pd.concat([model_data[i] for i in model_idx])
        pd_model_mean = pd_model_sample.groupby(pd_model_sample.index).mean()


        pd_model_mean_vector = \
                pd_model_mean[["Lead_error","Lag_error"]].values.ravel(order='F')
        (kendall_tau,spearman_r,rmse) = get_stats(pd_human,
                                                  pd_model_mean_vector)
        rmse_list.append(rmse)
    return rmse_list

def get_litovsky_correlations_bootstrapped_multi_azim(network_regex=None,
                                                      model_choices=None):
    assert model_choices is not None
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_litovsky_correlations_multi_azim(network_regex=model,
                                                                bootstrap_mode=True)
        model_data.append(pd_model)
    rmse_list = []
    for model_idx in model_choices:
        pd_model_sample = pd.concat([model_data[i] for i in model_idx])
        pd_model_mean = pd_model_sample.groupby(pd_model_sample.index).mean()


        pd_model_mean_vector = \
                pd_model_mean[["Lead_error","Lag_error"]].values.ravel(order='F')
        pd_model_mean_norm = pd_normalize_column(pd_model_mean_vector)
        pd_human_norm = pd_normalize_column(pd_human)
        (kendall_tau,spearman_r,rmse) = get_stats(pd_human_norm,
                                                  pd_model_mean_norm)
        rmse_list.append(rmse)
    return rmse_list

@lru_cache(maxsize=30)
def bootstrap_plots(regex_folder,random_seed=0,num_iterations=10000,
                    model_num=None,iter_num=100000):
    np.random.seed(random_seed)
    if isinstance(num_iterations,int) and model_num is None:
        bootstrapped_choices = np.random.choice(10,(num_iterations,10))
    elif model_num is not None:
        bootstrapped_choices = np.array([[x]*model_num for x in range(model_num)])
        if num_iterations is not None:
            warnings.warn("Ignoring num_iterations because model_num"
                          "is not None")
    else:
        raise ValueError("Either num_iterations (int) or bootstrapped_choices"
                         "(list) msut be set")

    wood_regex = (regex_folder+"arch_number_*init_0/batch_conditional_*"
                  "_wood_bizley_03092022_iter{}*".format(iter_num))
    yost_regex = (regex_folder+"arch_number_*init_0/batch_conditional*bandpass"
                  "*HRIR*iter{}.npy".format(iter_num))
    middlebrooks_regex = (regex_folder+"arch_number_*init_0/batch_conditional_"
                          "*middlebrooks_wrightman*999_side_aware_normalized_iter{}*".format(iter_num))
    kulkarni_regex = (regex_folder+"arch_number_*init_0/batch_conditional_"
                      "broadbandNoiseRecords_convolved_smooth*HRIR_direct_"
                      "stackedCH_upsampled_iter{}.npy".format(iter_num))
    van_opstal_regex = (regex_folder+"arch_number_[0-9][0,1,3-9][0-9]_init_0/"
                        "batch_conditional_broadbandNoiseRecords_"
                        "convolvedCIPIC*iter{}.npy".format(iter_num))
    litovsky_regex = (regex_folder+"arch_number_*init_0/batch_conditional_*"
                      "precedence*multiAzim*pinknoise*5degStep*iter{}*".format(iter_num))
    hebrank_wright_regex = (regex_folder+"arch_number_*_init_0/batch_conditional_"
                            "noiseRecords_hebrank_wright_iter{}.npy".format(iter_num))
    print("Starting Wood")
    rmse_wood = get_wood_correlations_bootstrapped(network_regex=wood_regex,
                                                   model_choices=bootstrapped_choices)
    print("Starting Yost")
    rmse_yost = get_yost_correlations_bootstrapped(network_regex=yost_regex,
                                                   model_choices=bootstrapped_choices)
    print("Starting Middlebrooks")
    rmse_middlebrooks_ITD,rmse_middlebrooks_ILD = get_middlebrooks_correlations_bootstrapped(
                                                   network_regex=middlebrooks_regex,
                                                   model_choices=bootstrapped_choices)
    print("Starting Kulkarni")
    rmse_kulkarni = get_kulkarni_correlations_bootstrapped(network_regex=kulkarni_regex,
                                                   model_choices=bootstrapped_choices)
    print("Starting Van Opstal")
    rmse_van_opstal_x,rmse_van_opstal_y = get_van_opstal_correlations_bootstrapped(
                                                   network_regex=van_opstal_regex,
                                                   model_choices=bootstrapped_choices)
    print("Starting Litovsky")
    rmse_litovsky = get_litovsky_correlations_bootstrapped_multi_azim(
                                                   network_regex=litovsky_regex,
                                                   model_choices=bootstrapped_choices)
    print("Starting Hebrank and Wright")
    rmse_hebrank_wright = get_hebrank_wright_correlations_bootstrapped(
                                                   network_regex=hebrank_wright_regex,
                                                   model_choices=bootstrapped_choices)

    van_opstal_added = [sum(x) for x in zip(rmse_van_opstal_x,rmse_van_opstal_y)]
    output_dict = {'wood':rmse_wood,'yost':rmse_yost,'ITD':rmse_middlebrooks_ITD,
                   'ILD':rmse_middlebrooks_ILD,'kulkarni':rmse_kulkarni,
                   'van_opstal':van_opstal_added,'litovsky':rmse_litovsky,
                   'hebrank_wright':rmse_hebrank_wright}
    return output_dict
    

def get_bootstapped_rank(*training_conditions_dicts,graphing_mode=False,
                         return_exp_order=False):
    for training_condition in training_conditions_dicts:
        if 'van_opstal_x' in training_condition:
            training_condition['van_opstal'] = [sum(x) for x in
                                                zip(training_condition['van_opstal_x'],
                                                    training_condition['van_opstal_y'])]
            del training_condition['van_opstal_x']
            del training_condition['van_opstal_y']
        assert training_condition.keys() == training_conditions_dicts[0].keys()
    rank_dict = {}
    raw_dict = {}
    all_conditions_list = []
    all_conditions_raw = []
    col_order = []
    for key in training_conditions_dicts[0].keys():
        print(key)
        col_order.append(key)
        psychophysics_test = [condtion[key] for condtion in training_conditions_dicts]
        np_psychophysics_test = np.array(psychophysics_test).T
        psychophysics_ranked = np.apply_along_axis(stats.rankdata,axis=1,
                                                   arr=np_psychophysics_test)
        rank_dict[key] = psychophysics_ranked
        raw_dict[key] = np_psychophysics_test
        all_conditions_list.append(psychophysics_ranked)
        all_conditions_raw.append(np_psychophysics_test)
    np_all_conditions = np.array(all_conditions_list)
    np_raw_conditions = np.array(all_conditions_raw)
    np_mean_condtions = np.mean(np_all_conditions,axis=0).squeeze()
    np_mean_raw_condtions = np.mean(np_raw_conditions,axis=0).squeeze()
    if graphing_mode:
        if return_exp_order:
            return col_order,np_raw_conditions
        return np_raw_conditions
    np_mean_sort = np.sort(np_mean_condtions,axis=0)
    num_samples = np_mean_sort.shape[0]
    low_ci = np_mean_sort[int((num_samples*.025)),:]
    high_ci = np_mean_sort[int((num_samples*.975)),:]
    median =  np_mean_sort[int((num_samples*.5)),:]
    mean = np.mean(np_mean_condtions,axis=0)
    pdb.set_trace()
    p_values = fit_gaussian(np_mean_sort[:,0].reshape(-1,1),
                            np_mean_sort[int((num_samples*.5)),1:].reshape(-1,1))
    return (mean,median,low_ci,high_ci,p_values)

def get_individual_error_graph(regex_folder_list,iter_nums=100000,condtion_names=None,model_num=10):
    iter_nums = [iter_nums]*len(regex_folder) if isinstance(iter_nums,int) else iter_nums
    bootstrap_data_list = []
    models_dict_list = []
    condtion_names = condtion_names if condtion_names is not None else regex_folder_list
    for iter_num,regex_folder in zip(iter_nums,regex_folder_list):
        if iter_num == 100000:
            bootstrap_dict = bootstrap_plots(regex_folder,random_seed=0,num_iterations=10000)
            models_dict = bootstrap_plots(regex_folder,random_seed=0,num_iterations=None,
                                          model_num=model_num)
        else:
            bootstrap_dict = bootstrap_plots(regex_folder,random_seed=0,iter_num=iter_num,num_iterations=10000)
            models_dict = bootstrap_plots(regex_folder,random_seed=0,iter_num=iter_num,num_iterations=None,
                                      model_num=model_num)
        bootstrap_data_list.append(bootstrap_dict)
        models_dict_list.append(models_dict)
    np_bootstraps = get_bootstapped_rank(*bootstrap_data_list,
                                              graphing_mode=True)
    np_mean_bootstraps = np_bootstraps.mean(axis=0)
    pd_raw = pd.DataFrame(data=np_mean_bootstraps,columns=condtion_names)
    pd_raw_error = pd_raw.melt(value_vars=pd_raw.columns,
                               var_name='Training Condition',
                               value_name="Network-Human Error")
    plt.clf()
    plt.yticks([0.0,0.05,0.1,0.15,0.2,0.25,0.3])
    plt.ylim(0,0.34)
    new_palette = ["#97B4DE","#785EF0","#DC267F","#FE6100"]
    sns.barplot(x="Training Condition",y="Network-Human Error",ci='sd',
                palette=sns.color_palette(new_palette),data=pd_raw_error)
    pd_raw_error.to_csv(plots_folder+"/mean_error_by_training_condition.csv")
    name = plots_folder+"/mean_error_by_training_condition.svg"
    #plt.yticks([0.0,0.05,0.1,0.15,0.2,0.25])
    plt.tight_layout()
    plt.savefig(name)
    cols_models = []
    models_raw = []
    models_mean_list = []
    for models_dict in models_dict_list:
        models_mean_list.append(np.array([*models_dict.values()]).mean(axis=0))
    np_models = np.array(models_mean_list)
    np_models_reformatted = np_models.T
    pd_models = pd.DataFrame(data=np_models_reformatted,columns=condtion_names)
    pd_models.reset_index(level=0, inplace=True)
    pd_models_error = pd_models.melt(id_vars=['index'],value_vars=condtion_names,
                               var_name='Training Condition',
                               value_name="Network-Human Error")
    idx_lookup = {x.get_text():idx for idx,x in zip(*plt.xticks())}
    pd_models_error['Training Condition'] = pd_models_error['Training Condition'].apply(lambda x:
                                                                   idx_lookup[x])
    plt.clf()
    pd_models_error.to_csv(plots_folder+"/individual_model_error_by_training_condition.csv")
    sns.lineplot(x='Training Condition',y="Network-Human Error",
                units="index",estimator=None, lw=2.5,ms=5,
                data=pd_models_error,style="index",
                markers=["o","o","o","o","o","o","o","o","o","o"],
                dashes=False,alpha=0.6)
    plt.gca().get_legend().remove()
    plt.ylim(0,.41)
    plt.yticks([0,0.1,0.2,0.3,0.4])
    plt.tight_layout()
    name = plots_folder+"/individual_model_error_by_training_condition.svg"
    plt.savefig(name)

    
def get_error_graph_by_task(regex_folder_list,condtion_names=None,iter_nums=100000,model_num=10):
    bootstrap_iterations=10000
    random_seed=0
    iter_nums = [iter_nums]*len(regex_folder) if isinstance(iter_nums,int) else iter_nums
    bootstrap_data_list = []
    models_dict_list = []
    condtion_names = condtion_names if condtion_names is not None else regex_folder_list
    for iter_num,regex_folder in zip(iter_nums,regex_folder_list):
        if iter_num == 100000:
            bootstrap_dict = bootstrap_plots(regex_folder,random_seed=random_seed,num_iterations=bootstrap_iterations)
            models_dict = bootstrap_plots(regex_folder,random_seed=random_seed,num_iterations=None,
                                          model_num=model_num)
        else:
            bootstrap_dict = bootstrap_plots(regex_folder,random_seed=random_seed,iter_num=iter_num,num_iterations=bootstrap_iterations)
            models_dict = bootstrap_plots(regex_folder,random_seed=random_seed,iter_num=iter_num,num_iterations=None,
                                          model_num=model_num)
        bootstrap_data_list.append(bootstrap_dict)
        models_dict_list.append(models_dict)
    col_order,np_bootstraps = get_bootstapped_rank(*bootstrap_data_list,
                                         graphing_mode=True,
                                        return_exp_order=True)
    m,n,r = np_bootstraps.shape
    np_pd_formatting = np.column_stack((np.repeat(np.arange(m),n),np_bootstraps.reshape(m*n,-1)))
    condtion_names_exp = ['Experiment IDX']+condtion_names
    pd_raw = pd.DataFrame(data=np_pd_formatting,columns=condtion_names_exp)
    pd_raw_error = pd_raw.melt(id_vars=['Experiment IDX'],value_vars=condtion_names,
                               var_name='Training Condition',
                               value_name="Network-Human Error")
    plt.clf()
    exp_lookup = {i:x for i,x in enumerate(col_order)}
    pd_raw_error['Experiment'] = pd_raw_error['Experiment IDX'].apply(lambda x:
                                                                      exp_lookup[x])
    cols_models = []
    models_raw = []
    models_mean_list = []
    for models_dict in models_dict_list:
        models_mean_list.append(np.array([*models_dict.values()]))
    np_models = np.array(models_mean_list)
    np_reformatted = np.transpose(np_models,(1,2,0))
    m,n,r = np_reformatted.shape
    np_pd_formatting = np.column_stack((np.repeat(np.arange(m),n),np.tile(np.arange(n),m),np_reformatted.reshape(m*n,-1)))
    condtion_names_exp = ['Experiment IDX','model_index']+condtion_names
    pd_models = pd.DataFrame(data=np_pd_formatting,columns=condtion_names_exp)
    pd_models_error = pd_models.melt(id_vars=['Experiment IDX','model_index'],
                                     value_vars=condtion_names,
                                     var_name='Training Condition',
                                     value_name="Network-Human Error")
    idx_lookup = {x:i for i,x in enumerate(condtion_names)}
    pd_models_error['Experiment'] = pd_models_error['Experiment IDX'].apply(lambda x:
                                                                      exp_lookup[x])
    pd_models_error['Training Condition'] = pd_models_error['Training Condition'].apply(lambda x:
                                                                   idx_lookup[x])
    np.random.seed(random_seed)
    def std_ddof(x): return np.std(x,ddof=1)
    idxs = [[choices+offset for offset in range(0,(10*len(col_order)),10)]
                for choices in np.random.choice(10,(bootstrap_iterations,10))]
    np_idxs= np.array(idxs).reshape(bootstrap_iterations,-1)
    pd_stats_list = []
    pd_stats_diff_list = []
    for i,np_idx in enumerate(np_idxs):
        pd_subset = pd_models.iloc[np_idx].groupby(['Experiment IDX']).agg([np.mean,std_ddof])
        #pd_diff_subset = (pd_models.iloc[np_idx].apply(get_diff_by_model,axis=1)
        #                  .groupby(['Experiment IDX']).agg([np.mean,std_ddof]))
        pd_stats = pd_subset.apply(cohens_d_by_model,axis=1)
        pd_stats_list.append(pd_stats.loc[:,(slice(None),'cohens_d')])
        if i%500==0: print(i)
    pd_stats_all = pd.concat(pd_stats_list)
    pd_stats_melted = pd_stats_all.loc[:,(slice(None),'cohens_d')].stack(level=[0]).reset_index()
    pd_stats_melted["Experiment"] = pd_stats_melted['Experiment IDX'].apply(lambda
                                                                            x:exp_lookup[x])
    pd_stats_melted.rename(columns={'level_1':'Training Condition'},inplace=True)
    colors = sns.color_palette('colorblind')
    del colors[4]
    del colors[4]
    colors=[colors[2],colors[3],colors[0],colors[1],colors[5],colors[4],colors[7],colors[6]]
    hue_order = ['ITD', 'ILD', 'wood', 'yost', 'van_opstal',
                 'kulkarni','hebrank_wright','litovsky']
    plt.figure(figsize=(10,8))
    sns.barplot(x='Training Condition',y='cohens_d',hue='Experiment',
                hue_order=hue_order,ci='sd',data=pd_stats_melted,palette=colors)
    plt.xticks([0,1,2],["Anechoic","No Background", "Unnatural"])
    plt.ylabel("Cohen's D")
    plt.ylim(-7,7)
    plt.gca().get_legend().remove()
    plt.tight_layout()
    pd_stats_melted.to_csv(plots_folder+"/cohens_d_by_training_cond_and_experiment_sem.csv")
    plt.savefig(plots_folder+"/cohens_d_by_training_cond_and_experiment_sem.svg")
    plt.clf()
    plt.figure(figsize=(10,8))
    pdb.set_trace()
    for exp in col_order:
        plt.clf()
        pd_models_error_subset= pd_models_error.query("Experiment == @exp")
        pd_raw_error_subset= pd_raw_error.query("Experiment == @exp")
        sns.barplot(x="Training Condition",y="Network-Human Error",ci='sd',
                                 order=condtion_names,data=pd_raw_error_subset)
        pdb.set_trace()
        sns.lineplot(x='Training Condition',y="Network-Human Error",
                    units="model_index",estimator=None, lw=2,ms=4,
                    data=pd_models_error_subset,style="model_index",
                    markers=["o","o","o","o","o","o","o","o","o","o"],
                    dashes=False,alpha=0.4)
        plt.gca().get_legend().remove()
        #plt.title("Network-Human Error for {} Experiment".format(exp))
        plt.tight_layout()
        name = plots_folder+("/network_human_error_by_training_cond"
                             "_experriment_{}.svg".format(exp))
        plt.savefig(name)


def cohens_d_by_model(pd_series):
    for key in pd_series.index.get_level_values(0).unique():
        if "model_index" in key: continue
        if "Normal" in key: continue
        u_cond = pd_series.loc[key,'mean']
        o_cond = pd_series.loc[key,'std_ddof']
        u_norm = pd_series.loc['Normal','mean']
        o_norm = pd_series.loc['Normal','std_ddof']
        diff = u_cond-u_norm
        pooled_o = math.sqrt((o_cond**2+o_norm**2)/2.0)        
        pd_series.loc[key,'cohens_d'] = diff/pooled_o
    return pd_series

def cohens_d_diff_by_model(pd_series):
    for key in pd_series.index.get_level_values(0).unique():
        if "diff" not in key: continue
        if "model_index" in key: continue
        if "IDX" in key: continue
        if "Normal" in key: continue
        diff = pd_series.loc[key,'mean']
        o_cond = pd_series.loc[key,'std_ddof']
        pd_series.loc[key,'cohens_d'] = diff/o_cond
    return pd_series

def get_diff_by_model(pd_series):
    for key in pd_series.index.get_level_values(0).unique():
        if "model_index" in key: continue
        if "Normal" in key: continue
        u_cond = pd_series.loc[key]
        u_norm = pd_series.loc['Normal']
        diff = u_cond-u_norm
        pd_series.loc[key+'_diff'] = diff
    return pd_series

def pd_cohens_d(pd_error,condition_names):
    std_ddof = lambda x : np.std(x,ddof=1)
    pd_stats = (pd_error.groupby('Training Condition')['Network-Human Error']
                .agg([('mean',np.mean),('std_ddof',std_ddof)]))
    for key,condition in pd_stats.iterrows():
        if "Normal" in key: continue
        u_cond = condition['mean']
        o_cond = condition['std_ddof']
        u_norm = pd_stats.loc['Normal','mean']
        o_norm = pd_stats.loc['Normal','std_ddof']
        diff = u_cond-u_norm
        pooled_o = math.sqrt((o_cond**2+o_norm**2)/2.0)        
        pd_stats.loc[key,'cohens_d'] = diff/pooled_o
    return pd_stats

    
def fit_gaussian(bootstrap_data_null,sample_values):
    #Fits a 1D histogram to the gaussian theen calculates
    #P(X > samle_value_i| bootstrap_data_null) from that gaussian
    hist, bin_edges = np.histogram(bootstrap_data_null,bins='auto',
                                      density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    p0 = [1., 0., 1.]
    coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
    print(coeff)
    p_values = 1- stats.norm.cdf(sample_values,loc=coeff[1],scale=coeff[2])
    return p_values


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

class ChainedAssignment:
    def __init__(self, chained=None):
        acceptable = [None, 'warn', 'raise']
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained
    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self
    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw

def fit_gauss_to_wood(bs_size=10000,return_dist=False):
   pd_wood_gaussian = format_data_wood_human_data("/om/user/francl/wood_localization_errorbar.csv",
                                               add_errorbars=True,return_pd=True)
   #Origianl dta was SEM so the width should be 2x SD of mean
   pd_wood_gaussian["SD"] = (pd_wood_gaussian["YErr_top"] - pd_wood_gaussian["YErr_bottom"])/2.0
   mean = pd_wood_gaussian["Y"]
   cov = np.diag(pd_wood_gaussian["SD"].pow(2))
   samples = np.random.multivariate_normal(mean,cov,size=bs_size)
   samples_pd_wide = pd.DataFrame(data=samples,columns=pd_wood_gaussian["X"])
   samples_pd = samples_pd_wide.stack(level=0).reset_index()
   samples_pd.rename({"level_0":"idx",0:"Y"},axis=1,inplace=True)
   pd_wood_human_norm = pd_normalize_column(pd_wood_gaussian)
   rmse_list = []
   with ChainedAssignment():
       for idx,sample_mean in samples_pd.groupby(["idx"]):
           pd_sample_norm = pd_normalize_column(sample_mean)
           (kendall_tau,spearman_r,rmse) = get_stats(pd_wood_human_norm,pd_sample_norm)
           rmse_list.append(rmse)
           if idx%500 ==0:print(idx)
   if return_dist:
       return rmse_list
   rmse_list.sort()
   bootstrapped_mean = sum(rmse_list)/len(rmse_list)
   ci_5,ci_95 = rmse_list[int(bs_size*.05)],rmse_list[int(bs_size*.95)]
   return ci_5,bootstrapped_mean,ci_95
    
def combine_gaussians(pd_series):
    #This calculates a new gaussian distribution that
    #represents the mean of the previous random variables.
    var_list = []
    mean_list = []
    keys = pd_series.index.get_level_values(0).unique()
    for key in keys:
        mean_list.append(pd_series.loc[key,'RMS Error (Degrees)'])
        var_list.append(pd_series.loc[key,'var'])
    grouped_var = (1/len(keys)**2) * sum(var_list)
    grouped_mean = (1/len(keys)) * sum(mean_list)
    pd_series.loc[keys,'grouped_var'] = grouped_var
    pd_series.loc[keys,'grouped_mean'] = grouped_mean
    return pd_series
    
def fit_gauss_to_broadband_yost(bs_size=10000,return_dist=False):

   pd_yost = make_bandwidth_vs_error_humabn_plot(return_pd=True)
   pd_yost_mean = (pd_yost.groupby("Bandwidth (Octaves)")["RMS Error (Degrees)"]
                          .agg([np.mean,'std'])
                          .reset_index())
   pd_yost_mean.rename({"mean":"Y"},axis=1,inplace=True)
   pd_yost["var"] = pd_yost["standard deviation"].pow(2)
   pd_yost = pd_yost.groupby("Bandwidth (Octaves)").apply(combine_gaussians)
   pd_yost_grouped = pd_yost.groupby("Bandwidth (Octaves)").mean().reset_index()
   mean = pd_yost_grouped["grouped_mean"]
   cov = np.diag(pd_yost_grouped["grouped_var"])
   samples = np.random.multivariate_normal(mean,cov,size=bs_size)
   samples_pd_wide = pd.DataFrame(data=samples,
                                  columns=pd_yost_grouped["Bandwidth (Octaves)"])
   samples_pd = samples_pd_wide.stack(level=0).reset_index()
   samples_pd.rename({"level_0":"idx",0:"Y"},axis=1,inplace=True)
   rmse_list = []
   with ChainedAssignment():
       for idx,sample_mean in samples_pd.groupby(["idx"]):
           sample_mean["human_Y"] = pd_yost_mean["Y"].values
           pd_data_norm = pd_normalize_column(sample_mean,
                                           columns=["Y","human_Y"],
                                           norm_across_cols=True,
                                           columns_out=["Y_norm","human_norm"])
           (kendall_tau,spearman_r,rmse) = get_stats(pd_data_norm["Y_norm"],
                                                     pd_data_norm["human_norm"])
           rmse_list.append(rmse)
           if idx%500 ==0:print(idx)
   if return_dist:
       return rmse_list
   rmse_list.sort()
   bootstrapped_mean = sum(rmse_list)/len(rmse_list)
   ci_5,ci_95 = rmse_list[int(bs_size*.05)],rmse_list[int(bs_size*.95)]
   return ci_5,bootstrapped_mean,ci_95

def fit_gauss_to_van_opstal(bs_size=10000,return_dist=False):
   pd_means_humans_before,pd_ref_human_before = get_van_opstal_human_plot('before')
   pd_means_humans_after,pd_ref_human_after = get_van_opstal_human_plot('after')
   pd_human_before_diff = pd_ref_human_before.apply(pd.Series) - pd_means_humans_before['xy'].apply(pd.Series)
   pd_human_after_diff = pd_ref_human_after.apply(pd.Series) - pd_means_humans_after['xy'].apply(pd.Series)
   pd_humans = pd.concat([pd_human_before_diff.abs().reset_index(drop=True),
                          pd_human_after_diff.abs().reset_index(drop=True)])
   pd_before,pd_after = human_data_sample_helper_van_opstal(pd_means_humans_before,
                                                            pd_ref_human_before,
                                                            pd_means_humans_after,
                                                            pd_ref_human_after)
   rmse_list = []
   with ChainedAssignment():
       for before,after in zip(pd_before.groupby(["idx"]),
                                  pd_after.groupby(["idx"])):
           idx = before[0]
           sample_mean_before = before[1]
           sample_mean_after = after[1]
           pd_sample_before_diff = (pd_ref_human_before.apply(pd.Series) - 
                                     sample_mean_before['xy'].apply(pd.Series).reset_index(drop=True))
           pd_sample_after_diff = (pd_ref_human_after.apply(pd.Series) - 
                                    sample_mean_after['xy'].apply(pd.Series).reset_index(drop=True))
           pd_sample = pd.concat([pd_sample_before_diff.abs().reset_index(drop=True),
                                  pd_sample_after_diff.abs().reset_index(drop=True)])
           pd_humans_norm = pd_normalize_column(pd_humans,columns=[0,1],
                                                columns_out=['x','y'],
                                               norm_across_cols=True)
           pd_sample_norm = pd_normalize_column(pd_sample,
                                                  columns=[0,1],
                                                  columns_out=['x','y'],
                                                 norm_across_cols=True)
           stats_x = get_stats(pd_humans_norm,pd_sample_norm,
                                      col="x")
           stats_y = get_stats(pd_humans_norm,pd_sample_norm,
                                  col="y")
           rmse_list.append(stats_x[2]+stats_y[2])
           if idx%500 ==0:print(idx)
   if return_dist:
       return rmse_list
   rmse_list.sort()
   bootstrapped_mean = sum(rmse_list)/len(rmse_list)
   ci_5,ci_95 = rmse_list[int(bs_size*.05)],rmse_list[int(bs_size*.95)]
   return ci_5,bootstrapped_mean,ci_95

def human_data_sample_helper_van_opstal(pd_means_humans_before,pd_ref_human_before,
                                        pd_means_humans_after,pd_ref_human_afer,bs_size=10000):
    #converting 95% CI
    pd_means_humans_before["var_y"] = pd_means_humans_before["yerr"].apply(lambda x: (1/1.96*sum(x)/len(x))**2)
    pd_means_humans_before["var_x"] = pd_means_humans_before["xerr"].apply(lambda x: (1/1.96*sum(x)/len(x))**2)
    pd_means_humans_after["var_y"] = pd_means_humans_after["yerr"].apply(lambda x: (1/1.96*sum(x)/len(x))**2)
    pd_means_humans_after["var_x"] = pd_means_humans_after["xerr"].apply(lambda x: (1/1.96*sum(x)/len(x))**2)
    before_mean_y = pd_means_humans_before["y"]
    before_mean_x = pd_means_humans_before["x"]
    after_mean_y = pd_means_humans_after["y"]
    after_mean_x = pd_means_humans_after["x"]
    before_cov_x = np.diag(pd_means_humans_before["var_x"])
    before_cov_y = np.diag(pd_means_humans_before["var_y"])
    after_cov_x = np.diag(pd_means_humans_after["var_x"])
    after_cov_y = np.diag(pd_means_humans_after["var_y"])

    #
    before_samples_x = np.random.multivariate_normal(before_mean_x,
                                                     before_cov_x,size=bs_size)
    pd_before_x = pd.DataFrame(data=before_samples_x,
                                   columns=pd_means_humans_before.index.values)
    pd_before_x = pd_before_x.stack(level=0).reset_index()
    pd_before_x.rename({"level_0":"idx",0:"X"},axis=1,inplace=True)

    #
    before_samples_y = np.random.multivariate_normal(before_mean_y,
                                                     before_cov_y,size=bs_size)
    pd_before_y = pd.DataFrame(data=before_samples_y,
                                   columns=pd_means_humans_before.index.values)
    pd_before_y = pd_before_y.stack(level=0).reset_index()
    pd_before_y.rename({"level_0":"idx",0:"Y"},axis=1,inplace=True)
    pd_before = pd_before_x.merge(pd_before_y,on=["idx","level_1"])
    pd_before["xy"] = pd_before.apply(lambda row: (row["X"],row["Y"]),axis=1)

    after_samples_x = np.random.multivariate_normal(after_mean_x,
                                                     after_cov_x,size=bs_size)
    pd_after_x = pd.DataFrame(data=after_samples_x,
                                   columns=pd_means_humans_after.index.values)
    pd_after_x = pd_after_x.stack(level=0).reset_index()
    pd_after_x.rename({"level_0":"idx",0:"X"},axis=1,inplace=True)
    after_samples_y = np.random.multivariate_normal(after_mean_y,
                                                     after_cov_y,size=bs_size)
    pd_after_y = pd.DataFrame(data=after_samples_y,
                                   columns=pd_means_humans_after.index.values)
    pd_after_y = pd_after_y.stack(level=0).reset_index()
    pd_after_y.rename({"level_0":"idx",0:"Y"},axis=1,inplace=True)
    pd_after = pd_after_x.merge(pd_after_y,on=["idx","level_1"])
    pd_after["xy"] = pd_after.apply(lambda row: (row["X"],row["Y"]),axis=1)
    return pd_before,pd_after

def fit_gauss_to_kulkarni(bs_size=10000,X_subset=[256,128,64,32,16,8],return_dist=False):
   pd_kulkarni = (format_kulkarni_human_data().groupby(["Smooth Factor"])
                                              .agg([np.mean,'std'])
                                              .reset_index())
   #Chnage from coordinates (0,355) to (-175,180)
   pd_kulkarni.columns = ['_'.join(col) for col in pd_kulkarni.columns]
   pd_kulkarni.rename({'X_mean':'X','Y_mean':'Y','Y_std':'SD'},axis=1,inplace=True)
   pd_kulkarni["X"] = [8,16,32,64,128,256,512,1024]
   pd_kulkarni = pd_kulkarni.query("X in @X_subset")
   pd_kulkarni["var"] = pd_kulkarni["SD"].pow(2)
   mean = pd_kulkarni["Y"]
   cov = np.diag(pd_kulkarni["SD"])
   samples = np.random.multivariate_normal(mean,cov,size=bs_size)
   samples_pd_wide = pd.DataFrame(data=samples, columns=pd_kulkarni["X"])
   samples_pd = samples_pd_wide.stack(level=0).reset_index()
   samples_pd.rename({"level_0":"idx",0:"Y"},axis=1,inplace=True)
   pd_kulkarni_human_norm = pd_normalize_column(pd_kulkarni)
   #flip network data and normalize
   rmse_list = []
   with ChainedAssignment():
       for idx,sample_mean in samples_pd.groupby(["idx"]):
           pd_kulkarni_samplew_norm = pd_normalize_column(sample_mean)
           (kendall_tau,spearman_r,rmse) = get_stats(pd_kulkarni_human_norm,
                                                     pd_kulkarni_samplew_norm)
           rmse_list.append(rmse)
           if idx%500 ==0:print(idx)
   if return_dist:
       return rmse_list
   rmse_list.sort()
   bootstrapped_mean = sum(rmse_list)/len(rmse_list)
   ci_5,ci_95 = rmse_list[int(bs_size*.05)],rmse_list[int(bs_size*.95)]
   return ci_5,bootstrapped_mean,ci_95


def fit_gauss_to_litovsky(bs_size=10000,return_dist=False):
   pd_litovsky_human = get_litovsky_human_data(add_errorbars=True)
   pd_litovsky_adult_mean = pd_litovsky_human.loc[:,(("Lead Click Adult","Lag Click Adult"),"Y")].values.ravel(order='F')
   pd_litovsky_X_vals = pd_litovsky_human.loc[:,(("Lead Click Adult","Lag Click Adult"),"X")].values.ravel(order="F")
   pd_litovsky_adult_sd = pd_litovsky_human.loc[:,(("Lead Click Adult Top","Lag Click Adult Top"),"Y")].values.ravel(order='F') - pd_litovsky_adult_mean
   pd_litovsky_adult_var = (pd_litovsky_adult_sd**2)
   cov = np.diag(pd_litovsky_adult_var)
   samples = np.random.multivariate_normal(pd_litovsky_adult_mean,cov,size=bs_size)
   samples_pd_wide = pd.DataFrame(data=samples, columns=pd_litovsky_X_vals)
   samples_pd = samples_pd_wide.stack(level=0).reset_index()
   samples_pd.rename({"level_0":"idx",0:"Y"},axis=1,inplace=True)
   pd_litovsky_human_norm = pd_normalize_column(pd_litovsky_adult_mean)
   rmse_list = []
   with ChainedAssignment():
       for idx,sample_mean in samples_pd.groupby(["idx"]):
           pd_litovsky_samplew_norm = pd_normalize_column(sample_mean)
           (kendall_tau,spearman_r,rmse) = get_stats(pd_litovsky_human_norm,
                                                     pd_litovsky_samplew_norm)
           rmse_list.append(rmse)
           if idx%500 ==0:print(idx)
   if return_dist:
       return rmse_list
   rmse_list.sort()
   bootstrapped_mean = sum(rmse_list)/len(rmse_list)
   ci_5,ci_95 = rmse_list[int(bs_size*.05)],rmse_list[int(bs_size*.95)]
   return ci_5,bootstrapped_mean,ci_95

def get_gauss_rmse_floor_estimate():
    bs_size=10000
    rmse_wood = fit_gauss_to_wood(return_dist=True)
    rmse_yost = fit_gauss_to_broadband_yost(return_dist=True)
    rmse_van_opstal = fit_gauss_to_van_opstal(return_dist=True)
    rmse_kulkarni = fit_gauss_to_kulkarni(return_dist=True)
    rmse_litovsky = fit_gauss_to_litovsky(return_dist=True)
    rmse_total = [sum(res)/5 for res in zip(rmse_wood,rmse_yost,rmse_van_opstal,
                                          rmse_kulkarni,rmse_litovsky)]

    rmse_total.sort()
    return rmse_total

def get_mean_error_graph_subset(regex_folder_list,iter_nums=100000,condtion_names=None,model_num=10,
                                experiment_subset=None):
    iter_nums = [iter_nums]*len(regex_folder) if isinstance(iter_nums,int) else iter_nums
    bootstrap_data_list = []
    models_dict_list = []
    condtion_names = condtion_names if condtion_names is not None else regex_folder_list
    for iter_num,regex_folder in zip(iter_nums,regex_folder_list):
        if iter_num == 100000:
            bootstrap_dict = bootstrap_plots(regex_folder,random_seed=0,num_iterations=10000)
            models_dict = bootstrap_plots(regex_folder,random_seed=0,num_iterations=None,
                                          model_num=model_num)
        else:
            bootstrap_dict = bootstrap_plots(regex_folder,random_seed=0,iter_num=iter_num,num_iterations=10000)
            models_dict = bootstrap_plots(regex_folder,random_seed=0,iter_num=iter_num,num_iterations=None,
                                      model_num=model_num)
        if experiment_subset:
            filtered_bootstrap_dict = {key: bootstrap_dict[key] for key in experiment_subset}
            filtered_models_dict = {key: models_dict[key] for key in experiment_subset}
        else:
            filtered_bootstrap_dict = bootstrap_dict
            filtered_models_dict = models_dict
        bootstrap_data_list.append(filtered_bootstrap_dict)
        models_dict_list.append(filtered_models_dict)
    np_bootstraps = get_bootstapped_rank(*bootstrap_data_list,
                                              graphing_mode=True)
    np_mean_bootstraps = np_bootstraps.mean(axis=0)
    pd_raw = pd.DataFrame(data=np_mean_bootstraps,columns=condtion_names)
    human_rmse = get_gauss_rmse_floor_estimate()
    pd_rmse_human = pd.DataFrame(data=np.array(human_rmse),columns=["Network-Human Error"])
    pd_rmse_human["Training Condition"] = "Between Human"

    plt.clf()
    pd_raw_error = pd_raw.melt(value_vars=pd_raw.columns,
                               var_name='Training Condition',
                               value_name="Network-Human Error")
    pd_raw_error = pd.concat([pd_raw_error,pd_rmse_human])
    new_palette = ["#97B4DE","#785EF0","#DC267F","#FE6100"]
    sns.barplot(x="Training Condition",y="Network-Human Error",ci='sd',
                palette=sns.color_palette(new_palette),data=pd_raw_error)
    name = plots_folder+"/mean_error_by_training_condition_for_gaussian_subset.png"
    plt.ylim(0,0.29)
    plt.yticks([0.0,0.05,0.1,0.15,0.2,0.25])
    plt.tight_layout()
    plt.savefig(name,dpi=400)

def get_bootstapped_mean(*training_conditions_dicts):
    for training_condition in training_conditions_dicts:
        assert training_condition.keys() == training_conditions_dicts[0].keys()
    rank_dict = {}
    for key in training_conditions_dicts[0].keys():
        psychophysics_test = [condtion[key] for condtion in training_conditions_dicts]
        np_psychophysics_test = np.array(psychophysics_test).T
        psychophysics_ranked = np.apply_along_axis(stats.rankdata,axis=1,
                                                   arr=np_psychophysics_test)
        rank_dict[key] = psychophysics_ranked
    all_conditions_list = []
    all_conditions_list = [rank_cond for key,rank_cond in rank_dict.items() if
                          key not in ['ITD','ILD']]
    np_all_conditions = np.array(all_conditions_list)
    pdb.set_trace()
    np_mean_condtions = np.mean(np_all_conditions,axis=0).squeeze()
    np_mean_sort = np.sort(np_mean_condtions,axis=0)
    low_ci = np_mean_sort[250,:]
    high_ci = np_mean_sort[9750,:]
    median =  np_mean_sort[5000,:]
    mean = np.mean(np_mean_condtions,axis=0)
    return (mean,median,low_ci,high_ci)

def get_middlebrooks_slopes(pd_model,regression=LinearRegression()):
    x = pd_model.loc[:4,"X"].values.reshape(-1,1)
    y = pd_model.loc[:4,"Y"].values.reshape(-1,1)
    slope_1 = regression.fit(x,y).coef_[0][0]
    x = pd_model.loc[5:,"X"].values.reshape(-1,1)
    y = pd_model.loc[5:,"Y"].values.reshape(-1,1)
    slope_2 = regression.fit(x,y).coef_[0][0]
    return slope_1,slope_2

def get_middlebrooks_slope_diffs_bootstrapped(network_regex=None,
                                              model_choices=None):
    assert model_choices is not None
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_middlebrooks_correlations(regex=model,bootstrap_mode=True)
        model_data.append((pd_model[0].reset_index(),pd_model[1].reset_index()))
    pd_human_ITD,pd_human_ILD = pd_human
    diffs_list = []
    for model_idx in model_choices:
        pd_model_sample_ITD = pd.concat([model_data[i][0] for i in model_idx])
        pd_model_sample_ILD = pd.concat([model_data[i][1] for i in model_idx])
        pd_model_mean_ITD = pd_model_sample_ITD.groupby(pd_model_sample_ITD.index).mean()
        pd_model_mean_ILD = pd_model_sample_ILD.groupby(pd_model_sample_ILD.index).mean()
        slope_1_ITD,slope_2_ITD = get_middlebrooks_slopes(pd_model_mean_ITD)
        slope_1_ILD,slope_2_ILD = get_middlebrooks_slopes(pd_model_mean_ILD)
        diff_ILD = slope_1_ILD - slope_2_ILD
        diff_ITD = slope_1_ITD - slope_2_ITD
        diff_of_diffs = diff_ILD - diff_ITD
        diffs_list.append(diff_of_diffs)
    p_val = fit_gaussian(diffs_list,0)
    return diffs_list,p_val


def get_middlebrooks_slope_dist_bootstrapped(network_regex=None,
                                              model_choices=None):
    assert model_choices is not None
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_middlebrooks_correlations(regex=model,bootstrap_mode=True)
        model_data.append((pd_model[0].reset_index(),pd_model[1].reset_index()))
    pd_human_ITD,pd_human_ILD = pd_human
    slopes_list = []
    for model_idx in model_choices:
        pd_model_sample_ITD = pd.concat([model_data[i][0] for i in model_idx])
        pd_model_sample_ILD = pd.concat([model_data[i][1] for i in model_idx])
        pd_model_mean_ITD = pd_model_sample_ITD.groupby(pd_model_sample_ITD.index).mean()
        pd_model_mean_ILD = pd_model_sample_ILD.groupby(pd_model_sample_ILD.index).mean()
        slope_1_ITD,slope_2_ITD = get_middlebrooks_slopes(pd_model_mean_ITD)
        slope_1_ILD,slope_2_ILD = get_middlebrooks_slopes(pd_model_mean_ILD)
        slopes_list.append((slope_1_ITD,slope_2_ITD,slope_1_ILD,slope_2_ILD))
    return slopes_list

def format_dataframe_num_sources(pd_num_sources):
    pd_num_sources['reference_elev'] = None
    pd_num_sources['reference_azim'] = None
    pd_num_sources['target_elev'] = None
    pd_num_sources['target_azim'] = None
    pd_num_sources['elev'] = pd_num_sources['elev'].apply(lambda x: [val//2 for val in x])
    pd_num_sources['reference_elev'] = pd_num_sources['elev'].apply(lambda x: x[0])
    pd_num_sources['reference_azim'] = pd_num_sources['azim'].apply(lambda x: x[0])
    pd_num_sources['target_elev'] = pd_num_sources['elev'].apply(lambda x: x[1])
    pd_num_sources['target_azim'] = pd_num_sources['azim'].apply(lambda x: x[1])
    pd_num_sources['predicted_num_sources'] = pd_num_sources['predicted'].apply(lambda x: np.argmax(x)+1)
    pd_num_sources['num_sources'] = pd_num_sources['azim'].apply(lambda x: sum(x>=0))
    return pd_num_sources

def format_dataframe_zhong_yost(pd_zhong_yost):
    pd_zhong_yost['predicted_np'] = pd_zhong_yost['predicted'].apply(lambda x: np.array(x))
    pd_zhong_yost['num_sources'] = pd_zhong_yost['azim'].apply(lambda x: sum(x>=0))
    pd_zhong_yost['pred_locs_top_8'] = (pd_zhong_yost['predicted']
                                        .apply(lambda x: np.array(x[:72]).argsort()[-1:-9:-1]))
                                                
    pd_zhong_yost['top_locs_prob'] =\
            pd_zhong_yost.apply(lambda x: [x['predicted'][idx] for idx in x['pred_locs_top_8']],axis=1)
    bins = sorted([x for x in set(itertools.chain.from_iterable(pd_zhong_yost['azim'])) if x >= 0])
    bin_mask = np.array(shift([x for x in bins for _ in range(72//len(bins))],2))

    pd_zhong_yost['grouped_prob'] = (pd_zhong_yost['predicted_np']
                                     .apply(lambda x : [sum(x[:72][bin_mask == i]) for i in bins]))
    pd_zhong_yost['digitized_top_8_pred'] = (pd_zhong_yost['pred_locs_top_8']
                                          .apply(lambda x: [bins[np.abs(bins-pos).argmin()] for pos in x]))
    #set cutoff by lower cutoff until 95 CI for predicted number of sources for one source contains one.
    #Mean arcross archs by stim.
    pd_zhong_yost['pred_sources'] =\
        pd_zhong_yost.apply(lambda x : set([idx for idx,prob in 
                                            zip(x['digitized_top_8_pred'],x['top_locs_prob']) 
                                            if prob/max(max(x['top_locs_prob']),1e-10) > 0.09]), axis=1)
    fold_dict = {0:0,6:6,12:12,18:18,24:12,30:6,36:0,42:66,48:60,54:54,60:60,66:66}
    pd_zhong_yost['pred_sources_folded'] = (pd_zhong_yost['pred_sources']
                                            .apply(lambda x: set([fold_dict[pos] for pos in x])))
    pd_zhong_yost['pred_num_sources_folded'] = pd_zhong_yost['pred_sources_folded'].apply(lambda x : len(x))
    pd_zhong_yost['pred_num_sources'] = pd_zhong_yost['pred_sources'].apply(lambda x : len(x))
    pd_zhong_yost['num_correct_preds'] = (pd_zhong_yost
                                          .apply(lambda x: len([pos for pos in x['pred_sources']
                                                                if pos in x['azim']]),axis=1))
    pd_zhong_yost['percent_correct_pred'] = (pd_zhong_yost
                                             .apply(lambda x: x['num_correct_preds']/x['num_sources']
                                                    ,axis=1))
    pd_zhong_yost['pred_azims'] =\
            pd_zhong_yost.apply(lambda x: x['digitized_top_8_pred'][:x['pred_num_sources']],axis=1)

def make_zhong_yost_plots(pd_zhong_yost,prefix=""):
    plt.clf()
    pd_zhong_yost_arch_mean = pd_zhong_yost.groupby(["arch_index","num_sources"]).mean().reset_index()
    pd_zhong_yost_stats = pd_zhong_yost_arch_mean.groupby(["num_sources"]).agg([np.mean,'std','sem']).reset_index()
    x_axis = [x for x in range(1,9)]
    y_num_sources = pd_zhong_yost_stats['pred_num_sources']['mean'].tolist()
    y_std_num_sources = pd_zhong_yost_stats['pred_num_sources']['std'].tolist()
    y_prop_corr = pd_zhong_yost_stats['percent_correct_pred']['mean'].tolist()
    y_std_prop_corr = pd_zhong_yost_stats['percent_correct_pred']['std'].tolist()
    sns.lineplot(data=pd_zhong_yost_arch_mean,x="num_sources",y="pred_num_sources",ci=None,color='k')
    plt.errorbar(x=x_axis,y=y_num_sources,yerr=y_std_num_sources,lw=0,elinewidth=1,capsize=0,ecolor='k')
    plt.plot([0, 8], [0, 8],ls="--",color='k',lw=0.5)
    plt.xticks([0,1,2,3,4,5,6,7,8],fontsize=10)
    plt.yticks([0,1,2,3,4,5,6,7,8],fontsize=10)
    for i in range(0,10):
        plt.axhline(i,color='k',alpha=0.5)
    plt.ylim(0,8)
    plt.xlim(0,8.5)
    plt.ylabel("Reported Number of Sources",fontsize=10)
    plt.xlabel("Actual Number of Sources",fontsize=10)
    pd_zhong_yost_stats.to_csv(plots_folder+"/"+prefix+"num_sources_vs_pred_num_sources_zhong_yost.csv")
    plt.savefig(plots_folder+"/"+prefix+"num_sources_vs_pred_num_sources_zhong_yost.svg")
    plt.clf()
    sns.lineplot(data=pd_zhong_yost_arch_mean,x="num_sources",y="percent_correct_pred",ci=None,color='k')
    plt.errorbar(x=x_axis,y=y_prop_corr,yerr=y_std_prop_corr,lw=0,elinewidth=1,capsize=0,ecolor='k')
    plt.xticks([0,1,2,3,4,5,6,7,8],fontsize=10)
    plt.yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0],fontsize=10)
    for i in [x/10.0 for x in range(0,10)]:
        plt.axhline(i,color='k',alpha=0.5)
    plt.ylim(0,1.0)
    plt.xlim(0,8.5)
    plt.ylabel("Proportion of Correct Responses",fontsize=10)
    plt.xlabel("Actual Number of Sources",fontsize=10)
    pd_zhong_yost_arch_mean.to_csv(plots_folder+"/"+prefix+"num_sources_vs_percent_correct_pred_zhong_yost.csv")
    plt.savefig(plots_folder+"/"+prefix+"num_sources_vs_percent_correct_pred_zhong_yost.svg")

def get_zhong_yost_human_data():
    pd_zhong_yost_num_sources = pd.read_csv("/om/user/francl/zhong_yost_num_sources.csv",header=[0,1])   
    pd_zhong_yost_pred_locs = pd.read_csv("/om/user/francl/zhong_yost_correct_pred_locations.csv",header=[0,1])   
    pd_x_value = [1,2,3,4,5,6,7,8]
    pd_zhong_yost_num_sources.loc[:,("Num Sources","X")] = pd_x_value
    pd_zhong_yost_num_sources.loc[:,("Top Error Bar","X")] = pd_x_value
    pd_zhong_yost_num_sources.loc[:,("Bottom Error Bar","X")] = pd_x_value
    pd_zhong_yost_pred_locs.loc[:,("Correct Responses","X")] = pd_x_value
    pd_zhong_yost_pred_locs.loc[:,("Top Error Bar","X")] = pd_x_value
    pd_zhong_yost_pred_locs.loc[:,("Bottom Error Bar","X")] = pd_x_value
    return pd_zhong_yost_num_sources,pd_zhong_yost_pred_locs

def make_zhong_yost_human_plots():
    pd_num_sources,pd_pred_locs = get_zhong_yost_human_data()
    plt.clf()
    pd_num_sources_line = pd_num_sources.loc[:,"Num Sources"]
    pd_num_sources_top = pd_num_sources.loc[:,"Top Error Bar"]
    pd_num_sources_bottom = pd_num_sources.loc[:,"Bottom Error Bar"]
    pd_pred_locs_line = pd_pred_locs.loc[:,"Correct Responses"]
    pd_pred_locs_top = pd_pred_locs.loc[:,"Top Error Bar"]
    pd_pred_locs_bottom = pd_pred_locs.loc[:,"Bottom Error Bar"]
    errorbar_vector_num_sources =np.array([abs(pd_num_sources_top["Y"]-pd_num_sources_line["Y"]),
                                  abs(pd_num_sources_bottom["Y"]-pd_num_sources_line["Y"])])
    errorbar_vector_pred_locs =np.array([abs(pd_pred_locs_top["Y"]-pd_pred_locs_line["Y"]),
                                  abs(pd_pred_locs_bottom["Y"]-pd_pred_locs_line["Y"])])
    sns.lineplot(data=pd_num_sources_line,x="X",y="Y",ci=None,color='k')
    plt.errorbar(x=pd_num_sources_line["X"],y=pd_num_sources_line["Y"],yerr=errorbar_vector_num_sources,
                 lw=0,elinewidth=1,capsize=0,ecolor='k')
    plt.plot([0, 8], [0, 8],ls="--",color='k',lw=0.5)
    plt.xticks([0,1,2,3,4,5,6,7,8],fontsize=10)
    plt.yticks([0,1,2,3,4,5,6,7,8],fontsize=10)
    for i in range(0,10):
        plt.axhline(i,color='k',alpha=0.5)
    plt.ylim(0,8)
    plt.xlim(0,8.5)
    plt.ylabel("Reported Number of Sources",fontsize=10)
    plt.xlabel("Actual Number of Sources",fontsize=10)
    plt.savefig(plots_folder+"/zhong_yost_num_sources_human_results.svg")
    plt.clf()
    sns.lineplot(data=pd_pred_locs_line,x="X",y="Y",ci=None,color='k')
    plt.errorbar(x=pd_pred_locs_line["X"],y=pd_pred_locs_line["Y"],yerr=errorbar_vector_pred_locs,
                 lw=0,elinewidth=1,capsize=0,ecolor='k')
    plt.xticks([0,1,2,3,4,5,6,7,8],fontsize=10)
    plt.yticks([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0],fontsize=10)
    for i in [x/10.0 for x in range(0,10)]:
        plt.axhline(i,color='k',alpha=0.5)
    plt.ylim(0,1.0)
    plt.xlim(0,8.5)
    plt.ylabel("Proportion of Correct Responses",fontsize=10)
    plt.xlabel("Actual Number of Sources",fontsize=10)
    plt.savefig(plots_folder+"/zhong_yost_correct_pred_locs_human_results.svg")


def get_validation_set_accuracy_num_sources(pd_num_sources):
    corr = (pd_num_sources['predicted_num_sources'] ==
            pd_num_sources['num_sources']).sum()
    total =  pd_num_sources.shape[0] 
    acc = corr / total
    return acc

def get_accuracies_num_sources(regex):
    fnames = sorted(glob(regex))
    for path in fnames:
        name = path.split("init_0/")[1]
        print(name)
        pd_num_sources = make_dataframe_multi_source(path)
        pd_num_sources_formatted = format_dataframe_num_sources(pd_num_sources)
        counts = pd_num_sources_formatted['predicted_num_sources'].value_counts()
        acc = get_validation_set_accuracy_num_sources(pd_num_sources_formatted)
        print("Accuracy:",acc)
        print("Counts:",counts)

def format_hebrank_wright_dataframe(pd_hebrank_wright):
    pd_hebrank_wright = pd_hebrank_wright.convert_objects(convert_numeric=True)
    if pd_hebrank_wright['azim'].dtype != 'int64':
        pd_hebrank_wright['azim'] = pd_hebrank_wright['azim'].apply(convert_from_numpy)
        pd_hebrank_wright['elev'] = pd_hebrank_wright['elev'].apply(convert_from_numpy)
    #Get azim and elevation based on predicted index
    pd_hebrank_wright['predicted_elev'] = None
    pd_hebrank_wright['predicted_azim'] = None
    pd_hebrank_wright['predicted_elev'] = pd_hebrank_wright['predicted'].apply(lambda x: 10*(x//72))
    pd_hebrank_wright['predicted_azim'] = pd_hebrank_wright['predicted'].apply(lambda x: x%72)
    pd_hebrank_wright['azim'] = pd_hebrank_wright['azim'].apply(lambda x : 5*x)
    pd_hebrank_wright['elev'] = pd_hebrank_wright['elev'].apply(lambda x : 10*(x/2))
    pd_hebrank_wright['hebrank_wright_pos'] = (pd_hebrank_wright
                                               .apply(lambda x: x['elev'] if x['azim'] == 0
                                                      else 180 - x['elev'],axis=1))
    pd_hebrank_wright['hebrank_wright_pred_pos'] = (pd_hebrank_wright
                                               .apply(lambda x: x['predicted_elev'] if
                                                      (x['predicted_azim'] >= 350 or
                                                       x['predicted_azim'] <=10)
                                                      else 180 - x['predicted_elev'],axis=1))
    #claculate error
    pd_hebrank_wright['elev_error'] = (pd_hebrank_wright['predicted_elev'] -
                                                    pd_hebrank_wright['elev']).abs()
    pd_hebrank_wright['hebrank_wright_error'] = (pd_hebrank_wright['hebrank_wright_pred_pos'] -
                                                    pd_hebrank_wright['hebrank_wright_pos']).abs()
    pd_hebrank_wright['correct_within_15'] = pd_hebrank_wright['hebrank_wright_error'].apply(lambda x: x < 15)
    pd_hebrank_wright['correct_within_45'] = pd_hebrank_wright['hebrank_wright_error'].apply(lambda x: x < 45)
    return pd_hebrank_wright

def format_hebrank_wright_dataframe_azim_limited(pd_hebrank_wright,elev_list=[0,30,60]):
    def get_filtering_condition(row):
        if (row['low_cutoff'] == 20):
            if (row['high_cutoff'] != 20000):
                return 'low_pass'
            else:
                return 'full_spec'
        return 'high_pass'
    def set_cutoff(row):
        if row['Filtering Condition'] == 'low_pass':
            return row['high_cutoff']
        return row['low_cutoff']
    pd_hebrank_wright = pd_hebrank_wright.convert_objects(convert_numeric=True)
    if pd_hebrank_wright['azim'].dtype != 'int64':
        pd_hebrank_wright['azim'] = pd_hebrank_wright['azim'].apply(convert_from_numpy)
        pd_hebrank_wright['elev'] = pd_hebrank_wright['elev'].apply(convert_from_numpy)
    elev_list = np.array(elev_list)
    #Get azim and elevation based on predicted index
    pd_hebrank_wright['predicted_elev'] = None
    pd_hebrank_wright['predicted_azim'] = None
    pd_hebrank_wright['predicted_elev'] = pd_hebrank_wright['predicted'].apply(lambda x: 10*(x//72))
    pd_hebrank_wright['predicted_azim'] = pd_hebrank_wright['predicted'].apply(lambda x: x%72)
    pd_hebrank_wright['azim'] = pd_hebrank_wright['azim'].apply(lambda x : 5*x)
    pd_hebrank_wright['elev'] = pd_hebrank_wright['elev'].apply(lambda x : 10*(x/2))
    pd_hebrank_wright = pd_hebrank_wright.query("elev in @elev_list")
    pd_hebrank_wright['hebrank_wright_pos'] = (pd_hebrank_wright
                                               .apply(lambda x: x['elev'] if x['azim'] == 0
                                                      else 180 - x['elev'],axis=1))
    pd_hebrank_wright['hebrank_wright_pred_elev'] = (pd_hebrank_wright['predicted_elev']
                                                     .apply(lambda x: elev_list[np.abs(elev_list-x).argmin()]))
    pd_hebrank_wright['hebrank_wright_pred_pos'] = (pd_hebrank_wright
                                               .apply(lambda x: x['hebrank_wright_pred_elev'] if
                                                      (x['predicted_azim'] >= 350 or
                                                       x['predicted_azim'] <=10)
                                                      else 180 - x['hebrank_wright_pred_elev'],axis=1))
    #claculate error
    pd_hebrank_wright['elev_error'] = (pd_hebrank_wright['predicted_elev'] -
                                                    pd_hebrank_wright['elev']).abs()
    pd_hebrank_wright['hebrank_wright_error'] = (pd_hebrank_wright['hebrank_wright_pred_pos'] -
                                                    pd_hebrank_wright['hebrank_wright_pos']).abs()
    pd_hebrank_wright['correct_within_15'] = pd_hebrank_wright['hebrank_wright_error'].apply(lambda x: x < 15)
    pd_hebrank_wright['correct_within_45'] = pd_hebrank_wright['hebrank_wright_error'].apply(lambda x: x < 45)
    #Adding columns for human/model comparison
    pd_hebrank_wright["Filtering Condition"] = pd_hebrank_wright.apply(lambda x: get_filtering_condition(x),axis=1)
    pd_hebrank_wright['Frequency (Hz)'] = pd_hebrank_wright.apply(lambda x: set_cutoff(x),axis=1)
    return pd_hebrank_wright

def make_hebrank_wright_plots(pd_hebrank_wright_formatted,y_axis="correct_within_15"):
    plt.clf()
    pd_hebrank_wright_arch_mean = (pd_hebrank_wright_formatted
                                   .groupby(['low_cutoff','high_cutoff','arch_index'])
                                   .mean().reset_index())
    pd_hebrank_wright_arch_mean_low = pd_hebrank_wright_arch_mean.query("low_cutoff == 20 & high_cutoff != 20000")
    pd_hebrank_wright_arch_mean_high = pd_hebrank_wright_arch_mean.query("high_cutoff == 20000 & low_cutoff != 20")
    g = sns.lineplot(data=pd_hebrank_wright_arch_mean_low,x="high_cutoff",y=y_axis,ci=68,
                     err_style='bars',lw=0.75)
    sns.lineplot(data=pd_hebrank_wright_arch_mean_high,x="low_cutoff",y=y_axis,ci=68,
                    err_style='bars',lw=0.75)
    g.set(xticklabels=[])
    g.set_xscale('log')
    g.set_yticks([0,.2,.4,.6,.8,1.0])
    g.set_yticklabels([0,.2,.4,.6,.8,1.0],fontsize=10)
    g.set_xticks([4000,6000,8000,10000,12000,15000,18000])
    g.set_xticklabels([4000,6000,8000,10000,12000,15000,18000],fontsize=10,rotation=45)
    g.set_ylabel("Proportion of Correct Responses",fontsize=10)
    g.set_xlabel("Cutoff Frequency (Hz)",fontsize=10)
    #plt.legend(["Low Pass","High Pass"],title="Filtering Condition",fontsize=15)
    pd_hebrank_wright_arch_mean.to_csv(plots_folder + "/hebrank_wright.csv")
    plt.savefig(plots_folder+"/hebrank_wright_{}.svg".format(y_axis))

def get_hebrank_wright_human_data():
    pd_hebrank_wright = pd.read_csv("/om/user/francl/hebrank_wright_human_data.csv",header=[0,1])   
    pd_x_low = [3900,6000,8000,10300,12000,14500,16000]
    pd_x_high = [3800,5800,7500,10000,13200,15300,np.nan]
    pd_hebrank_wright.loc[:,("high_pass","X")] = pd_x_high
    pd_hebrank_wright.loc[:,("high_pass","Y")] = (pd_hebrank_wright
                                                  .loc[:,("high_pass","Y")]
                                                  .apply(lambda x: x/100))
    pd_hebrank_wright.loc[:,("low_pass","X")] = pd_x_low
    pd_hebrank_wright.loc[:,("low_pass","Y")] = (pd_hebrank_wright
                                                  .loc[:,("low_pass","Y")]
                                                  .apply(lambda x: x/100))
    pd_hebrank_wright = pd_hebrank_wright.stack(level=0).reset_index()
    pd_hebrank_wright.rename({"level_1":"Filtering Condition",
                                   "X":"Frequency (Hz)",
                                   "Y":"Proportion of Correct Responses"},axis=1,inplace=True)
    return pd_hebrank_wright

def make_hebrank_wright_human_plot():
    pd_hebrank_wright_human = get_hebrank_wright_human_data()
    plt.clf()
    hue_order = ["low_pass","high_pass"]
    g = sns.lineplot(data=pd_hebrank_wright_human,x="Frequency (Hz)",
                 y="Proportion of Correct Responses",hue="Filtering Condition",
                 style="Filtering Condition",hue_order=hue_order,err_style='bars',
                 markers=["o","o"],dashes=False,ms=2.0,mew=0.05,lw=0.75)
    g.set(xticklabels=[])
    g.set_xscale('log')
    g.set_yticks([0,.2,.4,.6,.8,1.0])
    g.set_yticklabels([0,.2,.4,.6,.8,1.0],fontsize=10)
    g.set_xticks([4000,6000,8000,10000,12000,15000,18000])
    g.set_xticklabels([4000,6000,8000,10000,12000,15000,18000],fontsize=10,rotation=45)
    g.set_ylabel("Proportion of Correct Responses",fontsize=10)
    g.set_xlabel("Cutoff Frequency (Hz)",fontsize=10)
    #plt.xscale('log')
    #plt.yticks([0,.2,.4,.6,.8,1.0],fontsize=10)
    #plt.xticks([4000,6000,8000,10000,12000,15000,18000],fontsize=10,rotation=45)
    #plt.ylabel("Proportion of Correct Responses",fontsize=10)
    #plt.xlabel("Cutoff Frequency (Hz)",fontsize=10)
    plt.legend(["Low Pass","High Pass"],title="Filtering Condition",fontsize=15)
    plt.savefig(plots_folder+"/hebrank_wright_human_data.svg")

def get_hebrank_wright_correlations(network_regex=None,bootstrap_mode=False):
    if network_regex is None:
        network_regex = ("/om5/user/francl/grp-om2/gahlm/dataset_pipeline_test"
                         "/arch_number_*_init_0/batch_conditional_noiseRecords"
                         "_hebrank_wright_iter100000.npy")
    pd_hebrank_wright = make_dataframe(network_regex,elevation_predictions=True)
    pd_hebrank_wright_network_formatted = format_hebrank_wright_dataframe_azim_limited(pd_hebrank_wright)
    pd_hebrank_wright_network_formatted.rename(columns={"correct_within_15":"Proportion of Correct Responses"},
                                               inplace=True)
    pd_hebrank_wright_network_formatted_subset = (
        pd_hebrank_wright_network_formatted[
            pd_hebrank_wright_network_formatted["Filtering Condition"] != 'full_spec'
        ]
        .loc[:,["Filtering Condition","Frequency (Hz)","Proportion of Correct Responses"]]
        .sort_values(by=["Filtering Condition","Frequency (Hz)"])
    )
    pd_hebrank_wright_network_mean = (pd_hebrank_wright_network_formatted_subset
                                      .groupby(["Filtering Condition",
                                                "Frequency (Hz)"])
                                      .mean().reset_index())
    pd_hebrank_wright_human = (get_hebrank_wright_human_data()
                               .sort_values(by=["Filtering Condition","Frequency (Hz)"]))
    if bootstrap_mode:
        return (pd_hebrank_wright_human,pd_hebrank_wright_network_mean)
    pd_human_model= pd_hebrank_wright_network_mean
    pd_human_model["Proportion of Correct Human Responses"] = (
        pd_hebrank_wright_human["Proportion of Correct Responses"].values
    )
    pd_human_model_norm = (
        pd_normalize_column(pd_human_model,
                            columns=["Proportion of Correct Responses",
                                    "Proportion of Correct Human Responses"],
                            norm_across_cols=True,
                            columns_out=["model_norm","human_norm"])
    )
    (kendall_tau,spearman_r,rmse) = get_stats(pd_human_model_norm["human_norm"],
                                              pd_human_model_norm["model_norm"])
    return (kendall_tau,spearman_r,rmse)


def get_hebrank_wright_correlations_bootstrapped(network_regex=None,
                                           model_choices=None):
    assert model_choices is not None
    fnames = sorted(glob(network_regex))
    model_data = []
    for model in fnames:
        pd_human,pd_model = get_hebrank_wright_correlations(network_regex=model,
                                                      bootstrap_mode=True)
        model_data.append(pd_model)
    rmse_list = []
    for idx,model_idx in enumerate(model_choices):
        pd_model_sample = pd.concat([model_data[i] for i in model_idx])
        pd_model_mean = pd_model_sample.groupby(pd_model_sample.index).mean()
        pd_human_model= pd_model_mean
        pd_human_model["Proportion of Correct Human Responses"] = (
            pd_human["Proportion of Correct Responses"].values
        )
        pd_human_model_norm = (
            pd_normalize_column(pd_human_model,
                                columns=["Proportion of Correct Responses",
                                        "Proportion of Correct Human Responses"],
                                norm_across_cols=True,
                                columns_out=["model_norm","human_norm"])
        )
        (kendall_tau,spearman_r,rmse) = get_stats(pd_human_model_norm["human_norm"],
                                                  pd_human_model_norm["model_norm"])
        rmse_list.append(rmse)
    return rmse_list

def get_validation_set_error_degrees(regex=None):
    if regex is None:
        regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test"
                 "/arch_number_*_init_0/batch_conditional_testset_"
                 "stimRecords_convolved_oldHRIRdist140*iter100000.npy")
    pd_testset = make_dataframe(regex,fold_data=False,elevation_predictions=True)
    pd_testset = calculate_predictions_and_error_testset(pd_testset)
    pd_means = pd_testset.mean()
    pd_testset_dict = {"Elev Error":pd_means['elev_error_abs'],
                       "Azim Error":pd_means['azim_error_abs'],
                       "Azim Error Folded":pd_means['azim_error_folded_abs']}
    return pd_testset_dict

def format_multi_source_net(pd_multi_source):
    pd_multi_source['elev'] = pd_multi_source['elev'].apply(lambda x: [val//2 for val in x])
    #pd_multi_source['azim'] = pd_multi_source['azim'].apply(lambda x: [val*5 if val !=-1 else val for val in x])

def filter_multi_source_predictions(pd_multi_source,min_sigmoid_value=0.0,
                                    max_sigmoid_value=1.0):
    min_sgmoid_bool = pd_multi_source['predicted'].apply(lambda x: max(x) > min_sigmoid_value)
    max_sgmoid_bool = pd_multi_source['predicted'].apply(lambda x: max(x) < max_sigmoid_value)
    select_cols = min_sgmoid_bool & max_sgmoid_bool
    return pd_multi_source[select_cols]

def mean_prediction_by_ref(pd_multi_source):
    pd_pred_by_ref = pd_multi_source.groupby(['reference_azim','reference_elev'])['predicted']

def get_yost_correlations(network_regex=None,bootstrap_mode=False):
    make_bandwidth_vs_error_humabn_plot()
    pd_yost_human = get_data_from_graph(idx=0)
    pd_yost_network = make_yost_network_plot(network_regex)
    pd_yost_human["X"] = [0.0,.05,.1,.167,.333,1.0,2.0]
    pd_network_intersection = pd_yost_network["X"].round(2).isin(pd_yost_human["X"].round(2))
    pd_yost_network_subset = pd_yost_network[pd_network_intersection]
    if bootstrap_mode:
        return pd_yost_human,pd_yost_network_subset
    #flip network data and normalize
    (kendall_tau,spearman_r,rmse) = get_stats(pd_yost_human,
                                              pd_yost_network_subset)
    return (kendall_tau,spearman_r,rmse)

def make_kulkarni_plot_networks(regex=None):
    if regex is None:
        regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/arch_number_*_init_0"
                 "/batch_conditional_broadbandNoiseRecords_convolved_smooth*"
                 "HRIR_direct_stackedCH_upsampled_iter100000.npy")
    np_data_array_dictionary_smoothed_hrtf = make_dataframe(regex,fold_data=False,elevation_predictions=True)
    pd_dataframe_smoothed_hrtf_filtered_full_length = get_error_smoothed_hrtf(np_data_array_dictionary_smoothed_hrtf)
    pd_smoothed_hrir_mismatch_full_length = compare_smoothed_hrtf(pd_dataframe_smoothed_hrtf_filtered_full_length)
    pd_smoothed_hrir_mismatch_full_length["Total Error"] = (pd_smoothed_hrir_mismatch_full_length["Elevation Error"] +
                                                            pd_smoothed_hrir_mismatch_full_length["Azimuth Error"])
    plot_kulkarni_networks(pd_smoothed_hrir_mismatch_full_length)
    return pd_smoothed_hrir_mismatch_full_length

def make_kulkarni_network_prediction_plots(regex=None):
    if regex is None:
        regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/arch_number_*_init_0"
                 "/batch_conditional_broadbandNoiseRecords_convolved_smooth*"
                 "HRIR_direct_stackedCH_upsampled_iter100000.npy")
    np_data_array_dictionary_smoothed_hrtf = make_dataframe(regex,fold_data=False,elevation_predictions=True)
    pd_dataframe_smoothed_hrtf_filtered = get_error_smoothed_hrtf(np_data_array_dictionary_smoothed_hrtf)
    model_order = np.sort(pd_dataframe_smoothed_hrtf_filtered["smooth_factor"].unique())[::-1] 
    pd_smoothed_arch_mean = pd_dataframe_smoothed_hrtf_filtered.groupby(["azim","elev","smooth_factor","arch_index"]).mean().reset_index()
    plt.clf()
    sns.lineplot(x="azim",y="predicted_azim",hue="smooth_factor",
                 data=pd_smoothed_arch_mean[pd_smoothed_arch_mean['elev'].isin([0])],ci=68,
                 hue_order=model_order,legend='full',palette=sns.color_palette("colorblind",n_colors=9))
    plt.xticks([-75,-50,-25,0,25,50,75],fontsize=10,rotation=75)
    plt.yticks([-75,-50,-25,0,25,50,75],fontsize=10)
    plt.ylabel("Judged Azim",fontsize=10)
    plt.xlabel("Azim",fontsize=10)
    plt.savefig(plots_folder+"/azim_vs_predicted_by_smooth_factor_with_legend.png",dpi=400)
    
    plt.tight_layout()
    plt.gca().get_legend().remove()
    plt.savefig(plots_folder+"/azim_vs_predicted_by_smooth_factor.svg")
    plt.clf()
    sns.lineplot(x="elev",y="predicted_elev",hue="smooth_factor",
                 data=pd_smoothed_arch_mean,legend=False,ci=68,
                 hue_order=model_order,palette=sns.color_palette("colorblind",n_colors=9))
    plt.xticks([0,10,20,30,40,50,60],fontsize=10,rotation=75)
    plt.yticks([0,10,20,30,40,50,60],fontsize=10)
    plt.ylabel("Judged Elev",fontsize=10)
    plt.xlabel("Elev",fontsize=10)
    plt.tight_layout()
    pd_smoothed_arch_mean.to_csv(plots_folder+"/elev_vs_predicted_by_smooth_factor.csv")
    plt.savefig(plots_folder+"/elev_vs_predicted_by_smooth_factor.svg")


def get_kulkarni_correlations(network_regex=None,bootstrap_mode=False):
    format_kulkarni_human_data()
    pd_kulkarni_human = get_data_from_graph(use_points=True)
    pd_smoothed_hrir_mismatch_full_length = make_kulkarni_plot_networks(network_regex)
    pd_kulkarni_network = get_data_from_graph(idx=0)
    #Chnage from coordinates (0,355) to (-175,180)
    pd_kulkarni_human["X"] = [1024,512,256,128,64,32,16,8]
    pd_kulkarni_network["X"] = [1024,512,256,128,64,32,16,8,4,2,1]
    pd_kulkarni_network = pd_kulkarni_network[pd_kulkarni_network["Y"].notna()]
    pd_network_intersection = pd_kulkarni_network["X"].round(2).isin(pd_kulkarni_human["X"].round(2))
    pd_kulkarni_network_subset = pd_kulkarni_network[pd_network_intersection]
    pd_kulkarni_human_intersection = pd_kulkarni_human["X"].round(2).isin(pd_kulkarni_network_subset["X"].round(2))
    pd_kulkarni_human_subset = pd_kulkarni_human[pd_kulkarni_human_intersection]
    if bootstrap_mode:
        return (pd_kulkarni_human_subset,pd_kulkarni_network_subset)
    pd_kulkarni_network_norm = pd_normalize_column(pd_kulkarni_network_subset)
    pd_kulkarni_human_norm = pd_normalize_column(pd_kulkarni_human_subset)
    #flip network data and normalize
    (kendall_tau,spearman_r,rmse) = get_stats(pd_kulkarni_human_norm,
                                              pd_kulkarni_network_norm)
    return (kendall_tau,spearman_r,rmse)

def get_litovsky_human_data(add_errorbars=False):
    if add_errorbars:
        pd_litovsky = pd.read_csv("/om/user/francl/litovsky_errobars.csv",header=[0,1])   
    else:
        pd_litovsky = pd.read_csv("/om/user/francl/litovsky_human_data.csv",header=[0,1])   
    pd_x_value = [-1,5,10,25,50,100]
    pd_litovsky.loc[:,("Lead Click Adult","X")] = pd_x_value
    pd_litovsky.loc[:,("Lead Click Child","X")] = pd_x_value
    pd_litovsky.loc[:,("Lag Click Adult","X")] = pd_x_value
    pd_litovsky.loc[:,("Lag Click Child","X")] = pd_x_value
    if add_errorbars:
        pd_litovsky.loc[:,("Lead Click Adult Top","X")] = pd_x_value
        pd_litovsky.loc[:,("Lag Click Adult Top","X")] = pd_x_value

    return pd_litovsky

def grouped_barplot(df, cat,subcat, val , err):
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = np.sort(df[subcat].unique())[::-1]
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x+offsets[i], dfg[val].values, width=width, 
                label="{} {}".format(subcat, gr), yerr=dfg[err].values)
    plt.xlabel(cat)
    plt.ylabel(val)
    plt.xticks(x, u)

def make_litovsky_human_plot(bars_to_plot=["Lead Click Adult",
                                           "Lag Click Adult"],
                             add_errorbars=False):
    pd_human_data = get_litovsky_human_data(add_errorbars=add_errorbars)
    #unstack multiindex
    if add_errorbars:
        pd_human_data['Lead Click Adult','YErr'] = abs(pd_human_data['Lead Click Adult Top','Y']-pd_human_data['Lead Click Adult','Y'])
        pd_human_data['Lag Click Adult','YErr'] = abs(pd_human_data['Lag Click Adult Top','Y']-pd_human_data['Lag Click Adult','Y'])
    pd_human_data = pd_human_data.stack(level=[0]).reset_index()
    pd_human_data =  pd_human_data.replace({"X":-1},"SS")
    pd_human_data_filtered = pd_human_data.query("level_1 in @bars_to_plot")
    pd_human_data_filtered.rename({"level_1":"Click Timing",
                                   "X":"Lead Click Delay (ms)",
                                   "Y":"Average RMS Error"},axis=1,inplace=True)
    plt.clf()
    sns.barplot(x="Lead Click Delay (ms)",y="Average RMS Error",
                hue="Click Timing",hue_order=bars_to_plot,
                palette=sns.xkcd_palette(["black","light grey"]),
                data=pd_human_data_filtered,
                errcolor=".2", edgecolor=".2")
    if add_errorbars:
        bar_locs = sorted([(bar.get_x()+bar.get_width()/2.0,bar.get_height())
                           for bar in plt.gca().patches],key=lambda x: x[0])
        x = [x[0] for x in bar_locs]
        y = [x[1] for x in bar_locs]
        plt.errorbar(x,y,yerr=pd_human_data_filtered["YErr"],lw=0,elinewidth=2,ecolor='k')
    plt.yticks([0,10,20,30,40])

def format_click_precedence_effect_dataframe(pd_dataframe_precedence):
    pd_dataframe_precedence['predicted_folded'] = pd_dataframe_precedence['predicted_folded'].apply(lambda x: x*5)
    pd_dataframe_precedence['predicted_folded'] = pd_dataframe_precedence['predicted_folded'].apply(lambda x: x 
                                                                                                    if x < 180 else x-360)
    pd_dataframe_precedence.columns = ['predicted', 'delay', 'start_sample',
                                       'Lead Click Level Difference (dB)',
                                       'lag_level','Lead Click Position',
                                       'arch_index', 'init_index',
                                       'predicted_folded']
    pd_dataframe_precedence['Lead Click Level Difference (dB)'] = pd_dataframe_precedence['lag_level'].apply(lambda x:
                                                                                       {'-10.0':0,'-12.5':5,'-7.5':-5,
                                                                                        '-20.0':20,'-15.0':10,'-2.5':-15,
                                                                                        '-17.5':15,'-5.0':-10,'0.0':-20}[str(x)])
    pd_dataframe_precedence['Lead Click Position'] = pd_dataframe_precedence['Lead Click Position'].apply(lambda x: -45 if x==0
                                                                                                          else 45)
    pd_dataframe_precedence['predicted_folded'] = pd_dataframe_precedence['predicted_folded'].apply(lambda x: x*-1)
    return pd_dataframe_precedence

def make_click_precedence_effect_plot(regex=None):
    if regex is None:
        regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                 "arch_number_*init_0/batch_conditional_precedenceEffectRecords"
                 "_45DegOffset_jitteredStart_jitteredPt5msDelay_expanded_stackedCH*iter100000*")
        pd_dataframe_precedence = make_dataframe(regex,fold_data=True)
        pd_dataframe_precedence = format_click_precedence_effect_dataframe(pd_dataframe_precedence)
        pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['Lead Click Position'].isin([45])) & (pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0]))].astype('int32')
        pd_filtered_precedence_arch_mean = pd_filtered_precedence.groupby(['arch_index','delay','Lead Click Level Difference (dB)']).mean().reset_index()
        plt.clf()
        plt.figure()
        sns.lineplot(x="delay",
                     y="predicted_folded",data=pd_filtered_precedence_arch_mean,
                     color='k',err_style='bars',ci=68,lw=1.0)
        format_precedence_effect_graph()
        plt.yticks([-40,-20,0,20,40])
        plt.xticks([0,10,20,30,40,50])
        plt.tight_layout()
        pd_filtered_precedence_arch_mean.to_csv(plots_folder+"/"+"precedence_effect_graph_lineplot_clicks.csv")
        plt.savefig(plots_folder+"/"+"precedence_effect_graph_lineplot_clicks.svg")

def make_click_precedence_effect_across_conditions():
    plt.clf()
    plt.figure()
    regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
             "arch_number_*init_0/batch_conditional_precedenceEffectRecords"
             "_45DegOffset_jitteredStart_jitteredPt5msDelay_expanded_stackedCH*iter100000*")
    pd_dataframe_precedence = make_dataframe(regex,fold_data=True)
    pd_dataframe_precedence = format_click_precedence_effect_dataframe(pd_dataframe_precedence)
    pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['Lead Click Position'].isin([45])) & (pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0]))].astype('int32')
    pd_filtered_precedence_arch_mean_normal = pd_filtered_precedence.groupby(['arch_index','delay','Lead Click Level Difference (dB)']).mean().reset_index()
    pd_filtered_precedence_arch_mean_normal["Training Condition"] = "Normal"
    regex = ("/om2/user/francl/new_task_archs/new_task_archs_anechoic_training/"
             "arch_number_*init_0/batch_conditional_precedenceEffectRecords"
             "_45DegOffset_jitteredStart_jitteredPt5msDelay_expanded_stackedCH*iter100000*")
    pd_dataframe_precedence = make_dataframe(regex,fold_data=True)
    pd_dataframe_precedence = format_click_precedence_effect_dataframe(pd_dataframe_precedence)
    pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['Lead Click Position'].isin([45])) & (pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0]))].astype('int32')
    pd_filtered_precedence_arch_mean_anechoic = pd_filtered_precedence.groupby(['arch_index','delay','Lead Click Level Difference (dB)']).mean().reset_index()
    pd_filtered_precedence_arch_mean_anechoic["Training Condition"] = "Anechoic"
    regex = ("/om2/user/francl/new_task_archs/new_task_archs_no_background_noise_80dBSNR_training/"
             "arch_number_*init_0/batch_conditional_precedenceEffectRecords"
             "_45DegOffset_jitteredStart_jitteredPt5msDelay_expanded_stackedCH*iter100000*")
    pd_dataframe_precedence = make_dataframe(regex,fold_data=True)
    pd_dataframe_precedence = format_click_precedence_effect_dataframe(pd_dataframe_precedence)
    pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['Lead Click Position'].isin([45])) & (pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0]))].astype('int32')
    pd_filtered_precedence_arch_mean_no_background = pd_filtered_precedence.groupby(['arch_index','delay','Lead Click Level Difference (dB)']).mean().reset_index()
    pd_filtered_precedence_arch_mean_no_background["Training Condition"] = "No Background"
    regex = ("/om5/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_training/"
             "arch_number_*init_0/batch_conditional_precedenceEffectRecords"
             "_45DegOffset_jitteredStart_jitteredPt5msDelay_expanded_stackedCH*iter150000*")
    pd_dataframe_precedence = make_dataframe(regex,fold_data=True)
    pd_dataframe_precedence = format_click_precedence_effect_dataframe(pd_dataframe_precedence)
    pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['Lead Click Position'].isin([45])) & (pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0]))].astype('int32')
    pd_filtered_precedence_arch_mean_unnatural_sounds = pd_filtered_precedence.groupby(['arch_index','delay','Lead Click Level Difference (dB)']).mean().reset_index()
    pd_filtered_precedence_arch_mean_unnatural_sounds["Training Condition"] = "Unnatural Sounds"
    pd_filtered_precedence_arch_mean_all = pd.concat([pd_filtered_precedence_arch_mean_normal,
                                                      pd_filtered_precedence_arch_mean_anechoic,
                                                      pd_filtered_precedence_arch_mean_no_background,
                                                      pd_filtered_precedence_arch_mean_unnatural_sounds])
    #new_palette = ["#97B4DE","#785EF0","#DC267F","#FE6100","006838"]
    pdb.set_trace()
    new_palette = ["#97B4DE","#785EF0","#DC267F","#FE6100"]
    sns.lineplot(x="delay",y="predicted_folded",hue="Training Condition",
                 data=pd_filtered_precedence_arch_mean_all,err_style='bars',ci=68,
                 palette=sns.color_palette(new_palette))
    format_precedence_effect_graph()
    plt.gca().get_legend().remove()
    plt.xticks([0,10,20,30,40,50])
    plt.yticks([-40,-20,0,20,40])
    plt.tight_layout()
    pd_filtered_precedence_arch_mean_all.to_csv(plots_folder+"/"+"precedence_effect_graph_lineplot_clicks_across_conditions.csv")
    plt.savefig(plots_folder+"/"+"precedence_effect_graph_lineplot_clicks_across_conditions.svg")
    ####


def make_precedence_effect_plot_networks(regex=None,filter_lead_pos_list=[-45]):
    if regex is None:
       regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                "arch_number_*init_0/batch_conditional_*precedence*"
                "45Deg0DegOffset*pinknoise*5degStep*iter100000*")
    pd_dataframe_precedence = make_dataframe(regex,fold_data=True)
    pdb.set_trace()
    pd_dataframe_precedence = format_precedence_effect_dataframe(pd_dataframe_precedence)
    pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['Lead Click Position'].isin(filter_lead_pos_list)) &\
                            (pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0]))].astype('int32')
    pd_filtered_precedence_arch_mean = pd_filtered_precedence.groupby(['arch_index','delay','Lead Click Level Difference (dB)'])\
    .mean().reset_index()
    plt.clf()
    sns.lineplot(x="delay",y="predicted_folded",data=pd_filtered_precedence_arch_mean,err_style='bars',ci=68)
    format_precedence_effect_graph()
    return pd_filtered_precedence_arch_mean

def  make_precedence_effect_plot_networks_multi_azim(regex=None,
                                                     filter_lead_pos_list=None,
                                                     filter_flipped_values_list=None):
    if regex is None:
       regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                "arch_number_*init_0/batch_conditional_*precedence*"
                "multiAzim*pinknoise*5degStep*iter100000*")
    data_condtion = regex.split("batch_conditional")[1]
    pd_dataframe_precedence = make_dataframe(regex,fold_data=True)
    pd_dataframe_precedence = format_precedence_effect_dataframe_multi_azim(pd_dataframe_precedence)
    if filter_lead_pos_list is None:
        filter_lead_pos_list = pd_dataframe_precedence['azim'].unique()
    if filter_flipped_values_list is None:
        filter_flipped_values_list = pd_dataframe_precedence['flipped'].unique()
    for filter_flipped_value in  filter_flipped_values_list:
        for filter_lead_pos in filter_lead_pos_list:
            pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['azim'].isin([filter_lead_pos])) &\
                                    (pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0])) &\
                                    (pd_dataframe_precedence['flipped'].isin([filter_flipped_value]))].astype('int32')
            pd_filtered_precedence_arch_mean = pd_filtered_precedence.groupby(['arch_index','delay','Lead Click Level Difference (dB)'])\
                                                                     .mean().reset_index()
            plt.clf()
            sns.lineplot(x="delay",y="predicted_folded",data=pd_filtered_precedence_arch_mean,err_style='bars',ci=68)
            format_precedence_effect_graph_multi_azim()
            plt_name  = "/precedence_effect_multi_azim_{}_stim_{}_lead_click_{}_flipped.png".format(data_condtion,
                                                                                                   filter_lead_pos,
                                                                                                   filter_flipped_value)
            plt.savefig(plots_folder+plt_name)
    return pd_filtered_precedence_arch_mean

def get_lead_error(row):
    if row['Lead Click Position'] == 45:
        return (row['predicted_folded'] +45)**2
    else:
        return (row['predicted_folded'] +0)**2

def get_lag_error(row):
    if row['Lead Click Position'] == 45:
        return (row['predicted_folded'] +0)**2
    else:
        return (row['predicted_folded'] +45)**2

def get_lead_error_multi_azim(row):
    if row['flipped'] == 0:
        azim = row['azim']*5
        azim = azim if azim<180 else azim-360
        return (row['predicted_folded'] - azim)**2
    else:
        return (row['predicted_folded'] -0)**2

def get_lag_error_mult_azim(row):
    if row['flipped'] == 0:
        return (row['predicted_folded'] -0)**2
    else:
        azim = row['azim']*5
        azim = azim if azim<180 else azim-360
        return (row['predicted_folded'] -azim)**2

def litovsky_error_plot(regex=None):
    if regex is None:
       regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                "arch_number_*init_0/batch_conditional_*precedence*"
                "45Deg0DegOffset*pinknoise*5degStep*iter100000*")
    pd_dataframe_precedence = make_dataframe(regex,fold_data=True)
    pd_dataframe_precedence = format_precedence_effect_dataframe(pd_dataframe_precedence)
    #pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['Lead Click Position'].isin([45])) &\
    #                        (pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0]))].astype('int32')
    pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0]))].astype('int32')
    pd_filtered_precedence['Lead Click Position'] = pd_filtered_precedence['Lead Click Position'].apply(lambda x: 45 if x == -45
                                                                                                        else 0)
    pd_filtered_precedence["Lead_error"] = pd_filtered_precedence.apply(get_lead_error,axis=1)
    pd_filtered_precedence["Lag_error"] = pd_filtered_precedence.apply(get_lag_error,axis=1)
    pd_filtered_precedence_arch_mean = pd_filtered_precedence.groupby(['arch_index','delay',\
                                                                       'Lead Click Level Difference (dB)',\
                                                                       'Lead Click Position'])\
                                                             .mean().pow(1./2).reset_index()
    pd_x_value = [-1,5,10,25,50,100]
    pd_litovsky_network_subset = pd_filtered_precedence_arch_mean[pd_filtered_precedence_arch_mean["delay"].isin(pd_x_value)]
    #This replaces the leading click RMSE from the lag postion in the single click condtion with the
    #RMSE from the lagging click in the single click condtion to match the the
    #conditon in the paper
    pd_litovsky_network_subset_lag_pos_intersection = (pd_litovsky_network_subset["delay"] == -1) &\
                                                        (pd_litovsky_network_subset["Lead Click Position"] == 0)
    pd_litovsky_network_subset_lead_pos_intersection = (pd_litovsky_network_subset["delay"] == -1) &\
                                                        (pd_litovsky_network_subset["Lead Click Position"] == 45)
    pd_litovsky_network_subset.loc[pd_litovsky_network_subset_lead_pos_intersection,"Lag_error"] = \
            pd_litovsky_network_subset.loc[pd_litovsky_network_subset_lag_pos_intersection,"Lead_error"].values
    pd_litovsky_network_subset_filtered = pd_litovsky_network_subset[pd_litovsky_network_subset['Lead Click Position'] == 45]
    pd_litovsky_network_subset_filtered_rename = pd_litovsky_network_subset_filtered.replace({"delay":-1},"SS")
    #Combine dataframe rows for plotting
    pd_litovsky_error = pd_litovsky_network_subset_filtered_rename.melt(id_vars=["delay"],value_vars=["Lead_error","Lag_error"],
                                                       var_name='Error Condition',value_name='RMSE')
    plt.clf()
    sns.barplot(x="delay",y="RMSE",hue="Error Condition",data=pd_litovsky_error)
    plt.ylabel("Average RMS Error (Degrees)")
    plt.xlabel("Lead Click Delay (ms)")
    return pd_litovsky_network_subset_filtered

def litovsky_error_plot_multi_azim(regex=None):
    if regex is None:
       regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                "arch_number_*init_0/batch_conditional_*precedence*"
                "multiAzim*pinknoise*5degStep*iter100000*")
    pd_dataframe_precedence = make_dataframe(regex,fold_data=True)
    pd_dataframe_precedence = format_precedence_effect_dataframe_multi_azim(pd_dataframe_precedence)
    #pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['Lead Click Position'].isin([45])) &\
    #                        (pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0]))].astype('int32')
    pd_filtered_precedence = pd_dataframe_precedence[(pd_dataframe_precedence['Lead Click Level Difference (dB)'].isin([0]))].astype('int32')
    pd_filtered_precedence["Lead_error"] = pd_filtered_precedence.apply(get_lead_error_multi_azim,axis=1)
    pd_filtered_precedence["Lag_error"] = pd_filtered_precedence.apply(get_lag_error_mult_azim,axis=1)
    pd_filtered_precedence_arch_mean = pd_filtered_precedence.groupby(['delay','flipped',\
                                                                       'Lead Click Level Difference (dB)',\
                                                                       'arch_index',])\
                                                             .mean().reset_index()
    pd_x_value = [-1,5,10,25,50,100]
    pd_litovsky_network_subset = pd_filtered_precedence_arch_mean[pd_filtered_precedence_arch_mean["delay"].isin(pd_x_value)]
    #This replaces the leading click RMSE from the lag postion in the single click condtion with the
    #RMSE from the lagging click in the single click condtion to match the the
    #conditon in the paper
    pd_litovsky_network_subset_lag_pos_intersection = (pd_litovsky_network_subset["delay"] == -1) &\
                                                        (pd_litovsky_network_subset["flipped"] == 1)
    pd_litovsky_network_subset_lead_pos_intersection = (pd_litovsky_network_subset["delay"] == -1) &\
                                                        (pd_litovsky_network_subset["flipped"] == 0)
    pd_litovsky_network_subset.loc[pd_litovsky_network_subset_lead_pos_intersection,"Lag_error"] = \
            pd_litovsky_network_subset.loc[pd_litovsky_network_subset_lag_pos_intersection,"Lead_error"].values
    pd_litovsky_network_subset = pd_litovsky_network_subset[~pd_litovsky_network_subset_lag_pos_intersection]
    pd_litovsky_network_subset_arch_mean = pd_litovsky_network_subset.groupby(['delay',\
                                                                       'Lead Click Level Difference (dB)',\
                                                                       'arch_index',])\
                                                             .mean().pow(1./2).reset_index()
    pd_litovsky_network_subset_rename = pd_litovsky_network_subset_arch_mean.replace({"delay":-1},"SS")
    #Combine dataframe rows for plotting
    pd_litovsky_error = pd_litovsky_network_subset_rename.melt(id_vars=["delay"],value_vars=["Lead_error","Lag_error"],
                                                       var_name='Error Condition',value_name='RMSE')
    plt.clf()
    pd_litovsky_error.to_csv(plots_folder + "/litovsky_barplots.csv")
    sns.barplot(x="delay",y="RMSE",hue="Error Condition",
                palette=sns.xkcd_palette(["black","light grey"]),
                errcolor=".2", edgecolor=".2",
                data=pd_litovsky_error)
    plt.legend(loc=1)
    plt.yticks([0,10,20,30,40])
    plt.ylabel("Average RMS Error (Degrees)")
    plt.xlabel("Lead Click Delay (ms)")
    return pd_litovsky_network_subset_arch_mean
    #pd_litovsky_error_agg = (pd_litovsky_error.groupby(["delay","Error Condition"])
    #                         .agg([np.mean,'count',stats.sem])
    #                         .rename(columns={'mean':'mean_squared_error',
    #                                          'sem': 'mean_squared_error_sem',
    #                                          'count':'mean_squared_error_count'})
    #                         .reset_index())


def get_litovsky_correlations(network_regex=None,bootstrap_mode=False):
    pd_litovsky_human = get_litovsky_human_data()
    pd_litovsky_adult = pd_litovsky_human.loc[:,(("Lead Click Adult","Lag Click Adult"),"Y")].values.ravel(order='F')
    pd_pd_litovsky_child = pd_litovsky_human.loc[:,(("Lead Click Child","Lag Click Child"),"Y")].values.ravel(order='F')
    pd_litovsky_error_network = litovsky_error_plot(network_regex)
    pd_litovsky_error_network_mean = pd_litovsky_error_network.groupby(["delay"]).mean().reset_index()
    if bootstrap_mode:
        return pd_litovsky_adult,pd_litovsky_error_network_mean
    pd_litovsky_network_subset_vector = pd_litovsky_error_network_mean[["Lead_error","Lag_error"]].values.ravel(order='F')
    (kendall_tau,spearman_r,rmse) = get_stats(pd_litovsky_adult,pd_litovsky_network_subset_vector)
    return (kendall_tau,spearman_r,rmse)

def get_litovsky_correlations_multi_azim(network_regex=None,bootstrap_mode=False):
    pd_litovsky_human = get_litovsky_human_data()
    pd_litovsky_adult = pd_litovsky_human.loc[:,(("Lead Click Adult","Lag Click Adult"),"Y")].values.ravel(order='F')
    pd_pd_litovsky_child = pd_litovsky_human.loc[:,(("Lead Click Child","Lag Click Child"),"Y")].values.ravel(order='F')
    pd_litovsky_error_network = litovsky_error_plot_multi_azim(network_regex)
    pd_litovsky_error_network_mean = pd_litovsky_error_network.groupby(["delay"]).mean().reset_index()
    if bootstrap_mode:
        return pd_litovsky_adult,pd_litovsky_error_network_mean
    pd_litovsky_network_subset_vector = pd_litovsky_error_network_mean[["Lead_error","Lag_error"]].values.ravel(order='F')
    pd_litovsky_network_subset_vector_norm = pd_normalize_column(pd_litovsky_network_subset_vector)
    pd_litovsky_adult_norm = pd_normalize_column(pd_litovsky_adult)
    (kendall_tau,spearman_r,rmse) = get_stats(pd_litovsky_adult_norm,pd_litovsky_network_subset_vector_norm)
    return (kendall_tau,spearman_r,rmse)

def make_middlebrooks_network_plots(regex=None):
    if regex is None:
        regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/arch_number_*init_0"
                 "/batch_conditional_*middlebrooks_wrightman*999_side_aware"
                 "_normalized_iter100000*")
    pd_dataframe = make_dataframe(regex,fold_data=True,
                                  recenter_folded_data=False)
    pd_dataframe_formatted = format_middlebrooks_wrightman_dataframe(pd_dataframe)
    pd_dataframe_ILD = ILD_residuals_process(pd_dataframe_formatted)
    pd_network_data_ILD = ILD_residuals_plot(pd_dataframe_ILD,extract_lines=True)
    pd_dataframe_ITD = ITD_residuals_process(pd_dataframe_formatted)
    pd_network_data_ITD = ITD_residuals_plot(pd_dataframe_ITD,extract_lines=True)
    pd_dataframe_formatted.to_csv(plots_folder + '/middlebrooks_data.csv')
    return (pd_network_data_ILD,pd_network_data_ITD)



def get_middlebrooks_correlations(regex=None,bootstrap_mode=False):
    pd_middlebrooks_human = format_middlebrooks_human_data()
    pd_human_ILD = pd_middlebrooks_human.loc[:,(("ILD Low","ILD High"),"Y")].values.ravel(order='F')
    pd_human_ITD = pd_middlebrooks_human.loc[:,(("ITD Low","ITD High"),"Y")].values.ravel(order='F')
    pd_middlebrooks_network_ILD,pd_middlebrooks_network_ITD = make_middlebrooks_network_plots(regex=regex)
    pd_network_ILD = pd.concat(pd_middlebrooks_network_ILD)
    pd_network_ITD = pd.concat(pd_middlebrooks_network_ITD)
    if bootstrap_mode:
        return (pd_human_ITD,pd_human_ILD),(pd_network_ITD,pd_network_ILD)
    pd_stats_ITD = get_stats(pd_human_ITD,pd_network_ITD)
    pd_stats_ILD = get_stats(pd_human_ILD,pd_network_ILD)
    return (pd_stats_ITD,pd_stats_ILD)

def get_van_opstal_human_plot(condition,hardcode_ref_grid=False):
    pd_van_opstal_human = pd.read_pickle("/om/user/francl/pd_van_opstal_human_data_extracted.pkl")
    pd_collapsed = extract_van_opstal_data_square_plot(pd_van_opstal_human)
    if condition is 'after':
        columns_to_plot = ['RO After','RJ After','MZ After','PH After']
    elif condition is 'before':
        columns_to_plot = ['RO','RJ','MZ','PH']
    else:
        raise ValueError("Condtion: {} is not supported".format(condition))
    pd_mean_values = make_van_opstal_paper_plot_bootstrapped(pd_collapsed,columns_to_plot) 
    pd_ref_human = pd_collapsed['Reference Grid JO']
    if hardcode_ref_grid:
        grid = pd.Series([(x,y) for y in [20,6.667,-6.667,-20]
                          for x in [-20,-6.667,6.667,20]])
        make_van_opstal_paper_plot(grid,marker='o',color='k',dashes=[5,10],fill_markers=False)
    else:
        make_van_opstal_paper_plot(pd_collapsed['Reference Grid JO'],marker='o',color='k',
                                   dashes=[5,10],fill_markers=False)
    return pd_mean_values,pd_ref_human


def make_van_opstal_network_plots(regex=None):
    if regex is None:
       regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                "arch_number_[0-9][0,1,3-9][0-9]_init_0/"
                "batch_conditional_broadbandNoiseRecords"
                "_convolvedCIPIC*iter100000.npy")
    pd_cipic = make_dataframe(regex,fold_data=False,
                              elevation_predictions=True)
    pd_cipic_filtered = get_elevation_error_CIPIC(pd_cipic,subject=None)
    plt.figure()
    pd_ref_grid = make_van_opstal_paper_plot_network_ref_grid()
    pd_after,pd_before = make_van_opstal_paper_plot_network_data(pd_cipic_filtered)
    return (pd_before,pd_after,pd_ref_grid)

def pointplot_alpha(axis,alpha):
    for patch in axis.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))
        
def make_van_opstal_plot_individual_networks(regex=None):
    if regex is None:
       regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                "arch_number_[0-9][0,1,3-9][0-9]_init_0/"
                "batch_conditional_broadbandNoiseRecords"
                "_convolvedCIPIC*iter100000.npy")
    pd_cipic = make_dataframe(regex,fold_data=False,
                              elevation_predictions=True)
    pd_cipic_filtered = get_elevation_error_CIPIC(pd_cipic,subject=None)
    subjects_sorted = (pd_cipic_filtered.groupby("subject_num").mean().
                       sort_values(by="elev_error").reset_index()['subject_num'])
    quartile_subjects_idx = [round(x*subjects_sorted.shape[0]) for x in [.05,.25,.75,.95]]
    quartile_sujects = subjects_sorted.iloc[quartile_subjects_idx]
    num_quartile_sub = quartile_sujects.shape[0]
    num_total_sub = pd_cipic_filtered['subject_num'].unique().shape[0]-1
    pd_dataframe_cipic_subject_quartile = pd_cipic_filtered.query("subject_num in @quartile_sujects")
    pd_dataframe_cipic_new_ears = pd_cipic_filtered.query("subject_num != 999")
    plt.clf()
    plt.figure(figsize=(1.8,1.8))
    ax1=sns.lineplot(x='elev',y='predicted_elev',hue="Ears Used",data=pd_cipic_filtered,
                     style="Ears Used",markers=["o","o"],lw=1.25,ms=2.0,mew=0.15,err_style="bars",
                     ci=None,dashes=False)
    handles,labels = plt.gca().get_legend_handles_labels()
    ax2 = sns.lineplot(x='elev',y='predicted_elev',hue="subject_num",
                  data=pd_dataframe_cipic_subject_quartile,
                 palette=sns.color_palette(n_colors=1)*num_quartile_sub,
                 style="subject_num",markers=["o"]*num_quartile_sub,dashes=False,
                 err_style='bars',ci=None,lw=0.5,ms=.75,mew=0.15,alpha=0.8,ax=ax1)
    ax3 = sns.lineplot(x='elev',y='predicted_elev',hue="subject_num",
                  data=pd_dataframe_cipic_new_ears,
                 palette=sns.color_palette(n_colors=1)*num_total_sub,
                 style="subject_num",markers=["o"]*num_total_sub,dashes=False,
                 err_style='bars',ci=None,lw=0.35,ms=0.35,mew=0.05,alpha=0.2,ax=ax2)
    #plt.legend(handles,labels)
    plt.gca().legend().remove()
    plt.xlabel("Elevation (Degrees)")
    plt.ylabel("Predicted Elevation (Degrees)")
    plt.xticks([0,10,20,30,40,50],rotation=75)
    plt.yticks([0,10,20,30,40,50])
    plt.tight_layout()
    plt.savefig(plots_folder+"/"+"elev_predicted_vs_label_all_subjects_vs_CIPIC_elev_collapsed.svg")
    plt.clf()
    ax1=sns.lineplot(x='azim',y='predicted_azim',hue="Ears Used",data=pd_cipic_filtered,
                     style="Ears Used",markers=["o","o"],lw=1.25,ms=2.0,mew=0.15,err_style="bars",
                     ci=68,dashes=False)
    handles,labels = plt.gca().get_legend_handles_labels()
    ax2 = sns.lineplot(x='azim',y='predicted_azim',hue="subject_num",
                  data=pd_dataframe_cipic_subject_quartile,
                 palette=sns.color_palette(n_colors=1)*num_quartile_sub,
                 style="subject_num",markers=["o"]*num_quartile_sub,dashes=False,
                 err_style='bars',ci=68,lw=0.5,ms=1.0,mew=0.075,alpha=0.8,ax=ax1)
    ax3 = sns.lineplot(x='azim',y='predicted_azim',hue="subject_num",
                  data=pd_dataframe_cipic_new_ears,
                 palette=sns.color_palette(n_colors=1)*num_total_sub,
                 style="subject_num",markers=["o"]*num_total_sub,dashes=False,
                 err_style='bars',ci=68,lw=0.35,ms=0.35,mew=0.05,alpha=0.2,ax=ax2)
    plt.xlabel("Azimuth (Degrees)")
    plt.ylabel("Predicted Azimuth (Degrees)")
    plt.xticks([-75,-50,-25,0,25,50,75],rotation=75)
    plt.yticks([-75,-50,-25,0,25,50,75])
    plt.gca().legend().remove()
    plt.tight_layout()
    pd_cipic_filtered.to_csv(plots_folder+"/"+"predicted_vs_label_all_subjects_vs_CIPIC.csv")
    plt.savefig(plots_folder+"/"+"azim_predicted_vs_label_all_subjects_vs_CIPIC_elev_collapsed.svg")
    






def get_van_opstal_correlations(regex=None,bootstrap_mode=False):
        pd_means_humans_before,pd_ref_human = get_van_opstal_human_plot('before')
        pd_means_humans_after,pd_ref_human = get_van_opstal_human_plot('after')
        pd_mean_network_before, pd_mean_network_after,pd_networks_ref =\
                make_van_opstal_network_plots(regex)
        pd_networks_ref = pd_networks_ref.apply(tuple,axis=1).apply(pd.Series).reset_index()
        pd_mean_network_before = pd_mean_network_before.reset_index(drop=True)
        pd_mean_network_after = pd_mean_network_after.reset_index(drop=True)
        pd_human_before_diff = pd_ref_human.apply(pd.Series) - pd_means_humans_before['xy'].apply(pd.Series)
        pd_human_after_diff = pd_ref_human.apply(pd.Series) - pd_means_humans_after['xy'].apply(pd.Series)
        pd_network_before_diff = pd_networks_ref - pd_mean_network_before.apply(pd.Series)
        pd_network_after_diff = pd_networks_ref - pd_mean_network_after.apply(pd.Series)
        pd_humans = pd.concat([pd_human_before_diff.abs().reset_index(drop=True),
                               pd_human_after_diff.abs().reset_index(drop=True)])
        pd_networks = pd.concat([pd_network_before_diff.abs().reset_index(drop=True),
                                 pd_network_after_diff.abs().reset_index(drop=True)])
        if bootstrap_mode:
            del pd_networks['index']
            return (pd_humans,pd_networks)
        pd_humans_norm = pd_normalize_column(pd_humans,columns=[0,1],
                                             columns_out=['x','y'],
                                            norm_across_cols=True)
        pd_networks_norm = pd_normalize_column(pd_networks,
                                               columns=[0,1],
                                               columns_out=['x','y'],
                                              norm_across_cols=True)
        stats_x = get_stats(pd_humans_norm,pd_networks_norm,
                                   col="x")
        stats_y = get_stats(pd_humans_norm,pd_networks_norm,
                                   col="y")
        return [stats_x,stats_y]




def collapse_van_opstal_data_human(pd_van_opstal):
    df_x = pd.DataFrame(columns=['X Judged','X Presented','Condition'])
    df_y = pd.DataFrame(columns=['Y Judged','Y Presented','Condition'])
    pd_van_opstal.columns = pd_van_opstal.columns.to_flat_index()    
    pdb.set_trace()
    for condition in pd_van_opstal:
        new_df = pd_van_opstal[condition]
        if "X" in condition[1]:
            mean = [new_df.iloc[i::4].mean() for i in range(4)]
            np_data = np.array([mean,list(range(4)),[condition[0]]*4]).T
            df_x_temp =  pd.DataFrame(data=np_data,columns=['X Judged','X Presented','Condition'])
            df_x = pd.concat([df_x,df_x_temp])
        else:
            mean = [new_df.iloc[(i*4):((i+1)*4)].mean() for i in range(4)]
            np_data = np.array([mean,list(range(4)),[condition[0]]*4]).T
            df_y_temp =  pd.DataFrame(data=np_data,columns=['Y Judged','Y Presented','Condition'])
            df_y = pd.concat([df_y,df_y_temp])
    return df_x,df_y


def extract_van_opstal_data_square_plot(pd_van_opstal):
    not_reference = lambda x: "Reference" not in x
    pd_collapsed = pd.DataFrame()
    pd_van_opstal.columns = pd_van_opstal.columns.to_flat_index()
    pd_van_opstal_transpose = pd_van_opstal.T
        



    conditions = sorted([x for x in pd_van_opstal.T.index])
    for x_var,y_var in zip(conditions[::2],conditions[1::2]):
        pd_collapsed[x_var[0]] = pd_van_opstal[[x_var,y_var]].apply(tuple,axis=1)
    return pd_collapsed

def pandas_bootstrap(pd_dataframe,columns,alpha=0.05,
                     stat_func=bs_stats.mean,iters=10000):
    pd_filtered = pd_dataframe.filter(items=columns)
    pd_ci = pd.DataFrame(columns=["x","y","xerr","yerr"])

    for index, row in pd_filtered.iterrows():
        x = np.array([x[0] for x in row])
        y = np.array([x[1] for x in row])
        CI_x = bs.bootstrap(x,stat_func=stat_func,iteration_batch_size=iters,
                            alpha=alpha)
        CI_y = bs.bootstrap(y,stat_func=stat_func,iteration_batch_size=iters,
                            alpha=alpha)
        ci_tuple = (CI_x.value,CI_y.value,
                      (abs(CI_x.lower_bound-CI_x.value),
                       abs(CI_x.upper_bound-CI_x.value)),
                      (abs(CI_y.lower_bound-CI_y.value),
                       abs(CI_y.upper_bound-CI_y.value))
                     )
        pd_ci.loc[index] = ci_tuple
    return pd_ci

def make_van_opstal_paper_plot(pd_collapsed_col,marker="",fill_markers=True,color=None,
                              dashes=None):
    mfc = None if fill_markers else 'none'
    ms=8.5
    lw=3
    for counter in range(4):
        row = 4*counter
        col = counter
        for i in range(3):
            x_row,y_row = zip(pd_collapsed_col.iloc[row+i],pd_collapsed_col.iloc[row+i+1])
            if dashes is not None:
                plt.plot(x_row,y_row,marker=marker,mfc=mfc,color=color,dashes=dashes)
            else:
                plt.plot(x_row,y_row,marker=marker,mfc=mfc,color=color,lw=lw,ms=ms)
            x_col,y_col = zip(pd_collapsed_col.iloc[col+4*i],pd_collapsed_col.iloc[col+4*(i+1)])
            if dashes is not None:
                plt.plot(x_col,y_col,marker=marker,mfc=mfc,color=color,dashes=dashes)
            else:
                plt.plot(x_col,y_col,marker=marker,mfc=mfc,color=color,lw=lw,ms=ms)

def make_van_opstal_paper_plot_network_data(pd_data_cipic,
                                            azim_list=[-20,-10,10,20],
                                            elev_list=list(range(0,40,10)),
                                            marker="o",colors=None):
    pd_CI_means_by_cond = []
    if colors is None: colors = {"New Sets":'#0173b2',"Normal Set":'#de8f05'}
    for condtion in sorted(pd_data_cipic["Ears Used"].unique()):
        pd_CI = (pd_data_cipic[pd_data_cipic["Ears Used"]==condtion]
                                       .groupby(["azim","elev","Ears Used"])["predicted_azim","predicted_elev"]
                                       .agg(bootstrap_pandas_by_group)
                                       .reset_index())
        plt.clf()
        make_van_opstal_paper_plot_network_ref_grid()
        color = colors[condtion]
        for elev in elev_list:
            for azim in azim_list:
                query_string = ('azim == "{}" & elev=="{}"'.format(azim,elev))
                pd_CI_normal_filtered = pd_CI.query(query_string)
                x_mean,x_min,x_max = pd_CI_normal_filtered["predicted_azim"].iloc[0]
                xerr = np.array(x_min,x_max)
                y_mean,y_min,y_max = pd_CI_normal_filtered["predicted_elev"].iloc[0]
                yerr = np.array(y_min,y_max)
                plt.errorbar(x_mean,y_mean,xerr,yerr,marker=marker,mfc=color,mec=color,ecolor='darkgray')
        
        query_string = ('azim in @azim_list & elev in @elev_list'.format(azim,elev))
        pd_CI_normal_filtered = pd_CI.query(query_string)
        pd_CI_sorted = pd_CI_normal_filtered.sort_values(['elev','azim'],ascending=[False,True]) 
        pd_CI_means = pd_CI_sorted[['predicted_azim','predicted_elev']].apply(lambda x:(x['predicted_azim'][0],x['predicted_elev'][0]),axis=1)
        make_van_opstal_paper_plot(pd_CI_means,marker="",color=color)
        plt.ylim(-10,40)
        plt.xlim(-30,30)
        plt.yticks([0,10,20,30],size = 20)
        plt.xticks([-20,0,20],size = 20)
        plt.ylabel("Elevation (Degrees)",fontsize=20)
        plt.xlabel("Azimuth (Degrees)",fontsize=20)
        plt.tight_layout()
        pd_CI_means.to_csv(plots_folder+"/van_opstal_network_sem_{}.csv".format(condtion))
        plt.savefig(plots_folder+"/van_opstal_network_sem_{}.svg".format(condtion))

        pd_CI_means_by_cond.append(pd_CI_means)
    return pd_CI_means_by_cond

def make_van_opstal_paper_plot_network_ref_grid():
    data = [[(-20,30),(-10,30),(10,30),(20,30)],
            [(-20,20),(-10,20),(10,20),(20,20)],
            [(-20,10),(-10,10),(10,10),(20,10)],
            [(-20,0),(-10,0),(10,0),(20,0)]]
    data_np = np.array(data)
    x = data_np[:,:,0].flatten().tolist()
    y = data_np[:,:,1].flatten().tolist()
    d = {'x':x,'y':y}
    pd_refernce_grid = pd.DataFrame(columns=['x','y'],data=d)
    pd_refernce_grid['xy'] = list(zip(pd_refernce_grid.x, pd_refernce_grid.y))
    make_van_opstal_paper_plot(pd_refernce_grid['xy'],marker='o',dashes=[5, 10],color='k',fill_markers=False)
    return pd_refernce_grid[['x','y']]


def make_van_opstal_paper_plot_bootstrapped(pd_dataframe,columns):
    pd_bs = pandas_bootstrap(pd_dataframe,columns,alpha=0.314)
    xerr = pd.DataFrame(pd_bs['xerr'].apply(pd.Series)).values.T
    yerr = pd.DataFrame(pd_bs['yerr'].apply(pd.Series)).values.T
    plt.errorbar(pd_bs["x"],pd_bs["y"],xerr,yerr,marker="o",linestyle='',
                 color='k',ecolor='darkgray')
    pd_bs["xy"] = pd_bs[['x', 'y']].apply(tuple, axis=1)
    make_van_opstal_paper_plot(pd_bs["xy"],marker="",color='k')
    plt.ylim(-30,30)
    plt.xlim(-30,30)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.ylabel("Elevation (Degrees)",fontsize=20)
    plt.xlabel("Azimuth (Degrees)",fontsize=20)
    plt.tight_layout()
    return pd_bs

        

def format_van_opstal_human_dataframes(pd_van_opstal):
    pd_van_opstal = pd_van_opstal.convert_objects(convert_numeric=True) 
    pd_van_opstal["Reference"] = pd_van_opstal["Condition"].apply(lambda x: "Reference" in x) 
    pd_van_opstal["Ears Used"] = pd_van_opstal["Condition"].apply(lambda x:  "New Set" if "After" in x else "Normal Set") 
    if "X Presented" in pd_van_opstal:
        convert_idx_to_degrees = {0:-20,1:-6.666,2:6.666,3:20}
        pd_van_opstal["X Presented"] = pd_van_opstal["X Presented"].apply(lambda x:
                                                                          convert_idx_to_degrees[x])
        pd_van_opstal.rename({'X Presented':'Azimuth (Degrees)', 
                              'X Judged':'Judged Azimuth (Degrees)'},
                                                                axis=1,inplace=True)
    else:
        convert_idx_to_degrees = {0:20,1:6.666,2:-6.666,3:-20}
        pd_van_opstal["Y Presented"] = pd_van_opstal["Y Presented"].apply(lambda x:
                                                                          convert_idx_to_degrees[x])
        pd_van_opstal.rename({'Y Presented':'Elevation (Degrees)', 
                              'Y Judged':'Judged Elevation (Degrees)'},
                                                                axis=1,inplace=True)
    return pd_van_opstal

def get_elevation_error_CIPIC(pd_dataframe_cipic,subject=None):
    #convert datatypes
    pd_dataframe_cipic = pd_dataframe_cipic.convert_objects(convert_numeric=True)
    if pd_dataframe_cipic['azim'].dtype != 'int64':
        pd_dataframe_cipic['azim'] = pd_dataframe_cipic['azim'].apply(convert_from_numpy)
        pd_dataframe_cipic['elev'] = pd_dataframe_cipic['elev'].apply(convert_from_numpy)
    if subject is not None:
        subject_list = subject if isinstance(subject,list) else [subject]
        pd_dataframe_cipic_filtered =  pd_dataframe_cipic[pd_dataframe_cipic['subject_num'].isin(subject_list)]
    else:
        pd_dataframe_cipic_filtered = pd_dataframe_cipic
    #Get azim and elevation based on predicted index
    pd_dataframe_cipic_filtered['predicted_elev'] = None
    pd_dataframe_cipic_filtered['predicted_azim'] = None
    pd_dataframe_cipic_filtered['predicted_elev'] = pd_dataframe_cipic_filtered['predicted'].apply(lambda x: x//72)
    pd_dataframe_cipic_filtered['predicted_azim'] = pd_dataframe_cipic_filtered['predicted'].apply(lambda x: x%72)
    #Map elevation labels to indicies used for prediction for all subject data
    pd_dataframe_cipic_filtered['elev'] = pd_dataframe_cipic_filtered.apply(elevation_mapping,axis=1)
    pd_dataframe_cipic_filtered['azim'] = pd_dataframe_cipic_filtered.apply(azimuth_correction_hack,axis=1)
    #Front_back_folding
    pd_dataframe_cipic_filtered['predicted_azim'] = pd_dataframe_cipic_filtered['predicted_azim'].apply(CIPIC_azim_folding)
    pd_dataframe_cipic_filtered['azim'] = pd_dataframe_cipic_filtered['azim'].apply(CIPIC_azim_folding)
    #Change to degrees
    pd_dataframe_cipic_filtered['predicted_elev'] = pd_dataframe_cipic_filtered['predicted_elev'].apply(lambda x: x*10)
    pd_dataframe_cipic_filtered['elev'] = pd_dataframe_cipic_filtered['elev'].apply(lambda x: x*10)
    #claculate error
    pd_dataframe_cipic_filtered['elev_error'] = (pd_dataframe_cipic_filtered['predicted_elev'] -
                                                    pd_dataframe_cipic_filtered['elev']).abs()
    pd_dataframe_cipic_filtered['azim_error'] = (pd_dataframe_cipic_filtered['predicted_azim'] -
                                                    pd_dataframe_cipic_filtered['azim']).abs()
    #Add control variable to determine if it is from the KEMAR set or not
    pd_dataframe_cipic_filtered["Ears Used"] = "New Sets"
    pd_dataframe_cipic_filtered.loc[pd_dataframe_cipic_filtered["subject_num"].isin([999]),"Ears Used"] = "Normal Set"
    return pd_dataframe_cipic_filtered

def calculate_error_azims(pd_dataframe_azims):
    pd_dataframe_azims = pd_dataframe_azims.convert_objects(convert_numeric=True)
    if pd_dataframe_azims['azim'].dtype != 'int64' and pd_dataframe_azims['azim'].dtype != 'float64':
        pd_dataframe_azims['azim'] = pd_dataframe_azims['azim'].apply(convert_from_numpy)
    pd_dataframe_azims['predicted_azim'] = None
    pd_dataframe_azims['predicted_azim'] = pd_dataframe_azims['predicted'].apply(lambda x: x%72)
    pd_dataframe_azims['azim_folded'] = pd_dataframe_azims['azim'].apply(CIPIC_azim_folding)
    #Front_back_folding
    pd_dataframe_azims['predicted_azim_folded'] = pd_dataframe_azims['predicted_azim'].apply(CIPIC_azim_folding)
    pd_dataframe_azims['azim_error_folded'] = pd_dataframe_azims['azim_folded'] - pd_dataframe_azims['predicted_azim_folded']
    pd_dataframe_azims['azim_error_folded_abs'] = pd_dataframe_azims['azim_error_folded'].abs()
    return pd_dataframe_azims


def calculate_predictions_and_error_nsynth(pd_dataframe_nsynth):
    #convert datatypes
    if pd_dataframe_nsynth['azim'].dtype != 'int64':
        pd_dataframe_nsynth['azim'] = pd_dataframe_nsynth['azim'].apply(convert_from_numpy)
        pd_dataframe_nsynth['elev'] = pd_dataframe_nsynth['elev'].apply(convert_from_numpy)
        pd_dataframe_nsynth = pd_dataframe_nsynth.convert_objects(convert_numeric=True)
    #Get azim and elevation based on predicted index
    pd_dataframe_nsynth['predicted_elev'] = None
    pd_dataframe_nsynth['predicted_azim'] = None
    pd_dataframe_nsynth['predicted_elev'] = pd_dataframe_nsynth['predicted'].apply(lambda x: x//72)
    pd_dataframe_nsynth['predicted_azim'] = pd_dataframe_nsynth['predicted'].apply(lambda x: x%72)
    #Map elevation labels to indicies used for prediction for all subject data
    pd_dataframe_nsynth['elev'] = pd_dataframe_nsynth.apply(elevation_mapping_nsynth,axis=1)
    #Front_back_folding
    pd_dataframe_nsynth['predicted_azim_folded'] = pd_dataframe_nsynth['predicted_azim'].apply(CIPIC_azim_folding)
    pd_dataframe_nsynth['azim_folded'] = pd_dataframe_nsynth['azim'].apply(CIPIC_azim_folding)
    pd_dataframe_nsynth['azim_error_folded'] = pd_dataframe_nsynth['azim_folded'] - pd_dataframe_nsynth['predicted_azim_folded']
    pd_dataframe_nsynth['azim_error_folded_abs'] = pd_dataframe_nsynth['azim_error_folded'].abs()
    #Change to degrees
    pd_dataframe_nsynth['predicted_elev'] = pd_dataframe_nsynth['predicted_elev'].apply(lambda x: x*10)
    pd_dataframe_nsynth['elev'] = pd_dataframe_nsynth['elev'].apply(lambda x: x*10)
    pd_dataframe_nsynth['azim_error_abs'] = pd_dataframe_nsynth.apply(lambda x: azim_error_row(x),axis=1)
    pd_dataframe_nsynth['azim_error_abs'] = pd_dataframe_nsynth['azim_error_abs'].apply(lambda x: 10*x)
    pd_dataframe_nsynth['exemplar_name'] = pd_dataframe_nsynth['filename'].apply(get_exemplar_from_filename)
    return pd_dataframe_nsynth


def calculate_predictions_and_error_testset(pd_dataframe_testset):
    #convert datatypes
    pd_dataframe_testset = pd_dataframe_testset.convert_objects(convert_numeric=True)
    if pd_dataframe_testset['azim'].dtype != 'int64':
        pd_dataframe_testset['azim'] = pd_dataframe_testset['azim'].apply(convert_from_numpy)
        pd_dataframe_testset['elev'] = pd_dataframe_testset['elev'].apply(convert_from_numpy)
    #Get azim and elevation based on predicted index
    pd_dataframe_testset['predicted_elev'] = None
    pd_dataframe_testset['predicted_azim'] = None
    pd_dataframe_testset['predicted_elev'] = pd_dataframe_testset['predicted'].apply(lambda x: x//72)
    pd_dataframe_testset['predicted_azim'] = pd_dataframe_testset['predicted'].apply(lambda x: x%72)
    #Map elevation labels to indicies used for prediction for all subject data
    pd_dataframe_testset['elev'] = pd_dataframe_testset.apply(elevation_mapping_nsynth,axis=1)
    #Front_back_folding
    pd_dataframe_testset['predicted_azim_folded'] = pd_dataframe_testset['predicted_azim'].apply(CIPIC_azim_folding)
    pd_dataframe_testset['azim_folded'] = pd_dataframe_testset['azim'].apply(CIPIC_azim_folding)
    pd_dataframe_testset['azim_error_folded'] = pd_dataframe_testset['azim_folded'] - pd_dataframe_testset['predicted_azim_folded']
    pd_dataframe_testset['azim_error_folded_abs'] = pd_dataframe_testset['azim_error_folded'].abs()
    #Change to degrees
    pd_dataframe_testset['predicted_elev'] = pd_dataframe_testset['predicted_elev'].apply(lambda x: x*10)
    pd_dataframe_testset['elev'] = pd_dataframe_testset['elev'].apply(lambda x: x*10)
    pd_dataframe_testset['elev_error'] = pd_dataframe_testset['elev'] - pd_dataframe_testset['predicted_elev']
    pd_dataframe_testset['elev_error_abs'] = pd_dataframe_testset['elev_error'].abs()
    pd_dataframe_testset['azim_error_abs'] = pd_dataframe_testset.apply(lambda x: azim_error_row(x),axis=1)
    pd_dataframe_testset['azim_error_abs'] = pd_dataframe_testset['azim_error_abs'].apply(lambda x: 10*x)
    return pd_dataframe_testset
    
def get_error_smoothed_hrtf(pd_dataframe_smoothed_hrtf,subject=None):
    if pd_dataframe_smoothed_hrtf['azim'].dtype != 'int64':
        pd_dataframe_smoothed_hrtf['azim'] = pd_dataframe_smoothed_hrtf['azim'].apply(convert_from_numpy)
        pd_dataframe_smoothed_hrtf['elev'] = pd_dataframe_smoothed_hrtf['elev'].apply(convert_from_numpy)
        pd_dataframe_smoothed_hrtf = pd_dataframe_smoothed_hrtf.convert_objects(convert_numeric=True)
    if subject is not None:
        subject_list = subject if isinstance(subject,list) else [subject]
        pd_dataframe_smoothed_hrtf_filtered =  pd_dataframe_smoothed_hrtf[pd_dataframe_smoothed_hrtf['subject'].isin(subject_list)]
    else:
        pd_dataframe_smoothed_hrtf_filtered = pd_dataframe_smoothed_hrtf
    #Get azim and elevation based on predicted index
    pd_dataframe_smoothed_hrtf_filtered['predicted_elev'] = None
    pd_dataframe_smoothed_hrtf_filtered['predicted_azim'] = None
    pd_dataframe_smoothed_hrtf_filtered['predicted_elev'] = pd_dataframe_smoothed_hrtf_filtered['predicted'].apply(lambda x: x//72)
    pd_dataframe_smoothed_hrtf_filtered['predicted_azim'] = pd_dataframe_smoothed_hrtf_filtered['predicted'].apply(lambda x: x%72)
    #Front_back_folding
    pd_dataframe_smoothed_hrtf_filtered['predicted_azim'] = pd_dataframe_smoothed_hrtf_filtered['predicted_azim'].apply(CIPIC_azim_folding)
    pd_dataframe_smoothed_hrtf_filtered['azim'] = pd_dataframe_smoothed_hrtf_filtered['azim'].apply(CIPIC_azim_folding)
    #Change to degrees
    pd_dataframe_smoothed_hrtf_filtered['predicted_elev'] = pd_dataframe_smoothed_hrtf_filtered['predicted_elev'].apply(lambda x: x*10)
    pd_dataframe_smoothed_hrtf_filtered['elev'] = pd_dataframe_smoothed_hrtf_filtered['elev'].apply(lambda x: x*5)
    return pd_dataframe_smoothed_hrtf_filtered

def format_CIPIC_graph():
    xticks=plt.gca().xaxis.get_major_ticks()
    if len(xticks) > 10:
        for i in range(len(xticks)):
            if i%3 != 0:
                xticks[i].set_visible(False)
    plt.yticks(fontsize=40)
    plt.xticks(fontsize=40,rotation=45)
    plt.legend(fontsize=30,title ="Ears Used",title_fontsize=30)

def format_van_opstal_human_graph():
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30,rotation=45)
    plt.legend(fontsize=20)

def format_van_opstal_elevation():
    plt.ylim(-25,25)
    plt.ylabel("Judged Elevation (Degrees)", fontsize=30)
    plt.xlabel("Elevation (Degrees)", fontsize=30)
    format_van_opstal_human_graph()
    plt.tight_layout()


def format_van_opstal_azim():
    plt.ylim(-30,30)
    plt.ylabel("Judged Azimuth (Degrees)", fontsize=30)
    plt.xlabel("Azimuth (Degrees)", fontsize=30)
    format_van_opstal_human_graph()
    plt.tight_layout()

def make_van_opstal_human_plots():
    pd_van_opstal_human = pd.read_pickle("/om/user/francl/pd_van_opstal_human_data_extracted.pkl")
    df_x,df_y = collapse_van_opstal_data_human(pd_van_opstal_human)
    df_x = format_van_opstal_human_dataframes(df_x)  
    df_y = format_van_opstal_human_dataframes(df_y)  
    plt.clf()
    sns.lineplot(x='Azimuth (Degrees)',y='Judged Azimuth (Degrees)',hue="Ears Used",
                units="Condition",estimator=None, lw=4,ms=10,
                data=df_x.query('Reference == False'),style="Ears Used",
                markers=["o","o"],dashes=False,palette=reversed(sns.color_palette
                                                                      (n_colors=2)))
    format_van_opstal_azim()
    plt.savefig(plots_folder+"/"+"van_opstal_human_data_azimuth_before_after_pointplot_individual_subjects.png")
    plt.clf()
    sns.lineplot(x='Elevation (Degrees)',y='Judged Elevation (Degrees)',hue="Ears Used",
                estimator=None, lw=4,ms=10,
                data=df_x.query('Reference == False'),style="Ears Used",
                markers=["o","o"],dashes=False,palette=reversed(sns.color_palette
                                                                      (n_colors=2)))
    format_van_opstal_elevation()
    plt.savefig(plots_folder+"/"+"van_opstal_human_data_elevation_before_after_pointplot_individual_subjects.png")


def elevation_mapping(row):
    if row['subject_num'] == 999:
        elev_mapped = row['elev']//2
    else:
        CIPIC_mapping = {10:5,7:4,5:3,4:2,2:1,0:0}
        elev_mapped = CIPIC_mapping[row['elev']]
    return elev_mapped

def azimuth_correction_hack(row):
    if row['subject_num'] == 999:
        azimuth_hack_dict = {(5,5):10,(5,175):170,(5,335):340,(5,165):160,(4,15):20,(4,195):200,
                             (3,350):355,(3,345):350,(3,20):25,(3,15):20,(3,170):175,(3,165):170,
                             (3,200):205,(3,195):200}
        try:
            azim_deg = row['azim']*5
            azim = azimuth_hack_dict[(row['elev'],azim_deg)]
            azim = int(azim/5)
        except:
            azim = row['azim']
        return azim
    else:
        return row['azim']

def elevation_mapping_nsynth(row):
    elev_mapped = row['elev']//2
    return elev_mapped

def get_exemplar_from_filename(filename):
    #parses the non-spatialized nsynth stimuli name from the full filename
    return "_".join(str(filename).split("/")[-1].split("_")[:3])

def CIPIC_azim_folding(azim):
    folded_idx= get_folded_label_idx_5deg(azim)
    folded_degrees = 90-folded_idx*5
    return folded_degrees

    
def compare_smoothed_hrtf(pd_dataframe_smoothed_hrtf_filtered,elev_to_test=None,max_factor=256):
    pd_dataframe_smoothed_hrtf_filtered = pd_dataframe_smoothed_hrtf_filtered.astype({"arch_index":'int32','noise_idx':'int32'})
    cols = ["Azimuth","Elevation","Azimuth Error",
            "Elevation Error","Smooth Factor",
            "Architecture Number", "Noise Index"]
    pd_smoothed_hrir_mismatch = pd.DataFrame(columns=cols)
    azim_error_fn = lambda x, y: min(360-(abs(x-y)),abs(x-y))
    if elev_to_test is None:
        elev_to_test = pd_dataframe_smoothed_hrtf_filtered['elev'].unique()
    for elev in elev_to_test:
        for azim in pd_dataframe_smoothed_hrtf_filtered['azim'].unique():
            for arch_idx in pd_dataframe_smoothed_hrtf_filtered['arch_index'].unique():
                for noise_idx in pd_dataframe_smoothed_hrtf_filtered['noise_idx'].unique():
                    query_string  =('azim == "{}" & elev=="{}" & arch_index == "{}" & noise_idx == "{}"'.format(azim,elev,arch_idx,noise_idx))
                    same_position = pd_dataframe_smoothed_hrtf_filtered.query(query_string)
                    full_condition = same_position['smooth_factor'].max() 
                    #if full_condition != max_factor:
                        #print(azim,elev,arch_idx,noise_idx)
                    full_row = same_position[same_position['smooth_factor'] ==
                                             full_condition]
                    full_row_azim_pred = full_row['predicted_azim']
                    full_row_elev_pred = full_row['predicted_elev']
                    for index,example in same_position.iterrows():
                        example_azim_pred = example['predicted_azim']
                        example_elev_pred = example['predicted_elev']
                        azim_error = full_row_azim_pred.apply(lambda x: azim_error_fn(x,example_azim_pred)).mean()
                        elev_error = abs(full_row_elev_pred-example_elev_pred).mean()
                        pd_smoothed_hrir_mismatch = pd_smoothed_hrir_mismatch.append({"Azimuth":azim,"Elevation":elev,
                                                          "Azimuth Error":azim_error,
                                                          "Elevation Error":elev_error,
                                                          "Smooth Factor":example['smooth_factor'],
                                                          "Architecture Number": arch_idx,
                                                          "Noise Index":noise_idx},
                                                         ignore_index=True)
    return pd_smoothed_hrir_mismatch

def plot_kulkarni_networks(pd_smoothed_hrir_mismatch):
    plt.figure(figsize=(10,10),dpi=200)
    plt.clf()
    order=[1024,512,256,128,64,32,16,8,4,2,1] 
    pd_smoothed_hrir_mismatch_arch_mean = pd_smoothed_hrir_mismatch.groupby(["Smooth Factor","Architecture Number"]).mean().reset_index()
    sns.pointplot(x="Smooth Factor",y="Total Error",
                  order=order,color='k',markers=[""],
                  data=pd_smoothed_hrir_mismatch_arch_mean)
    plt.xticks(range(len(order)),order,fontsize=30,rotation=45)
    plt.ylabel("Spatial Error (Degrees)",fontsize=30)
    plt.xlabel("Smooth Factor",fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    pd_smoothed_hrir_mismatch_arch_mean.to_csv(plots_folder+"/"+"smooth_factor_vs_total_error_smoothed_fill_length_hrir_mathched_ylim.csv")
    plt.savefig(plots_folder+"/"+"smooth_factor_vs_total_error_smoothed_fill_length_hrir_mathched_ylim.svg")


azim_filter_list = [140,145,150,155,160,170,175,180,40,35,30,25,20,15,10,5,0,355,350,345,340,335,330,325,320,315,310,305,300,295,290,185,190,195,200,205,210,215,220]
                
def format_middlebrooks_wrightman_dataframe(pd_dataframe_middlebrooks_wrightman,
                                            azim_filter_list=list(range(0,360,5))):
    if pd_dataframe_middlebrooks_wrightman['azim'].dtype != 'int64':
        pd_dataframe_middlebrooks_wrightman['azim'] = pd_dataframe_middlebrooks_wrightman['azim'].apply(convert_from_numpy)
        pd_dataframe_middlebrooks_wrightman['elev'] = pd_dataframe_middlebrooks_wrightman['elev'].apply(convert_from_numpy)
        pd_dataframe_middlebrooks_wrightman = pd_dataframe_middlebrooks_wrightman.convert_objects(convert_numeric=True)
    if pd_dataframe_middlebrooks_wrightman['azim'].max() < 72:
        pd_dataframe_middlebrooks_wrightman['azim'] = pd_dataframe_middlebrooks_wrightman['azim'].apply(lambda x: x*5)
        pd_dataframe_middlebrooks_wrightman['predicted_folded'] = pd_dataframe_middlebrooks_wrightman['predicted_folded'].apply(lambda x: x*5)
        pd_dataframe_middlebrooks_wrightman['predicted_folded'] = pd_dataframe_middlebrooks_wrightman['predicted_folded'].apply(lambda x: 90-x)
    #pd_dataframe_middlebrooks_wrightman_filtered = pd_dataframe_middlebrooks_wrightman[pd_dataframe_middlebrooks_wrightman['azim'].isin([70,65,60,55,50,45,40,35,30,25,20,15,10,5,0,355,350,345,340,335,330,325,320,315,310,305,300,295,290])]
    pd_dataframe_middlebrooks_wrightman_filtered = pd_dataframe_middlebrooks_wrightman[pd_dataframe_middlebrooks_wrightman['azim'].isin(azim_filter_list)]
    #pd_dataframe_middlebrooks_wrightman_filtered = pd_dataframe_middlebrooks_wrightman[pd_dataframe_middlebrooks_wrightman['azim'].isin([40,35,30,25,20,15,10,5,0,355,350,345,340,335,330,325,320])]
    pd_dataframe_middlebrooks_wrightman_mean_groups = pd_dataframe_middlebrooks_wrightman_filtered.groupby(['azim','ITD','ILD',
                                                    'low_cutoff','high_cutoff','arch_index'])['predicted_folded'].agg([np.mean,'count','std']).reset_index()
    pd_dataframe_middlebrooks_wrightman_mean_groups['low_high_cutoff'] = pd_dataframe_middlebrooks_wrightman_mean_groups[['low_cutoff','high_cutoff']].apply(lambda x: str(tuple(x)),axis=1)
    return pd_dataframe_middlebrooks_wrightman_mean_groups

def make_ILD_graph(pd_dataframe_middlebrooks_wrightman_formatted):
    pd_dataframe_middlebrooks_wrightman_ITD_filtered = pd_dataframe_middlebrooks_wrightman_formatted[pd_dataframe_middlebrooks_wrightman_formatted['ITD'].isin([0])]
    pd_unique_azims = pd_dataframe_middlebrooks_wrightman_ITD_filtered['azim'].unique()
    for azim in pd_unique_azims:
        pd_dataframe_middlebrooks_wrightman_azim_filtered=pd_dataframe_middlebrooks_wrightman_ITD_filtered[pd_dataframe_middlebrooks_wrightman_ITD_filtered['azim']== azim]
        pd_dataframe_middlebrooks_wrightman_std_filtered=pd_dataframe_middlebrooks_wrightman_azim_filtered[pd_dataframe_middlebrooks_wrightman_azim_filtered['std']<= 7]
        plt.clf()
        sns.pointplot(x='ILD',y='mean',hue='low_high_cutoff',data=pd_dataframe_middlebrooks_wrightman_std_filtered)
        plt.ylim(-90,90)
        plt.title("ILD sensititivity at {} Azim".format(azim))
        plt.savefig(plots_folder+"/"+"ILD_sensitivity_vs_angle_at_{}_azim.png".format(azim))


def weighted_mean_index(azim,dataframe,low_cutoff=False,ITD_version=False):
    if azim < 0:
        azim = abs(azim)
        ILD_direction_weight = -1.0
    else:
        ILD_direction_weight = 1.0
    distance = (dataframe['Azim']-azim).abs()
    best_positions = distance.argsort()[:2]
    weights = (10-distance[best_positions]).abs()/10
    dataframe_slice = dataframe.iloc[best_positions]
    if ITD_version:
        ILD_value = (dataframe_slice["ITD"]*weights).sum()
    elif low_cutoff:
        ILD_value = (dataframe_slice["ILD_low_freq"]*weights).sum()
    else:
        ILD_value = (dataframe_slice["ILD_high_freq"]*weights).sum()
    ILD_value = ILD_direction_weight*ILD_value
    return ILD_value

def calculate_residual(row,pd_azim_averages,pd_ILD_list,ITD_version=False):
    low_cutoff = row['high_cutoff'] <= 2000 
    original_location = pd_azim_averages.query("azim=={} & low_high_cutoff=='{}'".format(row['azim'],row['low_high_cutoff']))['mean'].iloc[0]
    original_ILD = weighted_mean_index(original_location,pd_ILD_list,low_cutoff,ITD_version)
    observed_ILD = weighted_mean_index(row['mean'],pd_ILD_list,low_cutoff,ITD_version)
    residual = observed_ILD - original_ILD
    return float(residual)

def filter_ITD_values(row):
    azim = row['azim']
    ITD = row['ITD']
    if abs(ITD) == 9 or abs(ITD) == 0: return True
    if abs(ITD) == 22 and (azim <= 20 or azim >= 340 or 160 <= azim <= 200):
        return True
    if abs(ITD) == 18 and (azim <= 40 or azim >= 320 or 140 <= azim <= 220):
        return True
    return False

def filter_ILD_values(row):
    azim = row['azim']
    ILD = row['ILD']
    if ILD == 0: return True
    if abs(ILD) == 20 and (azim <= 20 or azim >= 340 or 160 <= azim <= 200):
        return True
    if abs(ILD) == 10 and (azim <= 40 or azim >= 320 or 140 <= azim <= 220):
        return True
    return False

def filter_ILD_values(row):
    azim = row['azim']
    ILD = row['ILD']
    if ILD == 0: return True
    if (ILD == -20 and (azim > 20 and azim < 160)) or (ILD == 20 and (azim < 340 and azim > 200)):
        return False
    if (ILD == -10 and (azim > 40 and azim < 140)) or (ILD == 10 and (azim < 320 and azim > 220)):
        return False
    return True

def filter_ITD_values(row):
    azim = row['azim']
    ITD = row['ITD']
    if (ITD <= -22 and (azim > 20 and azim < 160)) or (ITD >= 22 and (azim < 340 and azim > 200)):
        return False
    if (ITD <= -18 and (azim > 40 and azim < 140)) or (ITD >= 18 and (azim < 320 and azim > 220)):
        return False
    return True

def filter_ITD_values(row):
    #backwards ITD
    azim = row['azim']
    ITD = row['ITD']
    if (ITD >= 22 and (azim > 20 and azim < 160)) or (ITD <= -22 and (azim < 340 and azim > 200)):
        return False
    if (ITD >= 18 and (azim > 40 and azim < 140)) or (ITD <= -18 and (azim < 320 and azim > 220)):
        return False
    return True

#def filter_ILD_values(row):
#    azim = row['azim']
#    ILD = row['ILD']
#    if abs(ILD) != 20: return True
#    if ILD == -20 and (azim <= 20 or 160 <= azim <= 180):
#        return False
#    if ILD == 20 and (azim >= 340 or 180 <= azim <= 200):
#        return False
#    return True

def ILD_residuals_process(pd_dataframe_middlebrooks_wrightman_formatted):
    pd_dataframe_middlebrooks_wrightman_ITD_filtered = pd_dataframe_middlebrooks_wrightman_formatted[pd_dataframe_middlebrooks_wrightman_formatted['ITD'].isin([0])]
    pd_dataframe_middlebrooks_wrightman_std_filtered=pd_dataframe_middlebrooks_wrightman_ITD_filtered[pd_dataframe_middlebrooks_wrightman_ITD_filtered['std']<=14]
    pd_unique_azims = pd_dataframe_middlebrooks_wrightman_std_filtered['azim'].unique()
    pd_azim_averages = pd_dataframe_middlebrooks_wrightman_std_filtered.query("ILD==0").groupby(['azim','low_high_cutoff'])['mean'].mean().reset_index()
    #filters out azimuth/bandwidth pairs where the average condtion had too
    #high of variance to safely use
    pd_azim_bandwith_filter_bool = pd_dataframe_middlebrooks_wrightman_std_filtered\
            .set_index(['azim','low_high_cutoff']).index.\
            isin(pd_azim_averages.set_index(['azim','low_high_cutoff']).index)
    pd_dataframe_middlebrooks_wrightman_std_filtered  =\
            pd_dataframe_middlebrooks_wrightman_std_filtered[pd_azim_bandwith_filter_bool]
    pd_ILD_list = dataframe_from_pickle("/om/user/francl/HRTF_freq_specific_ILDs.pkl")
    pd_dataframe_middlebrooks_wrightman_std_filtered['ILD Observed Bias (dB)'] = \
            pd_dataframe_middlebrooks_wrightman_std_filtered.apply(lambda x: calculate_residual(x,pd_azim_averages,pd_ILD_list),axis=1)
    ILD_filtering_bool = pd_dataframe_middlebrooks_wrightman_std_filtered.apply(filter_ILD_values,axis=1)
    pd_dataframe_middlebrooks_wrightman_ILD_filtered=pd_dataframe_middlebrooks_wrightman_std_filtered[ILD_filtering_bool]
    pd_dataframe_middlebrooks_wrightman_ILD_filtered['ILD'] = pd_dataframe_middlebrooks_wrightman_ILD_filtered['ILD'].apply(invert)
    pd_dataframe_middlebrooks_wrightman_ILD_filtered.rename({'ILD':'ILD (dB)'},
                                                            axis=1,inplace=True)
    pd_dataframe_to_plot = pd_dataframe_middlebrooks_wrightman_ILD_filtered.query('low_cutoff == 4000 or'
                                                                                  ' high_cutoff == 2000')
    return pd_dataframe_to_plot

def ILD_residuals_plot(pd_dataframe_to_plot,extract_lines=False):
    title_dict = {'(500, 2000)':"Low Pass", '(4000, 16000)':"High Pass"}
    extracted_data = []
    for low_high_cutoff in sorted(pd_dataframe_to_plot['low_high_cutoff'].unique()):
        pd_dataframe_to_plot_filtered = pd_dataframe_to_plot[pd_dataframe_to_plot['low_high_cutoff'] ==
                                                             low_high_cutoff]
        plt.clf()
        plt.figure(figsize=(10,10),dpi=200)
        sns.set_style("ticks")
        with sns.axes_style("white"):
            sns.regplot(x='ILD (dB)',y='ILD Observed Bias (dB)',
                       data=pd_dataframe_to_plot_filtered,x_estimator=np.mean,
                       color="black")
        #sns.pointplot(x='ILD',y='ILD Bias Observed',hue='low_high_cutoff',
        #              data=pd_dataframe_to_plot)
        plt.ylim(-35,35)
        plt.xlim(-35,35)
        plt.tick_params(bottom=True,top=True,left=True,right=True,
                        labelbottom=True,labelleft=True)
        plt.xticks([-20,-10,0,10,20],rotation=45)
        plt.xlabel("Imposed ILD Bias (dB)")
        plt.ylabel("Observed ILD Bias (dB)")
        get_correlations_coefficients()
        #plt.title("ILD Added vs ILD Observed: {}".format(title_dict[low_high_cutoff]),fontsize=35,pad=30,x=.4)
        plt.text(0.25,1.05,"ILD bias: {}".format(title_dict[low_high_cutoff]),fontsize=25,transform=plt.gca().transAxes)
        plt.gca().set_aspect(1)
        plt.tight_layout(1.2)
        pd_out = (pd_dataframe_to_plot_filtered
                  .groupby(["ILD (dB)","arch_index"])['ILD Observed Bias (dB)']
                  .mean()
                  .reset_index())
        pd_out.to_csv(plots_folder+"/"+"ILD_residuals_regression_{}.csv".format(low_high_cutoff))
        plt.savefig(plots_folder+"/"+"ILD_residuals_regression_{}.svg".format(low_high_cutoff))
        if extract_lines:
            pd_data = get_data_from_graph(use_points=True)
            extracted_data.append(pd_data)
    if extract_lines:
        return extracted_data

            

def get_correlations_coefficients():
    lines = plt.gca().get_lines()
    handles,labels = plt.gca().get_legend_handles_labels()
    stats_list = []
    for idx,line in enumerate(lines):
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        if len(x_data) < 10: continue
        slope, intercept, r_value, p_value, std_err = stats.linregress(x=x_data,
                                                                       y=y_data)
        x0 = x_data[idx]
        y0 = y_data[idx]
        x1 = 0.5*min(x_data)
        y1 = 1.2*min(x_data)
        ann = plt.annotate("Bias Weight (Slope) = {}".format(round(slope,2)), xy=(x0, y0),
                           xytext=(x1, y1),fontsize=25)
        stats_list.append([idx,line,max(y_data),max(x_data),slope,ann])
    return stats_list



def ITD_residuals_process(pd_dataframe_middlebrooks_wrightman_formatted,pd_ITD_array=None):
    if pd_ITD_array is None:
        pd_ITD_array = dataframe_from_pickle("/om/user/francl/HRTF_ITDs.pkl")
    pd_dataframe_middlebrooks_wrightman_ILD_filtered = pd_dataframe_middlebrooks_wrightman_formatted[pd_dataframe_middlebrooks_wrightman_formatted['ILD'].isin([0])]
    pd_dataframe_middlebrooks_wrightman_std_filtered=pd_dataframe_middlebrooks_wrightman_ILD_filtered[pd_dataframe_middlebrooks_wrightman_ILD_filtered['std']<= 10]
    pd_unique_azims = pd_dataframe_middlebrooks_wrightman_std_filtered['azim'].unique()
    pd_azim_averages = pd_dataframe_middlebrooks_wrightman_std_filtered.query("ITD==0").groupby(['azim','low_high_cutoff'])['mean'].mean().reset_index()
    pd_azim_bandwith_filter_bool = pd_dataframe_middlebrooks_wrightman_std_filtered\
            .set_index(['azim','low_high_cutoff']).index.\
            isin(pd_azim_averages.set_index(['azim','low_high_cutoff']).index)
    pd_dataframe_middlebrooks_wrightman_std_filtered  =\
            pd_dataframe_middlebrooks_wrightman_std_filtered[pd_azim_bandwith_filter_bool]
    ITD_filtering_bool = pd_dataframe_middlebrooks_wrightman_std_filtered.apply(filter_ITD_values,axis=1)
    pd_dataframe_middlebrooks_wrightman_ITD_filtered=pd_dataframe_middlebrooks_wrightman_std_filtered[ITD_filtering_bool]
    pd_dataframe_middlebrooks_wrightman_ITD_filtered['ITD Observed Bias (us)'] = pd_dataframe_middlebrooks_wrightman_ITD_filtered.apply(lambda x: calculate_residual(x,pd_azim_averages,pd_ITD_array,ITD_version=True),axis=1)
    pd_dataframe_middlebrooks_wrightman_ITD_filtered['ITD'] = pd_dataframe_middlebrooks_wrightman_ITD_filtered['ITD'].apply(lambda x: round(x*(1000/44.1),-2))
    pd_dataframe_middlebrooks_wrightman_ITD_filtered.rename({'ITD':'ITD (us)'},
                                                            axis=1,inplace=True)
    pd_dataframe_to_plot = pd_dataframe_middlebrooks_wrightman_ITD_filtered.query('low_cutoff == 4000 or'
                                                                                  ' high_cutoff == 2000')
    return pd_dataframe_to_plot

def ITD_residuals_plot(pd_dataframe_to_plot,extract_lines=False):
    title_dict = {'(500, 2000)':"Low Pass", '(4000, 16000)':"High Pass"}
    extracted_data = []
    for low_high_cutoff in sorted(pd_dataframe_to_plot['low_high_cutoff'].unique()):
        pd_dataframe_to_plot_filtered = pd_dataframe_to_plot[pd_dataframe_to_plot['low_high_cutoff'] ==
                                                             low_high_cutoff]
        plt.clf()
        plt.figure(figsize=(10,10),dpi=200)
        sns.set_style("ticks")
        with sns.axes_style("white"):
            sns.regplot(x='ITD (us)',y='ITD Observed Bias (us)',
                       data=pd_dataframe_to_plot_filtered,x_estimator=np.mean,
                       color="black")
        #sns.pointplot(x='ITD',y='azim_change',hue='low_high_cutoff',
        #              data=pd_dataframe_to_plot)
        plt.ylim(-1000,1000)
        plt.xlim(-1000,1000)
        plt.tick_params(bottom=True,top=True,left=True,right=True,
                        labelbottom=True,labelleft=True)
        plt.yticks([-900,-600,-300,0,300,600,900])
        plt.xticks([-600,-300,0,300,600],rotation=45)
        plt.xlabel("Imposed ITD Bias (us)")
        plt.ylabel("Observed ITD Bias (us)")
        get_correlations_coefficients()
        #plt.title("ITD Added vs ITD Observed: {}".format(title_dict[low_high_cutoff]),fontsize=35,pad=30,x=0.4)
        plt.text(0.25,1.05,"ITD bias: {}".format(title_dict[low_high_cutoff]),fontsize=25,transform=plt.gca().transAxes)
        plt.gca().set_aspect(1)
        plt.tight_layout(1.2)
        pd_out = (pd_dataframe_to_plot_filtered
                  .groupby(["ITD (us)","arch_index"])['ITD Observed Bias (us)']
                  .mean()
                  .reset_index())
        pd_out.to_csv(plots_folder+"/"+"ITD_residuals_regression_{}.csv".format(low_high_cutoff))
        plt.savefig(plots_folder+"/"+"ITD_residuals_regression_{}.svg".format(low_high_cutoff))
        if extract_lines:
            pd_data = get_data_from_graph(use_points=True)
            extracted_data.append(pd_data)
    if extract_lines:
        return extracted_data

def make_ITD_graph(pd_dataframe_middlebrooks_wrightman_formatted):
    pd_dataframe_middlebrooks_wrightman_ITD_filtered = pd_dataframe_middlebrooks_wrightman_formatted[pd_dataframe_middlebrooks_wrightman_formatted['ILD'].isin([0])]
    pd_unique_azims = pd_dataframe_middlebrooks_wrightman_ITD_filtered['azim'].unique()
    for azim in pd_unique_azims:
        pd_dataframe_middlebrooks_wrightman_azim_filtered=pd_dataframe_middlebrooks_wrightman_ITD_filtered[pd_dataframe_middlebrooks_wrightman_ITD_filtered['azim']== azim]
        pd_dataframe_middlebrooks_wrightman_std_filtered=pd_dataframe_middlebrooks_wrightman_azim_filtered[pd_dataframe_middlebrooks_wrightman_azim_filtered['std']<= 15]
        plt.clf()
        sns.pointplot(x='ITD',y='mean',hue='low_high_cutoff',data=pd_dataframe_middlebrooks_wrightman_std_filtered)
        plt.title("ITD sensititivity at {} Azim".format(azim))
        plt.ylim(-90,90)
        plt.savefig(plots_folder+"/"+"ITD_sensitivity_vs_angle_at_{}_azim.png".format(azim))


def format_precedence_effect_dataframe(pd_dataframe_precedence):
    pd_dataframe_precedence['predicted_folded'] = pd_dataframe_precedence['predicted_folded'].apply(lambda x: x*5)
    pd_dataframe_precedence['predicted_folded'] = pd_dataframe_precedence['predicted_folded'].apply(lambda x: x 
                                                                                                    if x < 180 else x-360)
    pd_dataframe_precedence.columns = ['predicted', 'delay', 'start_sample',
                                       'Lead Click Level Difference (dB)',
                                       'lag_level','Lead Click Position',
                                       'arch_index', 'init_index',
                                       'predicted_folded']
    pd_dataframe_precedence['Lead Click Level Difference (dB)'] = pd_dataframe_precedence['lag_level'].apply(lambda x:
                                                                                       {'-10.0':0,'-12.5':5,'-7.5':-5,
                                                                                        '-20.0':20,'-15.0':10,'-2.5':-15,
                                                                                        '-17.5':15,'-5.0':-10,'0.0':-20}[str(x)])
    pd_dataframe_precedence['Lead Click Position'] = pd_dataframe_precedence['Lead Click Position'].apply(lambda x: -45 if x==0
                                                                                                          else 0)
    pd_dataframe_precedence['predicted_folded'] = pd_dataframe_precedence['predicted_folded'].apply(lambda x: x*-1)
    return pd_dataframe_precedence

def format_precedence_effect_dataframe_multi_azim(pd_dataframe_precedence):
    pd_dataframe_precedence = pd_dataframe_precedence.convert_objects(convert_numeric=True)
    if pd_dataframe_precedence['azim'].dtype != 'int64':
        pd_dataframe_precedence['azim'] = pd_dataframe_precedence['azim'].apply(convert_from_numpy)
    pd_dataframe_precedence['predicted_folded'] = pd_dataframe_precedence['predicted_folded'].apply(lambda x: x*5)
    pd_dataframe_precedence['predicted_folded'] = pd_dataframe_precedence['predicted_folded'].apply(lambda x: x 
                                                                                                    if x < 180 else x-360)
    pd_dataframe_precedence.columns = ['predicted', 'delay', 'azim','start_sample',
                                       'Lead Click Level Difference (dB)',
                                       'lag_level','flipped',
                                       'arch_index', 'init_index',
                                       'predicted_folded']
    pd_dataframe_precedence['Lead Click Level Difference (dB)'] = pd_dataframe_precedence['lag_level'].apply(lambda x:
                                                                                       {'-10.0':0,'-12.5':5,'-7.5':-5,
                                                                                        '-20.0':20,'-15.0':10,'-2.5':-15,
                                                                                        '-17.5':15,'-5.0':-10,'0.0':-20}[str(x)])
    pd_dataframe_precedence['Lead Click Position'] = pd_dataframe_precedence.apply(lambda x: x['azim'] if x['flipped']==0 else 0, axis=1)
    #pd_dataframe_precedence['predicted_folded'] = pd_dataframe_precedence['predicted_folded'].apply(lambda x: x*-1)
    return pd_dataframe_precedence

def format_precedence_effect_graph():
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.ylabel("Predicted Location",fontsize=20)
    plt.xlabel("Lead Time of 45 degree click (ms)",fontsize=20)
    plt.ylim([-50,50])
    plt.axhline(45,color='k',linestyle='dashed',alpha=0.5)
    plt.axhline(-45,color='k',linestyle='dashed',alpha=0.5)

def format_precedence_effect_graph_multi_azim():
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.ylabel("Predicted Location",fontsize=20)
    plt.xlabel("Lead Time of First Click (ms)",fontsize=20)
    plt.ylim([-70,70])

def get_power_spectrum(p_spec_full,center_freqs):
    assert center_freqs.shape[0] == p_spec_full.shape[0]
    p_spec_avg = np.abs(p_spec_full).mean(axis=1)
    p_spec_avg_flattened = p_spec_avg.T.flatten()
    p_spec_avg_flattened_dB = 20*np.log10(p_spec_avg_flattened)
    center_freq_col = np.concatenate([center_freqs,center_freqs])
    channel_col = np.squeeze(np.array([["Left"]*len(center_freqs) +
                           ["Right"]*len(center_freqs)]))
    cols = ["Filter Center Frequency","Average Cochlear Magnitude","Channel"]
    np_all_data = np.stack((center_freq_col,p_spec_avg_flattened,
                            channel_col),axis=1)
    df = pd.DataFrame(data=np_all_data,columns=cols)
    df = df.astype({"Filter Center Frequency":"float32", 
                    "Average Cochlear Magnitude":"float32"})
    return df
        

low_ci = lambda row: row['mean'] - 1.96*row['std']/math.sqrt(row['count'])
high_ci = lambda row: row['mean'] + 1.96*row['std']/math.sqrt(row['count'])

def format_binaural_test_set_frontback_folded(pd_dataframe_folded):
    pd_dataframe_folded['azim_folded'] = pd_dataframe_folded['azim'].apply(CIPIC_azim_folding)
    pd_dataframe_folded['predicted_folded'] = pd_dataframe_folded['predicted_folded'].apply(lambda x: 90-x*5)
    pd_dataframe_folded.rename({'azim_folded':'Actual Position (Degrees)',
                                'predicted_folded': 'Predicted Position (Degrees)'},
                                                            axis=1,inplace=True)
    return pd_dataframe_folded

def format_binaural_test_set(pd_dataframe_folded):
    center_at_zero = lambda x: -(5*x) if x < 36 else 360-5*x
    pd_dataframe_folded['azim_centered'] = pd_dataframe_folded['azim'].apply(center_at_zero)
    pd_dataframe_folded['predicted_centered'] = pd_dataframe_folded['predicted'].apply(center_at_zero)
    return pd_dataframe_folded

def format_binarural_test_folded_graph():
    plt.ylim(-105,105)
    plt.ylabel("Judged Position (Degrees)",fontsize=30)
    plt.xlabel("Actual Position (Degrees)",fontsize=30)
    plt.xticks(size = 30)
    plt.yticks([-90,-60,-30,0,30,60,90],size = 30)
    plt.tight_layout()

def get_grayscale_colormap(N=12):
    cmap =  LinearSegmentedColormap.from_list("my_colormap", ((0, 0, 0), (1, 1, 1),(0 ,0 ,0)), N=N, gamma=1.0)
    rgb_cmap = [cmap(i) for i in range(cmap.N)]
    return rgb_cmap

def fig_2c():
    pd_dataframe_folded = make_dataframe(("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                                          "arch_number_[0-9][0,1,3-9][0-9]*init_0/"
                                          "batch_conditional_*binaural_recorded_testset"
                                          "_4078_main_kemar_0elev_*iter100000*"),
                                         fold_data=True,recenter_folded_data=False)
    format_binaural_test_set_frontback_folded(pd_dataframe_folded)
    rgb_cmap = get_grayscale_colormap()
    plt.clf()
    sns.catplot(kind="violin",x="Actual Position (Degrees)",
                data=pd_dataframe_folded.astype('int32'),palette=rgb_cmap)
    format_binarural_test_folded_graph()
    plt.tight_layout()
    pd_dataframe_folded.to_csv(plots_folder+"/"+("binaural_recorded_4078_main_kemar_full_spec_folded"
                                                  "_violinplot_quartile_grayscale_front_limited.csv"))
    plt.savefig(plots_folder+"/"+("binaural_recorded_4078_main_kemar_full_spec_folded"
                                  "_violinplot_quartile_grayscale_front_limited.svg"))

def fig_2d():
    pd_dataframe_folded = make_dataframe(("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                                          "arch_number_[0-9][0,1,3-9][0-9]*init_0/"
                                          "batch_conditional_*binaural_recorded_testset"
                                          "_4078_main_kemar_0elev_*iter100000*"),
                                         fold_data=True,recenter_folded_data=False)
    format_binaural_test_set(pd_dataframe_folded)
    rgb_cmap = get_grayscale_colormap()
    plt.clf()
    sns.catplot(kind="violin",x="azim_centered",
                y="predicted_centered", height=8,
                aspect=1.45,inner='quartile',
                data=pd_dataframe_folded.astype('int32'),palette=rgb_cmap)
    plt.yticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180])
    plt.ylim(-190,190)
    plt.tight_layout()
    pd_dataframe_folded.to_csv(plots_folder+"/"+("binaural_recorded_4078_main_kemar_full_spec_folded"
                                                 "_violinplot_quartile_grayscale_centered.csv"))
    plt.savefig(plots_folder+"/"+("binaural_recorded_4078_main_kemar_full_spec_folded"
                                  "_violinplot_quartile_grayscale_centered.svg"))

def fig_4b():
    format_data_wood_human_data("/om/user/francl/wood_localization_errorbar.csv",add_errorbars=True)
    plt.savefig(plots_folder+"/azimuth_vs_error_wood_human.svg")

def fig_3c(regex=None,iter_num=100000):
    if regex is not None:
       regex = regex+ ("arch_number_*init_0/batch_conditional_"
                       "*middlebrooks_wrightman*999_side_aware_normalized_iter{}*".format(iter_num))
    make_middlebrooks_network_plots(regex)

def fig_4c(regex=None,iter_num=100000):
    if regex is not None:
        regex = regex + ("arch_number_*init_0/batch_conditional_"
                  "broadbandNoiseRecords_wood_convolved_anechoic_"
                  "oldHRIRdist140_stackedCH_upsampled_iter{}*".format(iter_num))
    make_wood_network_plot(regex)
    plt.xlim(-95,95)
    plt.xticks([-90,-45,0,45,90])
    plt.savefig(plots_folder+"/azimuth_vs_error_wood_graph_network.svg",dpi=400)

def fig_4c_corrected(regex=None,iter_num=100000):
    if regex is not None:
        regex = regex + ("arch_number_*init_0/batch_conditional_"
                         "*_wood_bizley_03092022_iter{}*".format(iter_num))
        print(regex)
    make_wood_bizley_network_plot(regex)
    plt.ylim(0,50)
    plt.savefig(plots_folder+"/azimuth_vs_error_wood_graph_network.png",dpi=400)

def fig_4e():
    make_bandwidth_vs_error_humabn_plot()

def fig_4f(regex=None,iter_num=100000):
    if regex is not None:
        regex = regex + ("arch_number_*init_0/batch_conditional*bandpass"
                  "*HRIR*iter{}.npy".format(iter_num))
    make_yost_network_plot(regex)
    plt.ylim(0,25)
    plt.savefig(plots_folder+"/bandwidth_vs_error_network_plot.svg")
    plt.clf()
    get_van_opstal_human_plot("after",hardcode_ref_grid=True)
    plt.savefig(plots_folder+"/van_opstal_after_human_hardcoded_ref.svg")
    get_van_opstal_human_plot("before",hardcode_ref_grid=True)
    plt.savefig(plots_folder+"/van_opstal_before_human_hardcoded_ref.svg")

def fig_5d_and_5e(regex=None,iter_num=100000):
    if regex is not None:
        regex = regex+ ("arch_number_[0-9][0,1,3-9][0-9]_init_0/"
                        "batch_conditional_broadbandNoiseRecords_"
                        "convolvedCIPIC*iter{}.npy".format(iter_num))
    make_van_opstal_network_plots(regex)

def fig_5f_and_g(regex=None):
    make_van_opstal_plot_individual_networks(regex)

    make_van_opstal_plot_individual_networks(regex)

def fig_5j(regex=None,iter_num=100000):
    if regex is not None:
        regex = regex + ("arch_number_*init_0/batch_conditional_"
                        "broadbandNoiseRecords_convolved_smooth*HRIR_direct_"
                        "stackedCH_upsampled_iter{}.npy".format(iter_num))
    make_kulkarni_plot_networks(regex)

def fig_5k_and_l(regex=None):
    make_kulkarni_network_prediction_plots(regex)

def fig_6b(regex=None):
    make_click_precedence_effect_plot(regex=None)

def fig_6c():
    make_litovsky_human_plot(add_errorbars=True)
    plt.gca().legend().remove()
    plt.savefig(plots_folder+"/litovsky_errorbars_seaborn.svg")

def fig_6d(regex=None,iter_num=100000):
    if regex is not None:
        regex = regex + ("arch_number_*init_0/batch_conditional_*"
                        "precedence*multiAzim*pinknoise*5degStep*iter{}*".format(iter_num))
    litovsky_error_plot_multi_azim(regex=regex)
    plt.gca().legend().remove()
    plt.savefig(plots_folder+"/precedence_effect_litovsky_rmse_plots_network.svg")

def fig_hebrank_wright(regex=None,iter_num=100000):
    if regex is not None:
        regex = regex + ("arch_number_*_init_0/batch_conditional_"
                            "noiseRecords_hebrank_wright_iter{}.npy".format(iter_num))
    else:
        regex = ("/om5/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                 "arch_number_*_init_0/batch_conditional_"
                 "noiseRecords_hebrank_wright_iter{}.npy".format(iter_num))
    pd_hebrank_wright = make_dataframe(regex,elevation_predictions=True)
    pd_hebrank_wright_formatted = format_hebrank_wright_dataframe_azim_limited(pd_hebrank_wright)
    make_hebrank_wright_human_plot()
    make_hebrank_wright_plots(pd_hebrank_wright_formatted)

    

def fig_7b_and_7c():
    get_individual_error_graph(["/om5/user/francl/grp-om2/gahlm/dataset_pipeline_test/",
                                "/om5/user/francl/new_task_archs/new_task_archs_anechoic_training/",
                                "/om5/user/francl/new_task_archs/new_task_archs_no_background_noise_80dBSNR_training/",
                                "/om5/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_training/"],
                                #"/om2/user/francl/new_task_archs/new_task_archs_spectrally_modulated_natural_sounds_40_MFCC_10_depth/"],
                               #"/om2/user/francl/new_task_archs/new_task_archs_spectrally_modulated_noise_248_MFCC_80_depth/"],
                               # "/om5/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_training/",
                                #"/om2/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_anechoic_no_background_noise_80dBSNR_training/"],
                               iter_nums=[100000,100000,100000,150000],
                               condtion_names=["Normal", "Anechoic", "No Background","Bandlimited"])
    #,"Spec. Mod. Sounds"])
                               #,"Combined Minipulations"]) 

def fig_7e():
    get_error_graph_by_task(["/om5/user/francl/grp-om2/gahlm/dataset_pipeline_test/",
                             "/om5/user/francl/new_task_archs/new_task_archs_anechoic_training/",
                             "/om5/user/francl/new_task_archs/new_task_archs_no_background_noise_80dBSNR_training/",
                             #"/om2/user/francl/new_task_archs/new_task_archs_spectrally_modulated_noise_2_MFCC_60_depth/"],
                             "/om5/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_training/"],
                             #"/om2/user/francl/new_task_archs/new_task_archs_no_background_noise_80dBSNR_with_neural_noise_61dBSNR_training/"],
                             #"/om2/user/francl/new_task_archs/new_task_archs_spectrally_modulated_natural_sounds_40_MFCC_10_depth/"],
                            iter_nums=[100000,100000,100000,150000],
                            condtion_names=["Normal", "Anechoic", "No Background","Bandlimited"])
    #,"Spectrally Modulated"])
                            #condtion_names=["Normal", "Anechoic", "No Background","Unnatural Low Spec. Mod. Sounds"])

def make_si_fig_rmse_with_human_floor():
    get_mean_error_graph_subset(["/om5/user/francl/grp-om2/gahlm/dataset_pipeline_test/"],
                                #"/om5/user/francl/new_task_archs/new_task_archs_anechoic_training/",
                                #"/om5/user/francl/new_task_archs/new_task_archs_no_background_noise_80dBSNR_training/",
                                #"/om2/user/francl/new_task_archs/new_task_archs_spectrally_modulated_natural_sounds_40_MFCC_10_depth/"],
                               #"/om2/user/francl/new_task_archs/new_task_archs_spectrally_modulated_noise_248_MFCC_80_depth/"],
                               # "/om5/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_training/",
                                #"/om5/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_training/",
                                #"/om2/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_anechoic_no_background_noise_80dBSNR_training/"],
                               iter_nums=[100000],
                               condtion_names=["Normal"],
                               experiment_subset=['wood','yost','kulkarni','van_opstal','litovsky'])
                               #,"Combined Minipulations"]) 


def fig_7d():
    make_click_precedence_effect_across_conditions()

def make_alternative_training_graphs(function,regex_list=None,
                                     output_folder_list=None):
    global plots_folder
    if regex_list is None:
        regex_list = [
                     "/om5/user/francl/new_task_archs/new_task_archs_anechoic_training/",
                     "/om5/user/francl/new_task_archs/new_task_archs_no_background_noise_80dBSNR_training/",
                     #"/om2/user/francl/new_task_archs/new_task_archs_spectrally_modulated_noise_2_MFCC_60_depth/",
                     #"/om2/user/francl/new_task_archs/new_task_archs_spectrally_modulated_noise_248_MFCC_80_depth/"]
                     "/om5/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_training/"]
                     #"/om2/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_anechoic_no_background_noise_80dBSNR_training/"]
                     #"/om2/user/francl/new_task_archs/new_task_archs_no_background_noise_80dBSNR_with_neural_noise_61dBSNR_training/"]
                     #"/om2/user/francl/new_task_archs/new_task_archs_spectrally_modulated_natural_sounds_40_MFCC_10_depth/"]
        iter_nums = [100000,100000,150000]
        #iter_nums = [100000]
        output_folder_list = ["/anechoic_training","/noiseless_training","/half_octave_noise"]
        #output_folder_list = ["/half_octave_noise"]
        #output_folder_list = ["/no_background_noise_with_neural_noise"]
        #output_folder_list = ["/MFCC_2_spec_mod_training","/MFCC_248_spec_mod_training"]
                              #"/unnatural_sounds_training","/unnatural_sounds_training_anechoic_80dBSNR"]
    plots_folder_original = plots_folder
    for regex,output_folder,iter_num in zip(regex_list,output_folder_list,iter_nums):
        plots_folder = plots_folder_original + output_folder
        try:
            os.makedirs(plots_folder)
        except OSError:
            if not os.path.isdir(plots_folder):
                raise
        function(regex,iter_num)
    plots_folder = plots_folder_original


fig_2c()
fig_2d()
fig_4b()
fig_3c()
fig_4c()
fig_4c_corrected()
fig_4e()
fig_4f()
fig_5d_and_5e()
fig_5f_and_g()
fig_5j()
fig_5k_and_l()
fig_6b()
fig_6c()
fig_6d()
fig_hebrank_wright()
fig_7b_and_7c()
fig_7e()
make_si_fig_rmse_with_human_floor()
fig_7d()
make_alternative_training_graphs()