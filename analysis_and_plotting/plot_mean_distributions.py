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
    Calculates distance between labeled azimutha and predicted azimuth. Check
    distance both directions around circle and returns smaller one.
    '''
    azim = row['azim']
    pred = row['predicted_azim']
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
        keylist_regex = fname.replace("batch_conditional","keys_test").replace(".npy","")+"*"
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
        var_order = ["azim","elev", "bandwidth",
                     "center_freq"]
    elif ("binaural_recorded" in regex) and ("speech_label" in regex):
        var_order = ["azim", "elev","speech"]
    elif "binaural_recorded" in regex:
        var_order = ["azim", "elev"]
    elif ("testset" in regex) and ("convolved" in regex):
        var_order = ["azim", "elev"]
    else:
        var_order = ["azim","freq"]
    return var_order
        

def get_order(regex,data):
    var_order = get_cols(regex)
    key_swap_indicies = [next((idx for idx,pos in enumerate(var_order) if pos in key.split('/')), None)
                         for key in data]
    return key_swap_indicies

def reshape_and_filter(temp_array,label_order):
    prob = np.vstack(temp_array[:,0])
    if all(isinstance(x,np.ndarray) for x in temp_array[:,1][0]):
        #This deals with testsets metadata still in tuples because vstack
        #dealts with things incorrectly if all elements are arrays
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


def make_bandwidth_vs_error_humabn_plot():
    pd_yost = pd.read_csv("/home/francl/Bandwidth_Data.csv",header=[0])
    #conver bandwidths in fractions to numeric values
    pd_yost['bandwidth'] = pd_yost['bandwidth'].apply(eval)
    #Align Pure tone data with other pure tone points
    #Difference was caused by error in original CSV write
    pd_yost.iloc[7,1] = .001
    pd_yost.iloc[16,1] = .001
    #dummy column needed to add point makers with seaborn
    pd_yost['same_val'] = 1
    #Rename columns
    pd_yost.columns = ['frequency', 'Bandwidth (Octaves)', 'RMS Error (Degrees)',
                       'standard deviation', 'same_val']
    fig = plt.figure(figsize=(13,11))
    plt.clf()
    sns.lineplot(x='Bandwidth (Octaves)',y='RMS Error (Degrees)',hue="same_val",
                 lw=4,ms=10,legend=False,data=pd_yost,style="same_val",
                 markers=["o"],err_style='bars',
                 err_kws={'capsize':4,'elinewidth':4,'capthick':4},
                 dashes=False,palette=['k'])
    plt.xticks(rotation=90,size=30)
    plt.yticks(size=30)
    plt.ylim(5,25)
    plt.ylabel("RMS Error (Degrees)",size=40)
    plt.xlabel("Bandwidth (Octaves)",size=40)
    [x.set_color("black") for x in plt.gca().get_lines()]
    plt.tight_layout()
    plt.savefig(plots_folder+"/"+"yost_human_data_collapsed_bandwidth.png")



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
    ax1.set_ylim(0,50)
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
    #colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
    #for i,j in enumerate(ax1.lines):
    #    j.set_color(colors[i])


def plot_means_squared_error_millsgraph(batch_conditionals_ordered,freqs,azim_lim=36,no_fold=False,bins_5deg=False,collapse_conditions=False):
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
                a = [i[0][:72] for i in batch_conditionals if i[1] == azim]
            else:
                a = [i[0][:36] for i in batch_conditionals if i[1] == azim]
            averages = [sum(i)/len(i) for i in zip(*a)]
            try:
                max_idxs = np.argmax(a,axis=1)
            except:
                print(azim,len(batch_conditionals))
                pdb.set_trace()
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
    transposed = list(map(list,zip(*mse_by_freq)))

    fig = plt.figure(figsize=(13,11))
    ax1 = fig.add_subplot(111)
    SEM_mean_array = []
    for line_azim in azim_idx:
        mean = [x[0] for x in transposed[line_azim]]
        bottom_error = [x[1] for x in transposed[line_azim]]
        top_error = [x[2] for x in transposed[line_azim]]
        if collapse_conditions:
            SEM_mean_array.append(mean)
        else:
            ax1.errorbar(freqs,mean,yerr=[bottom_error,top_error],marker='o',markersize=3, label = "{} Deg.".format(10*line_azim))
        #ax1.set_ylim(0,100)
        #ax1.plot(freqs,transposed[line_azim],marker='o',markersize=3, label = line_azim)
    if collapse_conditions:
        np_SEM_mean_array = np.array(SEM_mean_array)
        mean = np.mean(np_SEM_mean_array,axis=0)
        SEM = np.std(np_SEM_mean_array,axis=0)/np.sqrt(len(SEM_mean_array))
        top_error = SEM*1.96
        bottom_error = SEM*1.96
        ax1.errorbar(freqs,mean,yerr=[bottom_error,top_error],marker='o',markersize=3, label = "{} Deg.".format(10*line_azim))
    ax1.set_xscale('log')
    ax1.set_xlim(99,12000)
    ax1.set_xlabel("Frequency (Hz)", size=40)
    ax1.set_ylabel("Mean error (Degrees)", size=40)
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    ax1.legend(loc=2,fontsize=20)

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
                             marker='o', label = freq_label,
                             lw=4,ms=10,capsize=4,elinewidth=4,
                             capthick=4,color='k')
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

def polar_heat(values, thetas=None, radii=None, ax=None, fraction=0.3,
               **kwargs):

    values = np.atleast_2d(values)
    if thetas is None:
        thetas = np.linspace(0, 2*np.pi, values.shape[1]).reshape(1, -1)
    if radii is None:
        radii = np.linspace(0, 1, values.shape[0] + 1).reshape(-1, 1)
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'polar':True})

    mesh = ax.pcolormesh(thetas, radii, values, **kwargs)

    radrange = radii.ptp()
    ax.set_rlim(radrange * (1 - 1. / fraction), radrange)
    return mesh

def sphere_plot():
	n_theta = 8 # number of values for theta
	n_phi = 37  # number of values for phi
	r = 2        #radius of sphere

	theta, phi = np.mgrid[6/36*np.pi:0.55*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]

	x = r*np.sin(theta)*np.cos(phi)
	y = r*np.sin(theta)*np.sin(phi)
	z = r*np.cos(theta)

	# mimic the input array
	# array columns phi, theta, value
	# first n_theta entries: phi=0, second n_theta entries: phi=0.0315..
	inp = []
	for j in phi[0,:]:
		for i in theta[:,0]:
			val = 0.7+np.cos(j)*np.sin(i+np.pi/4.)# put something useful here
			inp.append([j, i, val])
	inp = np.array(inp)
	print(inp.shape)
	print(inp[49:60, :])

	#reshape the input array to the shape of the x,y,z arrays. 
	c = inp[:,2].reshape((n_phi,n_theta)).T
	print(z.shape)
	print(c.shape)
	c[0,0] = 3.2


	#Set colours and render
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')
	#use facecolors argument, provide array of same shape as z
	# cm.<cmapname>() allows to get rgba color from array.
	# array must be normalized between 0 and 1
	ax.plot_surface(
		x,y,z,  rstride=1, cstride=1, facecolors=mpl.cm.viridis(c/c.max()), alpha=0.7, linewidth=1) 
	ax.set_xlim([-2.2,2.2])
	ax.set_ylim([-2.2,2.2])
	ax.set_zlim([0,4.4])
	ax.set_aspect("equal")
	ax.view_init(elev=30, azim=90)

def split_noise_groups(batch_conditionals_noise):
    bins = [50,100,200,400,800,1600,3200,6400,12800]
    ordered = []
    for value in bins:
        matches = [example for example in batch_conditionals_noise if abs(example[2]- value) <= value*.20]
        ordered.append(matches)
    return ordered

def split_tone_groups(batch_conditionals_tone):
    bins = [(0,1000),(1000,3000),(3000,15000)]
    ordered = []
    for value_min,value_max in bins:
        matches = [example for example in batch_conditionals_tone if value_min <= example[2] <= value_max]
        ordered.append(matches)
    return ordered

def split_tone_indv_freqs(batch_conditionals_tone):
    freqs = list(set(batch_condtionals_tone[:,2]))
    ordered = []
    for value in freqs:
        matches = [example for example in batch_conditionals_tone if value == example[2]]
        ordered.append(matches)
    return ordered
    

def split_tone_mills_graph(batch_conditionals_tone):
    bins = [(250,350),(500,600),(750,850),(1000,1100),(1250,1350),(1500,1600),(1750,1850),(2000,2100),(3000,3100),(4000,4100),(6000,6100),(8000,8100),(10000,10100)]
    ordered = []
    for value_min,value_max in bins:
        matches = [example for example in batch_conditionals_tone if abs(example[2]- value_min) <= 0.2*value_min**.9]
        ordered.append(matches)
    freqs = [x[0]  for x in bins]
    return ordered, freqs

def split_ITD_tone_groups(batch_conditionals_noise):
    bins = [100,200,400,800,1000,1250,1500,1750,2000,3000,6000]
    ordered = []
    for value in bins:
        matches = [example for example in batch_conditionals_noise if abs(example[2]- value) <= 0.2*value**.9]
        ordered.append(matches)
    return ordered

def split_ITD_tone_groups_1octv(batch_conditionals_noise):
    bins = [100,200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,11000]
    ordered = []
    for value in bins:
        matches = [example for example in batch_conditionals_noise if abs(example[2]- value) <= 0.2*value**.85]
        ordered.append(matches)
    return ordered

def split_ITD_tone_groups_1octv_abs(batch_conditionals_noise):
    bins = [100,200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,11000]
    ordered = []
    for value in bins:
        matches = [example for example in batch_conditionals_noise if (abs(example[2]- value) <= 50) or (abs(example[2]- value) <= 100 and value > 3000)]
        ordered.append(matches)
    return ordered

def assign_ITD_tone_group_abs(row):
    bins = [100,200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000,2250,2500,2750,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,11000]
    freq = row['frequency']
    for cur_bin in bins:
        if (abs(freq - cur_bin) <= 50) or (abs(freq - cur_bin) <= 100 and cur_bin > 3000):
            return cur_bin
    

def split_sam_tones_by_carrier_freq(batch_conditional_data):
   keys = sorted(set([x[1] for x in batch_conditional_data]))
   ordered = []
   for key in keys:
       matches = [x for x in batch_conditional_data if x[1] == key]
       ordered.append(matches)
   return ordered, list(keys)


def split_sam_tones_by_modulator_freq(batch_conditional_data):
   keys = sorted(set([x[2] for x in batch_conditional_data]))
   ordered = []
   for key in keys:
       matches = [x for x in batch_conditional_data if x[2] == key]
       ordered.append(matches)
   return ordered, list(keys)


def split_ITD_comparisons_by_modulation_freq(afc_results):
   keys = sorted(set([x[2][0] for x in afc_results]))
   ordered = []
   for key in keys:
       matches = [x for x in afc_results if x[2][0] == key]
       ordered.append(matches)
   return ordered, list(keys)



def plot_noises_ordered(odered):
    ordered = split_noise_groups(batch_conditionals_noise)
    bins = [50,100,200,400,800,1600,3200,6400,12800]
    for i,noise_type in enumerate(ordered):
        plot_cond_prob_azim(noise_type)
        plt.savefig(plots_folder+"/"+"cond_dist_noise{}_var_env.png".format(bins[i])) 

def plot_noises_folded(odered):
    ordered = split_noise_groups(batch_conditionals_noise)
    bins = [50,100,200,400,800,1600,3200,6400,12800]
    for i,noise_type in enumerate(ordered):
        plot_cond_prob_azim_folded(noise_type)
        plt.savefig(plots_folder+"/"+"cond_dist_noise{}_folded_var_env.png".format(bins[i])) 

def plot_tones_ordered(ordered):
    ordered_loc = split_tone_groups(batch_conditionals_tone)
    bins = [1000,3000,15000]
    for i,noise_type in enumerate(ordered):
        plot_cond_prob_azim(noise_type)
        plt.savefig(plots_folder+"/"+"cond_dist_tones{}_var_env.png".format(bins[i])) 

def plot_tones_folded(ordered):
    ordered_loc = split_tone_groups(batch_conditionals_tone)
    bins = [1000,3000,15000]
    for i,noise_type in enumerate(ordered):
        plot_cond_prob_azim_folded(noise_type)
        plt.savefig(plots_folder+"/"+"cond_dist_tones{}_folded_var_env.png".format(bins[i])) 


def plot_mills_graph_tones(batch_conditionals_tone,azim_lim=36):
    ordered,freqs = split_tone_mills_graph(batch_conditionals_tone)
    plot_means_squared_error(ordered,freqs,azim_lim)
    plt.title("Mills Graph with Pure Tones")
    plt.savefig(plots_folder+"/"+"4orderfilt_mills_graph_jittered_tones_sphere_do_net.png")

def plot_mills_graph_noise(batch_conditionals_noise,azim_lim=36):
    freqs = [200,400,800,1600,3200,6400,12800]
    ordered = split_noise_groups(batch_conditionals_noise)
    plot_means_squared_error(ordered[2:],freqs,azim_lim)
    plt.title("Mills Graph with Noise Bursts")
    plt.savefig(plots_folder+"/"+"4orderfilt_mills_graph_noise_jittered_sphere_do_net.png")

def plot_error_by_azim_tones(batch_conditionals_tone):
    ordered_loc = split_tone_groups(batch_conditionals_tone)
    bins = [1000,3000,15000]
    for i in range(len(ordered_loc)):
        plot_means_squared_error_by_freq([ordered_loc[i]],labels=["Anechoic"])
        plt.title("Mean Error with Tones Near {}Hz".format(bins[i]))
        plt.savefig(plots_folder+"/"+"4orderfilt_mse_jittered_tones_no_fold_{}_sphere_do_net_by_azim.png".format(bins[i]))
    
def plot_error_by_azim_noise(batch_conditionals_noise):
    ordered_loc = split_noise_groups(batch_conditionals_noise)
    bins = [50,100,200,400,800,1600,3200,6400,12800]
    for i in range(len(ordered_loc)):
        plot_means_squared_error_by_freq([ordered_loc[i]],labels=["Anechoic"])
        plt.title("Mean Error with 1 Octave Noise Band at {}Hz".format(bins[i]))
        plt.savefig(plots_folder+"/"+"4orderfilt_mse_jittered_noise_no_fold_{}_sphere_do_net_by_azim.png".format(bins[i]))

def plot_ITD_graph(batch_conditionals_noise,azim_lim=36):
    freqs = [100,200,400,800,1600,3200,6400,12800]
    ordered = split_noise_groups(batch_conditionals_noise)
    plot_means_squared_error(ordered[1:],freqs,azim_lim,collapse_azims=False)
    plt.title("Mills Graph with ITDs only")
    plt.savefig(plots_folder+"/"+"4orderfilt_mills_graph_left_side_ITD_tones_jittered_sphere_do_net_iter75k.png")


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


def compare_ITDs_across_midline(batch_condtionals_tone,min_dist=0,max_dist=math.inf,manually_added=True,centerofmass=False):
    #compares all pairs of stimuli and determines if network can identify which
    #is on the right and on the left(2AFC task simulation)
    batch_results = []
    for batch in batch_condtionals_tone:
        cond_dict = {}
        for exemplar in batch:
            cond_dict.setdefault((abs(exemplar[1]),exemplar[2]),[]).append(exemplar)
        #sorted_batch = sorted(batch, key=lambda x: abs(x[1]))
        sorted_pairs = [(v[0],v[1]) for k,v in cond_dict.items() if len(v) ==2]
        #sorted_pairs = [(x,y) for x,y in zip(sorted_batch, sorted_batch[1:]) if y[1] != x[1] and abs(y[1]) == abs(x[1])]
        results = []
        for pair in sorted_pairs:
            correct = False
            x,y = pair[0],pair[1]
            x_label,y_label = x[1],y[1]
            if not manually_added:
                x_idx,y_idx = get_folded_label_idx(x_label),get_folded_label_idx(y_label)
            else:
                x_idx,y_idx = x_label,y_label
            label_diff = x_idx - y_idx
            if label_diff == 0:
                pdb.set_trace()
            #meant to allow for splitting 2AFC task into different distance
            #ranges to test system sensitivity
            if abs(label_diff) > max_dist or abs(label_diff) < min_dist:
                continue
            x_freq,y_freq = x[2],y[2]
            x_dist_folded,y_dist_folded = fold_locations_full_dist_5deg(x[0][:72]),\
                fold_locations_full_dist_5deg(y[0][:72])
            if centerofmass:
                x_center,y_center = ndimage.center_of_mass(x_dist_folded)[0],\
                    ndimage.center_of_mass(y_dist_folded)[0]
            else:
                x_center,y_center = x_dist_folded.argmax(), y_dist_folded.argmax()
            predicted_diff = x_center-y_center
            if sign(predicted_diff) == sign(label_diff):
                correct = True
            results.append([(x_label,y_label),(x_freq,y_freq),predicted_diff,label_diff,correct])
        batch_results.append(results)
        batch_results = sorted(batch_results, key= lambda x: x[1][1])
    return batch_results



def compare_sam_tones_across_midline(batch_condtionals_tone,min_dist=0,max_dist=math.inf,manually_added=True,centerofmass=False):
    #compares all pairs of stimuli and determines if network can identify wich
    #is on the right and on the left(2AFC task simulation)
    batch_results = []
    for batch in batch_condtionals_tone:
        cond_dict = {}
        for exemplar in batch:
            cond_dict.setdefault((abs(exemplar[1]),exemplar[2],exemplar[3],exemplar[4]),[]).append(exemplar)
        #sorted_batch = sorted(batch, key=lambda x: abs(x[1]))
        sorted_pairs = [(v[0],v[1]) for k,v in cond_dict.items() if len(v) ==2]
        #sorted_pairs = [(x,y) for x,y in zip(sorted_batch, sorted_batch[1:]) if y[1] != x[1] and abs(y[1]) == abs(x[1])]
        results = []
        for pair in sorted_pairs:
            correct = False
            x,y = pair[0],pair[1]
            #Label set to modulation delay value
            x_delay,y_delay = x[4],y[4]
            x_flipped, y_flipped = x[5],y[5]
            x_left, y_left = (-1)**x_flipped, (-1)**y_flipped
            x_label, y_label = x_delay*x_left,y_delay*y_left
            if not manually_added:
                x_idx,y_idx = get_folded_label_idx(x_label),get_folded_label_idx(y_label)
            else:
                x_idx,y_idx = x_label,y_label
            label_diff = x_idx - y_idx
            if label_diff == 0:
                pdb.set_trace()
            #meant to allow for splitting 2AFC task into different distance
            #ranges to test system sensitivity
            if abs(label_diff) > max_dist or abs(label_diff) < min_dist:
                continue
            x_carrier_freq,y_carrier_freq = x[1],y[1]
            x_modulator_freq,y_modulator_freq = x[2],y[2]
            x_dist_folded,y_dist_folded = fold_locations_full_dist_5deg(x[0][:72]),\
                fold_locations_full_dist_5deg(y[0][:72])
            if centerofmass:
                x_center,y_center = ndimage.center_of_mass(x_dist_folded)[0],\
                    ndimage.center_of_mass(y_dist_folded)[0]
            else:
                x_center,y_center = x_dist_folded.argmax(), y_dist_folded.argmax()
            predicted_diff = x_center-y_center
            if sign(predicted_diff) == sign(label_diff):
                correct = True
            results.append([(x_label,y_label),(x_carrier_freq,y_carrier_freq),
                            (x_modulator_freq,y_modulator_freq),predicted_diff,
                            label_diff,correct])
        batch_results.append(results)
        batch_results = sorted(batch_results, key= lambda x: x[0][2])
    return batch_results


def compare_transposed_tones_across_midline(batch_condtionals_tone,min_dist=0,max_dist=math.inf,manually_added=True,centerofmass=False):
    #compares all pairs of stimuli and determines if network can identify wich
    #is on the right and on the left(2AFC task simulation)
    batch_results = []
    for batch in batch_condtionals_tone:
        cond_dict = {}
        for exemplar in batch:
            cond_dict.setdefault((abs(exemplar[1]),exemplar[2],exemplar[3]),[]).append(exemplar)
        #sorted_batch = sorted(batch, key=lambda x: abs(x[1]))
        sorted_pairs = [(v[0],v[1]) for k,v in cond_dict.items() if len(v) ==2]
        #sorted_pairs = [(x,y) for x,y in zip(sorted_batch, sorted_batch[1:]) if y[1] != x[1] and abs(y[1]) == abs(x[1])]
        results = []
        for pair in sorted_pairs:
            correct = False
            x,y = pair[0],pair[1]
            #Label set to modulation delay value
            x_delay,y_delay = x[3],y[3]
            x_flipped, y_flipped = x[4],y[4]
            x_left, y_left = (-1)**(x_flipped-1), (-1)**(y_flipped-1)
            x_label, y_label = x_delay*x_left,y_delay*y_left
            if not manually_added:
                x_idx,y_idx = get_folded_label_idx(x_label),get_folded_label_idx(y_label)
            else:
                x_idx,y_idx = x_label,y_label
            label_diff = x_idx - y_idx
            if label_diff == 0:
                pdb.set_trace()
            #meant to allow for splitting 2AFC task into different distance
            #ranges to test system sensitivity
            if abs(label_diff) > max_dist or abs(label_diff) < min_dist:
                continue
            x_carrier_freq,y_carrier_freq = x[1],y[1]
            x_modulator_freq,y_modulator_freq = x[2],y[2]
            x_dist_folded,y_dist_folded = fold_locations_full_dist_5deg(x[0][:72]),\
                fold_locations_full_dist_5deg(y[0][:72])
            if centerofmass:
                x_center,y_center = ndimage.center_of_mass(x_dist_folded)[0],\
                    ndimage.center_of_mass(y_dist_folded)[0]
            else:
                x_center,y_center = x_dist_folded.argmax(), y_dist_folded.argmax()
            predicted_diff = x_center-y_center
            if sign(predicted_diff) == sign(label_diff):
                correct = True
            results.append([(x_label,y_label),(x_carrier_freq,y_carrier_freq),
                            (x_modulator_freq,y_modulator_freq),predicted_diff,
                            label_diff,correct])
        batch_results.append(results)
        batch_results = sorted(batch_results, key= lambda x: x[2])
    return batch_results

def bin_ITDs(batch_condtionals_tone,centerofmass=False,min_pos=-math.inf,max_pos=math.inf):
    #finda most likely locaciont for each distribution
    results = []
    for batch in batch_condtionals_tone:
        label = batch[1]
        #meant to allow for splitting 2AFC task into different distance
        #ranges to test system sensitivity
        if label > max_pos or label < min_pos:
            continue
        freq = batch[2]
        folded= fold_locations_full_dist(batch[0][:36])
        if centerofmass:
            center = ndimage.center_of_mass(folded)[0]-1
        else:
            center = folded.argmax()
        results.append([label,freq,center])
    return np.array(results)

def filter_comparisons_by_offset(afc_results,cutoff):
    afc_filtered = []
    for freq_group in afc_results:
        filtered = [x for x in freq_group if abs(x[0][0]) < cutoff]
        afc_filtered.append(filtered)
    return afc_filtered


def filter_comparisons_by_offset_equal(afc_results, min_cutoff, max_cutoff):
    afc_filtered = []
    for freq_group in afc_results:
        filtered = [x for x in freq_group if min_cutoff <= abs(x[0][0]) <= max_cutoff]
        afc_filtered.append(filtered)
    return afc_filtered

def filter_comparisons_by_carrier(afc_results,carrier,
                                  jitter_distance_octave_fraction=0):
    carrier = [carrier] if carrier is int else carrier
    afc_filtered = []
    fuzzy_mult = 2**(jitter_distance_octave_fraction)-1
    for freq_group in afc_results:
        filtered = [x for x in freq_group if val_in_list_fuzzy(abs(x[1][0]),carrier,fuzzy_mult)]
        afc_filtered.append(filtered)
    return afc_filtered


def calc_2AFC_vs_ITD(afc_results_filtered):
    keys = sorted(set([abs(x[0][0]) for x in afc_results_filtered]))
    acc_arr = []
    for key in keys:
        values = [x for x in afc_results_filtered if abs(x[0][0]) == key]
        np_values = np.array(values)
        bool_arr = np_values[:,-1].astype(np.bool)
        total = np_values.shape[0]
        mean_error,low_ci,high_ci = calc_CI(bool_arr,single_entry=True,
                                        stat_func=bs_stats.sum,iteration_batch_size=100)
        acc = mean_error/total
        low_acc = low_ci/total
        high_acc = high_ci/total
        acc_arr.append([acc,low_acc,high_acc])
    transposed = list(zip(*acc_arr))
    mean = [100*x for x in transposed[0]]
    bottom_error = [100*x for x in transposed[1]]
    top_error = [100*x for x in transposed[2]]
    return (keys,mean,bottom_error,top_error)

def plot_2AFC_vs_ITD_by_mod_freq(afc_results_array,keys,carrier_list =None,
                                 plotted_keys=[50,150,300,600]):

    assert all(plotkey in keys for plotkey in plotted_keys), \
            ("Plotted Key not Available! \n Available Keys: {} \
             \n Requested Keys: {}".format(keys,plotted_keys))

    fig = plt.figure(figsize=(13,11))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("Delay (microseconds)",size=40)
    ax1.set_ylabel("2AFC % correct",size=40)
    ax1.set_ylim(-10,110)
    ax1.set_xlim(0,1050)
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    ax1.axhline(50,color='k',linestyle='dashed',alpha=0.5)
    #iterates through keys and skips if not in plotted_keys set
    for i,key in enumerate(keys):
        #iterates through networks and calculates individual repsonses
        if key not in plotted_keys:
            continue
        mean_array = []
        for afc_results in afc_results_array:
            #finds correct response data for given key
            data_idx = [i for i,x in enumerate(afc_results) if 
                        x[0][2] == (key,key)]
            assert len(data_idx) == 1 ,\
                    ("Data not formatted as expected! " 
                    "Multiple sublists with same modulation frequency!")
            #calculates 2AFC results
            ITDs,mean,bottom_error,top_error = \
                    calc_2AFC_vs_ITD(afc_results[data_idx[0]])

            mean_array.append(mean)
        #calculates SEM over networks
        np_mean_array = np.array(mean_array)
        grand_mean = np.mean(np_mean_array,axis=0)
        SEM = np.std(np_mean_array,axis=0)/np.sqrt(len(mean_array))
        top_error = SEM*1.96
        bottom_error = SEM*1.96
        x_labels = [x*1000 for x in ITDs]
        #ax1.plot(x_labels,mean,marker='o',markersize=3,
        #         label="{} Modulation Freq.".format(float(key)))
        #plots line over networks for single modulation frequency
        ax1.errorbar(x_labels,grand_mean,yerr=[bottom_error,top_error],marker='o',markersize=3,
                     label= "{} Modulation Freq.".format(int(key)))
    ax1.legend(loc='best')
    

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






def plot_localization_manaully_added_ILD(results,freq_ranges=[("low",200,400),("high",2000,4000)]):
    ILD_list = [-20,-15,-10,-5,0,5,10,15,20]
    for freq_range,min_val,max_val in freq_ranges:
        results_freq_filtered = results[np.all([results[:,1] <= max_val, results[:,1] >=
                                           min_val],axis=0)]
        for ILD in ILD_list:
            results_filtered = results_freq_filtered[results_freq_filtered[:,0] == ILD]
            fig = plt.figure(figsize=(11,8))
            ax1 = fig.add_subplot(111)
            bins = [i for i in range(90,280,10)]
            ax1.hist(results_filtered[:,2])
            ax1.set_xlabel("Predicted Center Position (Folded)")
            ax1.set_xlim(0,18)
            ax1.set_ylabel("Count")
            ax1.set_title("Folded Localization with {} ILD".format(ILD))
            plt.savefig(plots_folder+"/"+"localization_manually_added_ILD_{}_freq_{}ILD.png".format(freq_range,ILD)) 

    
def plot_localization_manaully_added_ITD(results, freq_ranges=[("low",200,400),("high",2000,4000)]):
    ITD_list = [-30,-20,-15,-10,-5,0,5,10,15,20,30]
    for freq_range,min_val,max_val in freq_ranges:
        results_freq_filtered = results[np.all([results[:,1] <= max_val, results[:,1] >=
                                           min_val],axis=0)]
        for ITD in ITD_list:
            results_filtered = results_freq_filtered[results_freq_filtered[:,0] == ITD]
            fig = plt.figure(figsize=(11,8))
            ax1 = fig.add_subplot(111)
            bins = [i for i in range(90,280,10)]
            ax1.hist(results_filtered[:,2])
            ax1.set_xlabel("Predicted Center Position (Folded)")
            ax1.set_xlim(0,18)
            ax1.set_ylabel("Count")
            ax1.set_title("Folded Localization with {} ITD".format(ITD))
            plt.savefig(plots_folder+"/"+"localization_manually_added_ITD_{}_freq_{}ITD.png".format(freq_range,ITD)) 
            
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




def plot_relative_comparison_sam_tones_multiple_lines(results_ILD_array,freqs=None,collapse_across_conditions=False,sing_freq=False):
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
            bool_arr = res_np[:,-1].astype(np.bool)
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
    np_mean_array = np.array(mean_array)
    mean = np.mean(np_mean_array,axis=0)
    SEM = np.std(np_mean_array,axis=0)/np.sqrt(len(mean_array))
    top_error = SEM*1.96
    bottom_error = SEM*1.96
    ax1.errorbar(freqs,mean,yerr=[bottom_error,top_error],marker='o',markersize=3,alpha=0.6)


def filter_joint_cues_by_agreement_new(np_data_array,cues_agree,included_ILD,
                                   remove_ILD_col=True):
    assert isinstance(included_ILD,list), TypeError("included_ILD must be of "
                                                    "type list")
    filtered_np_data_array =[]
    for np_data in np_data_array:
        ITD_bool = np_data[:,3] >= 0.0 
        ILD_bool = np_data[:,4] >= 0.0
        if cues_agree:
            c1 = ITD_bool == ILD_bool
        else:
            c1 = ITD_bool != ILD_bool
        ILD_zero_bool = np.isin(np.absolute(np_data[:,4]),[0])
        c1_2 = ILD_zero_bool|c1
        c3 = np.isin(np.absolute(np_data[:,4]),included_ILD)
        idx_to_remove = [1,2,4] if remove_ILD_col else [1,2]
        selector = [x for x in range(np_data.shape[1]) if x not in
                    idx_to_remove]
        filtered_np_data = np_data[c1_2&c3][:,selector]
        filtered_np_data_array.append(filtered_np_data)
    return filtered_np_data_array


def make_ILD_ITD_joint(np_data_array,included_ILD_list,ILD_results_dict=None):
    if ILD_results_dict is None:
        ILD_results_dict = {}
    for ILD in included_ILD_list:
        filtered_np_data_array =[]
        for np_data in np_data_array:
            c1 = np.isin(np_data[:,4],[ILD])
            selector = [x for x in range(np_data.shape[1]) if x not in [1,2,4]]
            filtered_np_data = np_data[c1][:,selector]
            filtered_np_data_array.append(filtered_np_data)
        afc_results_filtered = preprocess_relative_comparisons_interpolated(filtered_np_data_array)
        afc_results_freq_combined = list(map(lambda x: sum(x,[])
                                        ,afc_results_filtered))
        ILD_results_dict[ILD] = afc_results_freq_combined
    return ILD_results_dict

def make_heatmap_from_ITD_ILD(ILD_results_dict):
    ITD_ILD_dict = {}
    for ILD,afc_array in ILD_results_dict.items():
        for arch_afc_array in afc_array:
            ITD_ILD_dict_single_arch = {}
            for exemplar in arch_afc_array:
                ITD_ILD_dict_single_arch.setdefault((exemplar[0][0],ILD),[]).append(exemplar[-1])
            for ITD_ILD_pair,is_correct_by_trial in ITD_ILD_dict_single_arch.items():
                pos_count = is_correct_by_trial.count(True)
                total_count = float(len(is_correct_by_trial))
                success_rate = pos_count / total_count
                ITD_ILD_dict.setdefault(ITD_ILD_pair,[]).append(success_rate)
    pdb.set_trace()
    heatmap_list = []
    for ITD_ILD_pair, success_rate_list in ITD_ILD_dict.items():
        avg_success_rate = sum(success_rate_list)/len(success_rate_list)
        heatmap_list.append([avg_success_rate,ITD_ILD_pair[0],ITD_ILD_pair[1]])
    cols = ['Correct', 'ITD' , 'ILD']
    np_heatmap_list = np.array(heatmap_list)
    df = pd.DataFrame(data=np_heatmap_list,columns=cols)
    return df


def preprocess_relative_comparisons_joint_ITD_ILD_interpolated(np_data_array,cues_agree=True,included_ILD_list=[20,10,5]):
    for included_ILD in included_ILD_list:
        included_ILD = included_ILD if isinstance(included_ILD,list) else list(included_ILD)
        np_data_array_filtered = filter_joint_cues_by_agreement(np_data_array,cues_agree=cues_agree,included_ILD=included_ILD)
        afc_results_array = []
        for np_data in np_data_array:
            ordered = split_ITD_tone_groups_1octv_abs(np_data)
            afc_results = compare_ITDs_across_midline(ordered)
            afc_results_array.append(afc_results)
        plot_relative_comparison_acc_multiple_lines(afc_results_array)
        plt.savefig(plots_folder+"/"+"man_added__collapsed_freqs_across_midline_interpolated_ITDfull-ILD{}_tones_jitteredPhase_no_hanning_trained_valid_padded_naturalSoundsReveb_sparse_textures_iter{}k_foldfix_collapsedArchitectures_dictdatapassing.png".format(included_ILD,man_ver,iteration))

def preprocess_relative_comparisons(np_data_array):
    afc_results_array = []
    for np_data in np_data_array:
        ordered = split_ITD_tone_groups_1octv(np_data)
        afc_results = compare_ITDs_across_midline(ordered)
        afc_results_filtered = filter_comparisons_by_offset(afc_results,30)
        afc_results_array.append(afc_results_filtered)
    return afc_results_array
        

def preprocess_relative_comparisons_interpolated(np_data_array):
    afc_results_array = []
    for np_data in np_data_array:
        ordered = split_ITD_tone_groups_1octv_abs(np_data)
        afc_results = compare_ITDs_across_midline(ordered)
        afc_results_filtered = filter_comparisons_by_offset(afc_results,600)
        afc_results_array.append(afc_results_filtered)
    return afc_results_array

def preprocess_relative_comparisons_by_offset(np_data_array,
                                              man_ver="DEFAULTVAL",
                                              training_condition = "",
                                              iteration="DEFUALTVAL"):
    offsets = set(abs(np_data_array[0][:,1])) - {0}
    for offset in offsets:
        afc_results_array = []
        for np_data in np_data_array:
            ordered = split_ITD_tone_groups_1octv(np_data)
            afc_results = compare_ITDs_across_midline(ordered)
            afc_results_filtered = filter_comparisons_by_offset_equal(afc_results,
                                                                min_cutoff=offset,
                                                                max_cutoff=offset)
            afc_results_array.append(afc_results_filtered)
        plot_relative_comparison_acc_multiple_lines(afc_results_array)
        plt.savefig(plots_folder+"/"+"man_added_{}offset_collapsed_freqs_across_midline_{}full_tones_jitteredPhase_no_hanning_trained_valid_padded_naturalSoundsReveb{}_sparse_textures_iter{}k_foldfix_collapsedArchitectures_dictdatapassing.png".format(offset,man_ver,training_condition,iteration))
    
def preprocess_relative_comparisons_by_offset_interpolated(np_data_array,
                                              man_ver="DEFAULTVAL",
                                              iteration="DEFUALTVAL"):
    offsets = set(abs(np_data_array[0][:,1])) - {0}
    for offset in offsets:
        afc_results_array = []
        for np_data in np_data_array:
            ordered = split_ITD_tone_groups_1octv_abs(np_data)
            afc_results = compare_ITDs_across_midline(ordered)
            afc_results_filtered = filter_comparisons_by_offset_equal(afc_results,
                                                                min_cutoff=offset,
                                                                max_cutoff=offset)
            afc_results_array.append(afc_results_filtered)
        plot_relative_comparison_acc_multiple_lines(afc_results_array)
        plt.savefig(plots_folder+"/"+"man_added_{}offset_collapsed_freqs_across_midline_interpolated_{}full_tones_jitteredPhase_no_hanning_trained_valid_padded_naturalSoundsReveb_sparse_textures_iter{}k_foldfix_collapsedArchitectures_dictdatapassing.png".format(offset,man_ver,iteration))

def preprocess_relative_comparisons_for_envelopes(np_data_array, 
                                                  carrier_list=None):
    afc_results_array = []
    for np_data in np_data_array:
        ordered,freqs = split_sam_tones_by_modulator_freq(np_data)
        afc_results = compare_sam_tones_across_midline(ordered)
        if carrier_list is not None:
            afc_results = filter_comparisons_by_carrier(afc_results,carrier_list,jitter_distance_octave_fraction=.1)
        #afc_results_filtered = filter_comparisons_by_offset(afc_results,30)
        afc_results_array.append(afc_results)
    return freqs,afc_results_array

def preprocess_relative_comparisons_for_transposed_envelopes(np_data_array, 
                                                  carrier_list=None):
    afc_results_array = []
    for np_data in np_data_array:
        ordered,freqs = split_sam_tones_by_modulator_freq(np_data)
        afc_results = compare_transposed_tones_across_midline(ordered)
        if carrier_list is not None:
            afc_results = filter_comparisons_by_carrier(afc_results,carrier_list)
        #afc_results_filtered = filter_comparisons_by_offset(afc_results,30)
        afc_results_array.append(afc_results)
    return freqs,afc_results_array


def plot_ILD_comparison(batch_condtionals_tone):
    test_split_ILD = split_ITD_tone_groups(test_ILD)
    print("Running Comparison...")
    results_ILD = compare_ITDs(test_split_ILD,min_dist=11,max_dist=30)
    plot_relative_comparison_acc(results_ILD)
    plt.savefig(plots_folder+"/"+"ILD_diff11-30_2AFC.png") 

def plot_ITD_comparison(batch_condtionals_tone):
    test_split_ILD = split_ITD_tone_groups(batch_condtionals_tone)
    print("Running Comparison...")
    results_ILD = compare_ITDs(test_split_ILD,min_dist=0,max_dist=4)
    plot_relative_comparison_acc(results_ILD)
    plt.savefig(plots_folder+"/"+"ITD_diff0-40_2AFC.png") 


def plot_cond_prob_azim_folded_manual_ILD(batch_conditionals):
    ILD_list = [-20,-15,-10,-5,0,5,10,15,20]
    x_axis = [i for i in range(90,280,10)]
    fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize=(30,30))
    for azim in ILD_list:
        row = ILD_list.index(azim)//3
        col = ILD_list.index(azim)%3
        a = [i[0][:36]/sum(i[0][:36]) for i in batch_conditionals if i[1] == azim] 
        azim_degrees = azim*10
        if azim_degrees > 270 or azim_degrees < 90:
            reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
        else:
            reversal_point =  azim_degrees
        np_a = np.array(a)
        folded = fold_locations_full_dist(np_a)
        folded_means,bottom_error,top_error = calc_CI(folded)
        ax[row][col].errorbar(x_axis,folded_means,yerr=[bottom_error,top_error],marker='o',markersize=2,linestyle='-')
        ax[row][col].set_title("Sounds at {}dB ILD".format(azim))
        ax[row][col].set_ylabel("Conditional Probability")
        ax[row][col].set_xlabel("Degrees")
        ax[row][col].set_ylim(0.0,1.0)
        ax[row][col].set_xticks([90,135,180,225,270])
        #ax[azim//6][azim%6].axvline(x=reversal_point, color='k', linestyle='--')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                        wspace=0.65)

def plot_cond_prob_azim_folded_manual_ITD(batch_conditionals):
    ILD_list = [-30,-20,-15,-10,-5,0,5,10,15,20,30]
    x_axis = [i for i in range(90,280,10)]
    fig, ax = plt.subplots(nrows = 4, ncols = 3, figsize=(30,30))
    for azim in ILD_list:
        row = ILD_list.index(azim)//3
        col = ILD_list.index(azim)%3
        a = [i[0][:36]/sum(i[0][:36]) for i in batch_conditionals if i[1] == azim] 
        azim_degrees = azim*10
        if azim_degrees > 270 or azim_degrees < 90:
            reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
        else:
            reversal_point =  azim_degrees
        np_a = np.array(a)
        folded = fold_locations_full_dist(np_a)
        folded_means,bottom_error,top_error = calc_CI(folded)
        ax[row][col].errorbar(x_axis,folded_means,yerr=[bottom_error,top_error],marker='o',markersize=2,linestyle='-')
        ax[row][col].set_title("Sounds at {} samples offset".format(azim))
        ax[row][col].set_ylabel("Conditional Probability")
        ax[row][col].set_xlabel("Degrees")
        ax[row][col].set_ylim(0.0,1.0)
        ax[row][col].set_xticks([90,135,180,225,270])
        #ax[azim//6][azim%6].axvline(x=reversal_point, color='k', linestyle='--')

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35,
                        wspace=0.65)

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
    plt.savefig(plots_folder+"/yost_frontal_localization_human.png",dpi=400)
    
    

def format_kulkarni_human_data():
    pd_kulkarni = pd.read_csv("/om/user/francl/kulkarni_colburn_0_azim.csv",header=[0])   
    pd_x_value = [x for x in [1024,512,256,128,64,32,16,8] for i in range(4)]
    pd_kulkarni["Smooth Factor"] = pd_x_value
    plt.clf()
    plt.figure(figsize=(10,10),dpi=200)
    order=[1024,512,256,128,64,32,16,8,4,2,1]
    sns.pointplot(x="Smooth Factor", y="Y",order=order,data=pd_kulkarni)
    plt.ylim(45,100)
    plt.ylabel("Correct (%)",fontsize=30)
    plt.xlabel("Smooth Factor",fontsize=30)
    plt.yticks(fontsize=30)
    plt.xticks(range(len(order)),order,fontsize=30,rotation=45)
    plt.text(0.25,0.90,"0 Degrees",fontsize=35,transform=plt.gca().transAxes)
    plt.axhline(5,color='k',linestyle='dashed',alpha=0.5)
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

def format_data_wood_human_data(human_data_csv=None,add_errorbars=False):
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
        pd_top = pd_wood.loc[slice(None),("Broadband Noise Top Error",slice(None))]
        pd_top.columns = pd_top.columns.droplevel(0)
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
    plt.xlabel("Source Azimuth (Degrees)",fontsize=20)
    plt.yticks([1,2,3],fontsize=15)
    plt.xticks([-50,0,50],fontsize=15,rotation=45)
    plt.tight_layout()
    #pd_wood_broadband["Y"]=(pd_y_col-pd_y_col.min())/(pd_y_col.max()-pd_y_col.min())
    
def make_wood_network_plot(regex=None):
    if regex is None:
        regex = ("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/arch_number_*init_0/"
                 "batch_conditional_broadbandNoiseRecords_wood_convolved_anechoic_"
                 "oldHRIRdist140_stackedCH_upsampled_iter100000*")
    np_data_array_dictionary = getDatafilenames(regex)
    plt.clf()
    plot_means_squared_error_by_freq(np_data_array_dictionary,azim_lim=72,
                                     bins_5deg=True,collapse_conditions=True)
    plt.savefig(plots_folder+"/azimuth_vs_error_wood_graph_network.png")

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
        pd_human,pd_model = get_wood_correlations(network_regex=model,bootstrap_mode=True)
        model_data.append(pd_model)
    pd_wood_human_norm = pd_normalize_column(pd_human)
    rmse_list = []
    for model_idx in model_choices:
        pd_model_sample = pd.concat([model_data[i] for i in model_idx])
        pd_model_mean = pd_model_sample.groupby(pd_model_sample.index).mean()
        pd_model_mean["Y"] = pd_model_mean["Y"].apply(lambda x:x*-1)
        pd_wood_network_subset_norm = pd_normalize_column(pd_model_mean)
        (kendall_tau,spearman_r,rmse) = get_stats(pd_wood_human_norm,
                                                  pd_wood_network_subset_norm)
        rmse_list.append(rmse)
    return rmse_list


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
        pd_model_mean_ITD['human'] = pd_human_ITD
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

@lru_cache(maxsize=10)
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

    wood_regex = (regex_folder+"arch_number_*init_0/batch_conditional_"
                  "broadbandNoiseRecords_wood_convolved_anechoic_"
                  "oldHRIRdist140_stackedCH_upsampled_iter{}*".format(iter_num))
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
    van_opstal_added = [sum(x) for x in zip(rmse_van_opstal_x,rmse_van_opstal_y)]
    output_dict = {'wood':rmse_wood,'yost':rmse_yost,'ITD':rmse_middlebrooks_ITD,
                   'ILD':rmse_middlebrooks_ILD,'kulkarni':rmse_kulkarni,
                   'van_opstal':van_opstal_added,'litovsky':rmse_litovsky}
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
    new_palette = ["#97B4DE","#785EF0","#DC267F","#FE6100"]
    sns.barplot(x="Training Condition",y="Network-Human Error",ci='sd',
                palette=sns.color_palette(new_palette),data=pd_raw_error)
    name = plots_folder+"/mean_error_by_training_condition.png"
    plt.yticks([0.0,0.05,0.1,0.15,0.2,0.25])
    plt.tight_layout()
    plt.savefig(name,dpi=400)
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
    sns.lineplot(x='Training Condition',y="Network-Human Error",
                units="index",estimator=None, lw=2.5,ms=5,
                data=pd_models_error,style="index",
                markers=["o","o","o","o","o","o","o","o","o","o"],
                dashes=False,alpha=0.6)
    plt.gca().get_legend().remove()
    plt.ylim(0,.41)
    plt.yticks([0,0.1,0.2,0.3,0.4])
    plt.tight_layout()
    name = plots_folder+"/individual_model_error_by_training_condition.png"
    plt.savefig(name,dpi=400)

    
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
    pdb.set_trace()
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
    idxs = [[choices+offset for offset in range(0,70,10)] for choices in np.random.choice(10,(bootstrap_iterations,10))]
    np_idxs= np.array(idxs).reshape(bootstrap_iterations,-1)
    pd_stats_list = []
    pd_stats_diff_list = []
    for i,np_idx in enumerate(np_idxs):
        pd_subset = pd_models.iloc[np_idx].groupby(['Experiment IDX']).agg([np.mean,std_ddof])
        #pd_diff_subset = (pd_models.iloc[np_idx].apply(get_diff_by_model,axis=1)
        #                  .groupby(['Experiment IDX']).agg([np.mean,std_ddof]))
        pd_stats = pd_subset.apply(cohens_d_by_model,axis=1)
        #pd_stats_diff = pd_diff_subset.apply(cohens_d_diff_by_model,axis=1)
        pd_stats_list.append(pd_stats.loc[:,(slice(None),'cohens_d')])
        #pd_stats_diff_list.append(pd_stats_diff.loc[:,(slice(None),'cohens_d')])
        print(i)
    pd_stats_all = pd.concat(pd_stats_list)
    pd_stats_melted = pd_stats_all.loc[:,(slice(None),'cohens_d')].stack(level=[0]).reset_index()
    pd_stats_melted["Experiment"] = pd_stats_melted['Experiment IDX'].apply(lambda
                                                                            x:exp_lookup[x])
    pd_stats_melted.rename(columns={'level_1':'Training Condition'},inplace=True)
    colors = sns.color_palette('colorblind')
    del colors[4]
    del colors[4]
    plt.figure(figsize=(10,8))
    sns.barplot(x='Training Condition',y='cohens_d',hue='Experiment',
                ci='sd',data=pd_stats_melted,palette=colors)
    plt.xticks([0,1,2],["Anechoic","No Background", "Unnatural"])
    plt.ylabel("Cohen's D")
    plt.ylim(-7,7)
    plt.gca().get_legend().remove()
    plt.tight_layout()
    plt.savefig(plots_folder+"/cohens_d_by_training_cond_and_experiment_sem.png",dpi=400)
    plt.clf()
    plt.figure()
    for exp in col_order:
        plt.clf()
        pd_models_error_subset= pd_models_error.query("Experiment == @exp")
        pd_raw_error_subset= pd_raw_error.query("Experiment == @exp")
        sns.barplot(x="Training Condition",y="Network-Human Error",ci='sd',
                                 order=condtion_names,data=pd_raw_error_subset)
        sns.lineplot(x='Training Condition',y="Network-Human Error",
                    units="model_index",estimator=None, lw=2,ms=4,
                    data=pd_models_error_subset,style="model_index",
                    markers=["o","o","o","o","o","o","o","o","o","o"],
                    dashes=False,alpha=0.4)
        plt.gca().get_legend().remove()
        plt.title("Network-Human Error for {} Experiment".format(exp))
        plt.tight_layout()
        name = plots_folder+("/network_human_error_by_training_cond"
                             "_experriment_{}.png".format(exp))
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
    plt.savefig(plots_folder+"/azim_vs_predicted_by_smooth_factor.png",dpi=400)
    plt.clf()
    sns.lineplot(x="elev",y="predicted_elev",hue="smooth_factor",
                 data=pd_smoothed_arch_mean,legend=False,ci=68,
                 hue_order=model_order,palette=sns.color_palette("colorblind",n_colors=9))
    plt.xticks([0,10,20,30,40,50,60],fontsize=10,rotation=75)
    plt.yticks([0,10,20,30,40,50,60],fontsize=10)
    plt.ylabel("Judged Elev",fontsize=10)
    plt.xlabel("Elev",fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_folder+"/elev_vs_predicted_by_smooth_factor.png",dpi=400)


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
                palette=sns.xkcd_palette(["black","white"]),
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
                     y="predicted_folded",data=pd_filtered_precedence_arch_mean,err_style='bars',ci=68)
        format_precedence_effect_graph()
        plt.tight_layout()
        plt.savefig(plots_folder+"/"+"precedence_effect_graph_lineplot_clicks.png")

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
    new_palette = ["#97B4DE","#785EF0","#DC267F","#FE6100"]
    sns.lineplot(x="delay",y="predicted_folded",hue="Training Condition",
                 data=pd_filtered_precedence_arch_mean_all,err_style='bars',ci=68,
                 palette=sns.color_palette(new_palette))
    format_precedence_effect_graph()
    plt.gca().get_legend().remove()
    plt.xticks([0,10,20,30,40,50])
    plt.yticks([-40,-20,0,20,40])
    plt.tight_layout()
    plt.savefig(plots_folder+"/"+"precedence_effect_graph_lineplot_clicks_across_conditions.png",dpi=400)
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
    sns.barplot(x="delay",y="RMSE",hue="Error Condition",
                palette=sns.xkcd_palette(["black","white"]),
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
    pd_mean_values = make_van_opstal_paper_plot_bootstrapped(pd_collapsed,columns_to_plot,'ko') 
    pd_ref_human = pd_collapsed['Reference Grid JO']
    if hardcode_ref_grid:
        grid = pd.Series([(x,y) for y in [20,6.667,-6.667,-20]
                          for x in [-20,-6.667,6.667,20]])
        make_van_opstal_paper_plot(grid,'ko--',fill_markers=False)
    else:
        make_van_opstal_paper_plot(pd_collapsed['Reference Grid JO'],'ko--',fill_markers=False)
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
    pdb.set_trace()
    num_total_sub = pd_cipic_filtered['subject_num'].unique().shape[0]-1
    pd_dataframe_cipic_subject_quartile = pd_cipic_filtered.query("subject_num in @quartile_sujects")
    pd_dataframe_cipic_new_ears = pd_cipic_filtered.query("subject_num != 999")
    plt.clf()
    ax1=sns.lineplot(x='elev',y='predicted_elev',hue="Ears Used",data=pd_cipic_filtered,
                     style="Ears Used",markers=["o","o"],lw=3.5,ms=8.5,err_style="bars",
                     ci=68,dashes=False)
    handles,labels = plt.gca().get_legend_handles_labels()
    ax2 = sns.lineplot(x='elev',y='predicted_elev',hue="subject_num",
                  data=pd_dataframe_cipic_subject_quartile,
                 palette=sns.color_palette(n_colors=1)*num_quartile_sub,
                 style="subject_num",markers=["o"]*num_quartile_sub,dashes=False,
                 err_style='bars',ci=68,lw=2,ms=5,alpha=0.8,ax=ax1)
    ax3 = sns.lineplot(x='elev',y='predicted_elev',hue="subject_num",
                  data=pd_dataframe_cipic_new_ears,
                 palette=sns.color_palette(n_colors=1)*num_total_sub,
                 style="subject_num",markers=["o"]*num_total_sub,dashes=False,
                 err_style='bars',ci=68,lw=1,ms=0,alpha=0.2,ax=ax2)
    plt.legend(handles,labels)
    plt.xlabel("Elevation (Degrees)")
    plt.ylabel("Predicted Elevation (Degrees)")
    plt.xticks([0,10,20,30,40,50],rotation=75)
    plt.yticks([0,10,20,30,40,50])
    plt.tight_layout()
    plt.savefig(plots_folder+"/"+"elev_predicted_vs_label_all_subjects_vs_CIPIC_elev_collapsed.png")
    plt.clf()
    ax1=sns.lineplot(x='azim',y='predicted_azim',hue="Ears Used",data=pd_cipic_filtered,
                     style="Ears Used",markers=["o","o"],lw=3.5,ms=8.5,err_style="bars",
                     ci=68,dashes=False)
    handles,labels = plt.gca().get_legend_handles_labels()
    ax2 = sns.lineplot(x='azim',y='predicted_azim',hue="subject_num",
                  data=pd_dataframe_cipic_subject_quartile,
                 palette=sns.color_palette(n_colors=1)*num_quartile_sub,
                 style="subject_num",markers=["o"]*num_quartile_sub,dashes=False,
                 err_style='bars',ci=68,lw=2,ms=5,alpha=0.8,ax=ax1)
    ax3 = sns.lineplot(x='azim',y='predicted_azim',hue="subject_num",
                  data=pd_dataframe_cipic_new_ears,
                 palette=sns.color_palette(n_colors=1)*num_total_sub,
                 style="subject_num",markers=["o"]*num_total_sub,dashes=False,
                 err_style='bars',ci=68,lw=1,ms=0,alpha=0.2,ax=ax2)
    plt.xlabel("Azimuth (Degrees)")
    plt.ylabel("Predicted Azimuth (Degrees)")
    plt.xticks([-75,-50,-25,0,25,50,75],rotation=75)
    plt.yticks([-75,-50,-25,0,25,50,75])
    plt.tight_layout()
    plt.savefig(plots_folder+"/"+"azim_predicted_vs_label_all_subjects_vs_CIPIC_elev_collapsed.png")
    






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

def make_van_opstal_paper_plot(pd_collapsed_col,fmt_str,fill_markers=True,color=None):
    mfc = None if fill_markers else 'none'
    for counter in range(4):
        row = 4*counter
        col = counter
        for i in range(3):
            x_row,y_row = zip(pd_collapsed_col.iloc[row+i],pd_collapsed_col.iloc[row+i+1])
            if color is not None:
                plt.plot(x_row,y_row,fmt_str,mfc=mfc,color=color)
            else:
                plt.plot(x_row,y_row,fmt_str,mfc=mfc)
            x_col,y_col = zip(pd_collapsed_col.iloc[col+4*i],pd_collapsed_col.iloc[col+4*(i+1)])
            if color is not None:
                plt.plot(x_col,y_col,fmt_str,mfc=mfc,color=color)
            else:
                plt.plot(x_col,y_col,fmt_str,mfc=mfc)

def make_van_opstal_paper_plot_network_data(pd_data_cipic,
                                            azim_list=[-20,-10,10,20],
                                            elev_list=list(range(0,40,10)),
                                            fmt_str="ko",colors=None):
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
                plt.errorbar(x_mean,y_mean,xerr,yerr,fmt_str,mfc=color,mec=color,ecolor='darkgray')
        
        query_string = ('azim in @azim_list & elev in @elev_list'.format(azim,elev))
        pd_CI_normal_filtered = pd_CI.query(query_string)
        pd_CI_sorted = pd_CI_normal_filtered.sort_values(['elev','azim'],ascending=[False,True]) 
        pd_CI_means = pd_CI_sorted[['predicted_azim','predicted_elev']].apply(lambda x:(x['predicted_azim'][0],x['predicted_elev'][0]),axis=1)
        make_van_opstal_paper_plot(pd_CI_means,fmt_str.replace("o",""),color=color)
        plt.ylim(-10,40)
        plt.xlim(-30,30)
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        plt.ylabel("Elevation (Degrees)",fontsize=20)
        plt.xlabel("Azimuth (Degrees)",fontsize=20)
        plt.tight_layout()
        plt.savefig(plots_folder+"/van_opstal_network_sem_{}.png".format(condtion),dpi=400)
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
    make_van_opstal_paper_plot(pd_refernce_grid['xy'],'ko--',fill_markers=False)
    return pd_refernce_grid[['x','y']]


def make_van_opstal_paper_plot_bootstrapped(pd_dataframe,columns,fmt_str):
    pd_bs = pandas_bootstrap(pd_dataframe,columns,alpha=0.314)
    xerr = pd.DataFrame(pd_bs['xerr'].apply(pd.Series)).values.T
    yerr = pd.DataFrame(pd_bs['yerr'].apply(pd.Series)).values.T
    plt.errorbar(pd_bs["x"],pd_bs["y"],xerr,yerr,fmt_str,ecolor='darkgray')
    pd_bs["xy"] = pd_bs[['x', 'y']].apply(tuple, axis=1)
    make_van_opstal_paper_plot(pd_bs["xy"],fmt_str.replace("o",""))
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
    pd_smoothed_hrir_mismatch_arch_mean = pd_smoothed_hrir_mismatch.groupby(["Smooth Factor","Architecture Number"]).mean().reset_index
    sns.pointplot(x="Smooth Factor",y="Total Error",
                  order=order,color='k',data=pd_smoothed_hrir_mismatch_arch_mean)
    plt.xticks(range(len(order)),order,fontsize=30,rotation=45)
    plt.ylabel("Spatial Error (Degrees)",fontsize=30)
    plt.xlabel("Smooth Factor",fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.savefig(plots_folder+"/"+"smooth_factor_vs_total_error_smoothed_fill_length_hrir_mathched_ylim.png")


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
    pd_dataframe_middlebrooks_wrightman_std_filtered=pd_dataframe_middlebrooks_wrightman_ITD_filtered[pd_dataframe_middlebrooks_wrightman_ITD_filtered['std']<= 14]
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
        plt.savefig(plots_folder+"/"+"ILD_residuals_regression_{}.png".format(low_high_cutoff))
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
        plt.savefig(plots_folder+"/"+"ITD_residuals_regression_{}.png".format(low_high_cutoff))
        if extract_lines:
            pd_data = get_data_from_graph(use_points=True)
            extracted_data.append(pd_data)
    if extract_lines:
        return extracted_data

def make_ILD_swarm_graph(pd_dataframe_middlebrooks_wrightman_formatted):
    pd_dataframe_middlebrooks_wrightman_formatted = pd_dataframe_middlebrooks_wrightman_formatted[pd_dataframe_middlebrooks_wrightman_formatted['azim'].isin([70,65,60,55,50,45,40,35,30,25,20,15,10,5,0,355,350,345,340,335,330,325,320,315,310,305,300,295,290])]
    pd_dataframe_middlebrooks_wrightman_ITD_filtered = pd_dataframe_middlebrooks_wrightman_formatted[pd_dataframe_middlebrooks_wrightman_formatted['ITD'].isin([0])]
    pd_dataframe_middlebrooks_wrightman_ITD_filtered['low_high_cutoff'] = pd_dataframe_middlebrooks_wrightman_ITD_filtered[['low_cutoff','high_cutoff']].apply(lambda x: str(tuple(x)),axis=1)
    pd_unique_azims = pd_dataframe_middlebrooks_wrightman_ITD_filtered['azim'].unique()
    for azim in pd_unique_azims:
        query_string = "azim == {}".format(azim)
        pd_dataframe_middlebrooks_wrightman_azim_filtered=pd_dataframe_middlebrooks_wrightman_ITD_filtered.query(query_string)
        plt.clf()
        sns.swarmplot(x='ILD',y='predicted_folded',hue='low_high_cutoff',data=pd_dataframe_middlebrooks_wrightman_azim_filtered)
        plt.ylim(-90,90)
        plt.title("ILD sensititivity swarm plot at {} Azim".format(azim))
        plt.savefig(plots_folder+"/"+"ILD_sensitivity_vs_angle_swarmplot_at_{}_azim.png".format(azim))


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
    
def open_SDR_data(fname):
    with open(fname,'r') as f:
        SDR_list = json.load(f)
    cols = ["SNR","NOISE","SDR_input_mean","SDR_processed_mean","SDR_input_var","SDR_processed_var","IDX"]
    pd_SDR_data = pd.DataFrame(columns=cols)
    for stim_dict in SDR_list:
        pd_SDR_data = pd_SDR_data.append(stim_dict,ignore_index=True)
    pd_SDR_data['SDR Improvement (dB)'] = pd_SDR_data["SDR_processed_mean"] - pd_SDR_data["SDR_input_mean"]
    pd_SDR_data['SNR'] = pd_SDR_data['SNR'].apply(float_from_string_of_bytes)
    pd_SDR_data = pd_SDR_data.convert_objects(convert_numeric=True)
    return pd_SDR_data

def afc_to_pd(afc_results_array,afc_arch_names):
    cols = ['offset','frequency','predicted_difference','ground_truth_direction',
            'correct','arch_index','init_index']
    for afc_results,arch_name,init_name in zip(afc_results_array,*afc_arch_names):
        flat_list = flatten(afc_results)
        arch_afc_results = [[abs(x[0][0]),x[1][1],x[2],sign(x[3]),
                            x[4],arch_name,init_name] for x in flat_list]
        np_afc_results = np.array(arch_afc_results)
        df = pd.DataFrame(data=np_afc_results,columns=cols)
        main_df_afc = df if 'main_df_afc' not in locals() else pd.concat([main_df_afc,df]).reset_index(drop=True)
    main_df_afc['frequency_bin'] = pd_afc_large_step.apply(assign_ITD_tone_group_abs,axis=1)
    return main_df_afc
        

low_ci = lambda row: row['mean'] - 1.96*row['std']/math.sqrt(row['count'])
high_ci = lambda row: row['mean'] + 1.96*row['std']/math.sqrt(row['count'])

def find_adaptive_threshold(pd_afc_results,start_value=300):
    afc_grouped = pd_afc_results.groupby(['frequency_bin','offset'])
    afc_correct_stats = afc_grouped['correct'].agg([np.mean,'count',np.std]).reset_index()
    afc_correct_stats['low_95ci'] = afc_correct_stats.apply(low_ci,axis=1)
    afc_correct_stats['high_95ci'] = afc_correct_stats.apply(high_ci,axis=1)
    afc_correct_stats_grouped = afc_correct_stats.groupby(['frequency_bin'])
    freq_min_ITD = {}
    for freq_bin,group in afc_correct_stats_grouped:
        for row_num,row in group.sort_values(by=['offset'],ascending=False).iterrows():
            if row['offset'] > start_value: continue
            if row['low_95ci'] < 0.5:break
            freq_min_ITD[freq_bin] = row['offset']
    return freq_min_ITD

def find_adaptive_threshold_mean(pd_afc_results,start_value=300,threshold=0.60):
    afc_grouped = pd_afc_results.groupby(['frequency_bin','offset'])
    afc_correct_stats = afc_grouped['correct'].agg([np.mean,'count',np.std]).reset_index()
    afc_correct_stats['low_95ci'] = afc_correct_stats.apply(low_ci,axis=1)
    afc_correct_stats['high_95ci'] = afc_correct_stats.apply(high_ci,axis=1)
    afc_correct_stats_grouped = afc_correct_stats.groupby(['frequency_bin'])
    freq_min_ITD = {}
    for freq_bin,group in afc_correct_stats_grouped:
        for row_num,row in group.sort_values(by=['offset'],ascending=False).iterrows():
            if row['offset'] > start_value: continue
            if row['mean'] < threshold:  break
            freq_min_ITD[freq_bin] = row['offset']
    return freq_min_ITD

def make_psychometric_functions(pd_afc_results,start_value=300):
    afc_grouped = pd_afc_results.groupby(['frequency_bin','offset'])
    afc_correct_stats = afc_grouped['correct'].agg([np.mean,'count',np.std]).reset_index()
    afc_correct_stats['low_95ci'] = afc_correct_stats.apply(low_ci,axis=1)
    afc_correct_stats['high_95ci'] = afc_correct_stats.apply(high_ci,axis=1)
    afc_correct_stats_grouped = afc_correct_stats.groupby(['frequency_bin'])
    freq_min_ITD = {}
    for freq_bin,group in afc_correct_stats_grouped:
        plt.clf()
        sns.lineplot(x="offset",y="mean",data=group)
        plt.ylim(0,10)
        plt.savefig(plots_folder+"/"+"psychometric_function_at_freq{}_trained_reverb_bkgd5db40dBSNR_large_step.png".format(freq_bin))
    
def interactive_adaptive_threshold(pd_afc_results,start_value=300):
    afc_grouped = pd_afc_results.groupby(['frequency_bin','offset'])
    afc_correct_stats = afc_grouped['correct'].agg([np.mean,'count',np.std]).reset_index()
    afc_correct_stats['low_95ci'] = afc_correct_stats.apply(low_ci,axis=1)
    afc_correct_stats['high_95ci'] = afc_correct_stats.apply(high_ci,axis=1)
    afc_correct_stats_grouped = afc_correct_stats.groupby(['frequency_bin'])
    freq_min_ITD = {}
    for freq_bin,group in afc_correct_stats_grouped:
        sorted_group = group.sort_values(by=['offset'],ascending=False)
        
        pdb.set_trace()
        for row_num,row in sorted_group.iterrows():
            if row['offset'] > start_value: continue
            if row['low_95ci'] < 0.5:break
            freq_min_ITD[freq_bin] = row['offset']
    return freq_min_ITD

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
                y="Predicted Position (Degrees)", height=8,
                aspect=1.45,inner='quartile',
                data=pd_dataframe_folded.astype('int32'),palette=rgb_cmap)
    format_binarural_test_folded_graph()
    plt.tight_layout()
    plt.savefig(plots_folder+"/"+("binaural_recorded_4078_main_kemar_full_spec_folded"
                                  "_violinplot_quartile_grayscale_front_limited.png"))

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
    plt.savefig(plots_folder+"/"+("binaural_recorded_4078_main_kemar_full_spec_folded"
                                  "_violinplot_quartile_grayscale_centered.png"),dpi=400)

def fig_4b():
    format_data_wood_human_data("/om/user/francl/wood_localization_errorbar.csv",add_errorbars=True)
    plt.savefig(plots_folder+"/azimuth_vs_error_wood_human.png",dpi=400)

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
    plt.xticks([-50,0,50])
    plt.savefig(plots_folder+"/azimuth_vs_error_wood_graph_network.png",dpi=400)

def fig_4e():
    make_bandwidth_vs_error_humabn_plot()

def fig_4f(regex=None,iter_num=100000):
    if regex is not None:
        regex = regex + ("arch_number_*init_0/batch_conditional*bandpass"
                  "*HRIR*iter{}.npy".format(iter_num))
    make_yost_network_plot(regex)
    plt.ylim(0,35)
    plt.savefig(plots_folder+"/bandwidth_vs_error_network_plot.png")

def fig_5b_and_5c():
    plt.clf()
    get_van_opstal_human_plot("before",hardcode_ref_grid=True)
    plt.savefig(plots_folder+"/van_opstal_before_human_hardcoded_ref.png",dpi=400)
    plt.clf()
    get_van_opstal_human_plot("after",hardcode_ref_grid=True)
    plt.savefig(plots_folder+"/van_opstal_after_human_hardcoded_ref.png",dpi=400)

def fig_5d_and_5e(regex=None,iter_num=100000):
    if regex is not None:
        regex = regex+ ("arch_number_[0-9][0,1,3-9][0-9]_init_0/"
                        "batch_conditional_broadbandNoiseRecords_"
                        "convolvedCIPIC*iter{}.npy".format(iter_num))
    make_van_opstal_network_plots(regex)

def fig_5f_and_g(regex=None):
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
    plt.savefig(plots_folder+"/litovsky_errorbars_seaborn.png",dpi=400)

def fig_6d(regex=None,iter_num=100000):
    if regex is not None:
        regex = regex + ("arch_number_*init_0/batch_conditional_*"
                        "precedence*multiAzim*pinknoise*5degStep*iter{}*".format(iter_num))
    litovsky_error_plot_multi_azim(regex=regex)
    plt.gca().legend().remove()
    plt.savefig(plots_folder+"/precedence_effect_litovsky_rmse_plots_network.png",dpi=400)

def fig_7b_and_7c():
    get_individual_error_graph(["/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/",
                                "/om2/user/francl/new_task_archs/new_task_archs_anechoic_training/",
                                "/om2/user/francl/new_task_archs/new_task_archs_no_background_noise_80dBSNR_training/",
                                "/om2/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_training/"],
                               iter_nums=[100000,100000,100000,150000],
                               condtion_names=["Normal", "Anechoic", "No Background","Unnatural Sounds"]) 

def fig_7e():
    get_error_graph_by_task(["/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/",
                             "/om2/user/francl/new_task_archs/new_task_archs_anechoic_training/",
                             "/om2/user/francl/new_task_archs/new_task_archs_no_background_noise_80dBSNR_training/",
                             "/om2/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_training/"],
                            iter_nums=[100000,100000,100000,150000],
                            condtion_names=["Normal", "Anechoic", "No Background","Unnatural Sounds"])

def fig_7d():
    make_click_precedence_effect_across_conditions()

def make_alternative_training_graphs(function,regex_list=None,
                                     output_folder_list=None):
    global plots_folder
    if regex_list is None:
        regex_list = ["/om5/user/francl/new_task_archs/new_task_archs_anechoic_training/",
                     "/om5/user/francl/new_task_archs/new_task_archs_no_background_noise_80dBSNR_training/",
                     "/om5/user/francl/new_task_archs/new_task_archs_logspaced_0.5octv_fluctuating_noise_training/"]
        iter_nums = [100000,100000,150000]
        output_folder_list = ["/anechoic_training","/noiseless_training","/unnatural_sounds_training"]
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




