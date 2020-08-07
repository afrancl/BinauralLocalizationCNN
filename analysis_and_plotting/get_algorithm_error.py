import numpy as np
from scipy.io.wavfile import read
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import datetime
import os
import sys
import pdb
import json
import math
from glob import glob
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats



SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 50

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure titleo
colors =  sns.color_palette('colorblind')
del colors[4]
del colors[4]
sns.set_palette(colors)



mic_offset = int(sys.argv[1])

c = 343.    # speed of sound
fs = 44100  # sampling frequency
nfft = 512  # FFT size
freq_range = [30, 20000]

plots_folder='/om/user/francl/{date:%Y-%m-%d}_plots'.format(date=datetime.datetime.now())

try:
    os.makedirs(plots_folder)
except OSError:
    if not os.path.isdir(plots_folder):
        raise


def getDatafilenames(regex):
    fnames = glob(regex)
    print("FOUND {} FILES:".format(len(fnames)))
    np_data_list = []
    for fname in fnames:
        keylist_regex = fname.replace("batch_conditional","keys_test").replace(".npy","")+"*"
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
        temp_array = np.load(fname)
        reshaped_array = reshape_and_filter(temp_array,label_order)
        np_data_list.append(reshaped_array)
    return np_data_list


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

def make_dataframe(regex,fold_data=False,elevation_predictions=False):
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
    if fold_data:
        main_df['predicted_folded'] = main_df.apply(add_fold_offset, axis=1)
    return main_df



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
    elif "CIPIC" in regex:
        var_order = ["azim", "elev", "subject", "noise_idx"]
    elif "samTone" in regex:
        var_order = ["carrier_freq", "modulation_freq",
                     "carrier_delay", "modulation_delay",
                     "flipped"]
    elif "transposed" in regex:
        var_order = ["carrier_freq", "modulation_freq",
                     "delay", "flipped"]
    elif "precedence" in regex:
        var_order = ["delay","start_sample",
                     "lead_level","lag_level", "flipped"]
    elif "bandpass" in regex:
        var_order = ["azim","elev", "bandwidth",
                     "center_freq"]
    elif ("binaural_recorded" in regex) and ("speech_label" in regex):
        var_order = ["azim", "elev","speech"]
    elif "binaural_recorded" in regex:
        var_order = ["azim", "elev"]
    else:
        var_order = ["azim","freq"]
    return var_order
        

def get_order(regex,data):
    var_order = get_cols(regex)
    key_swap_indicies = [next((idx for idx,pos in enumerate(var_order) if pos in key), None)
                         for key in data]
    return key_swap_indicies




def collapse_degrees_to_bin_indicies(df):
    '''
    Takes a dataframe with column 'azim' containing positions in degrees and
    modifies the dataframe in place to replace the location in degrees with the
    bin index in 5 degrees increments

    Arguments:
        df (pandas dataframe): needs column azim with source locations in
        degrees

    Returns:
        df (pandas dataframe): returns column azim with source locations in bin
        indicies
    '''
    msg = "There must be and azimuth column in the provided DataFrame."
    assert "azim" in df,msg 
    df['azim'] = df['azim'].map(lambda x: x/5)
    return df



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

def add_fold_offset_df(row):
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

print("Mic Offset:",mic_offset)
df = pd.DataFrame(columns=["predicted_folded","azim","predicted","algorithm","source_name"])
for azim in range(0,360,30):
    print(azim)
    azim_binned = azim/5
    soruce_files = glob('/om/user/francl/recorded_binaural_audio_4078_main_2_mic_array_rescaled/*_{}_azim.wav'.format(azim))
    for fname in soruce_files:
        freq,stim = read(fname)
        source_name = os.path.basename(fname)
        stim= stim.T
        X = np.array([pra.stft(signal, nfft, nfft // 2, transform=np.fft.rfft).T for
                      signal in stim])

        algo_names = ['SRP', 'MUSIC', 'TOPS','CSSM','WAVES']
        spatial_resp = dict()

        microphone = np.array([[0-(mic_offset*0.01)/2.0, 0],[0+(mic_offset*0.01)/2.0, 0]]).T
        # loop through algos
        for algo_name in algo_names:
            # Construct the new DOA object
            # the max_four parameter is necessary for FRIDA only
            doa = pra.doa.algorithms[algo_name](microphone, fs, nfft, c=c, num_src=1, max_four=4,n_grid=72)

            # this call here perform localization on the frames in X
            doa.locate_sources(X, freq_range=freq_range)
            
            # store spatial response
            if algo_name is 'FRIDA':
                spatial_resp[algo_name] = np.abs(doa._gen_dirty_img())
            else:
                spatial_resp[algo_name] = doa.grid.values
                
            # normalize   
            min_val = spatial_resp[algo_name].min()
            max_val = spatial_resp[algo_name].max()
            spatial_resp[algo_name] = (spatial_resp[algo_name] - min_val) / (max_val - min_val)
        for k,v in spatial_resp.items():
            rolled_response = np.roll(v,-18)
            predicted = rolled_response.argmax()
            predicted_folded = fold_locations_full_dist_5deg(rolled_response).argmax()
            predicted_folded_rolled = add_fold_offset(predicted_folded,predicted,azim_binned)
            df = df.append({
                    "predicted_folded": predicted_folded_rolled,
                    "azim": azim_binned,
                    "predicted": predicted,
                    "algorithm": k,
                    "source_name": source_name
                      }, ignore_index=True)
fname_pkl = '/om/user/francl/baseline_algorithm_results_0PosMic/pd_kemar_offset{}'.format(mic_offset)
df.to_pickle(fname_pkl)

def run_at_azim(azim):
    azim_binned = azim/5
    soruce_files = glob('/om/user/francl/recorded_binaural_audio_4078_main_kemar_rescaled/*_{}_azim.wav'.format(azim))
    df = pd.DataFrame(columns=["azim","predicted","algorithm","source_name"])
    for fname in soruce_files[:7]:
        freq,stim = read(fname)
        source_name = os.path.basename(fname)
        stim= stim.T
        X = np.array([pra.stft(signal, nfft, nfft // 2, transform=np.fft.rfft).T for
                      signal in stim])

        algo_names = ['SRP', 'MUSIC', 'TOPS','CSSM','WAVES']
        spatial_resp = dict()

        microphone = np.array([[0-(mic_offset*0.01)/2.0, 0],[0+(mic_offset*0.01)/2.0, 0]]).T
        # loop through algos
        for algo_name in algo_names:
            # Construct the new DOA object
            # the max_four parameter is necessary for FRIDA only
            doa = pra.doa.algorithms[algo_name](microphone, fs, nfft, c=c, num_src=1, max_four=4,n_grid=72)

            # this call here perform localization on the frames in X
            doa.locate_sources(X, freq_range=freq_range)
            
            # store spatial response
            if algo_name is 'FRIDA':
                spatial_resp[algo_name] = np.abs(doa._gen_dirty_img())
            else:
                spatial_resp[algo_name] = doa.grid.values
                
            # normalize   
            min_val = spatial_resp[algo_name].min()
            max_val = spatial_resp[algo_name].max()
            spatial_resp[algo_name] = (spatial_resp[algo_name] - min_val) / (max_val - min_val)
        for k,v in spatial_resp.items():
            rolled_response = np.roll(v,-18)
            predicted = rolled_response.argmax()
            predicted_folded = fold_locations_full_dist_5deg(rolled_response).argmax()
            predicted_folded_rolled = add_fold_offset(predicted_folded,predicted,azim_binned)
            df = df.append({
                    "predicted_folded": predicted_folded_rolled,
                    "azim": azim_binned,
                    "predicted": predicted,
                    "algorithm": k,
                    "source_name": source_name
                      }, ignore_index=True)
    return df


error = lambda x, y: min(72-(abs(x-y)),abs(x-y))


def get_error_all_mic_offsets(filepath):
    df_error_all_conditions = None
    for mic_offset in range(10,38,2):
        fname = filepath + '/pd_kemar_offset{}'.format(mic_offset)
        df = pd.read_pickle(fname)
        df_error = get_errror(df)
        df_error['mic_offset'] = mic_offset
        if df_error_all_conditions is None:
            df_error_all_conditions = df_error
        df_error_all_conditions = pd.concat([df_error_all_conditions,df_error])
    return df_error_all_conditions


def get_error_per_mic_offset(df_error_all_conditions):
    df_average_rms = pd.DataFrame(columns=["algorithm","rms_folded","rms","mic_offset"])
    for mic_offset in range(10,38,2):
        query_string  ='mic_offset =={}'.format(mic_offset)
        average_rms_error_folded = df_error_all_conditions.query(query_string)['rms_folded'].mean()
        average_rms_error = df_error_all_conditions.query(query_string)['rms'].mean()
        df_average_rms = df_average_rms.append({
                                        "rms_folded": average_rms_error_folded,
                                        "rms": average_rms_error,
                                        "mic_offset": mic_offset,
                                          }, ignore_index=True)
    return df_average_rms



        
def get_errror(pd_localization_results):
    error = lambda x, y: min(72-(abs(x-y)),abs(x-y))
    df = pd.DataFrame(columns=["azim","algorithm","rms_folded","rms"])
    for azim in range(0,360,30):
        azim_binned = azim/5
        algo_names = ['SRP', 'MUSIC', 'TOPS','CSSM','WAVES']
        for algo_name in algo_names:
            query_string = 'algorithm == "{}" & azim == {}'.format(algo_name,
                                                                   int(azim_binned))
            predictions_folded = pd_localization_results.query(query_string)['predicted_folded']
            predictions = pd_localization_results.query(query_string)['predicted']
            error_folded = predictions_folded.apply(lambda x: error(azim_binned,x))
            azim_error = predictions.apply(lambda x: error(azim_binned,x))
            rms_folded = math.sqrt((error_folded**2).mean())
            rms = math.sqrt((azim_error**2).mean())
            df = df.append({
                    "rms_folded": rms_folded,
                    "azim": azim_binned,
                    "rms": rms,
                    "algorithm": algo_name
                      }, ignore_index=True)
    return df

def get_error_cnn(pd_localization_results):
    error = lambda x, y: min(72-(abs(x-y)),abs(x-y))
    df = pd.DataFrame(columns=["azim","algorithm","rms_folded","rms","arch_index"])
    arch_index_set = pd_localization_results["arch_index"].unique()    
    for arch_index in arch_index_set:
        for azim in range(0,360,30):
            azim_binned = azim/5
            algo_names = ['CNN (Ours)']
            for algo_name in algo_names:
                query_string = 'algorithm == "{}" & azim == {} & arch_index == {}'.format(algo_name,
                                                                       int(azim_binned),int(arch_index))
                predictions_folded = pd_localization_results.query(query_string)['predicted_folded']
                predictions = pd_localization_results.query(query_string)['predicted']
                error_folded = predictions_folded.apply(lambda x: error(azim_binned,x))
                azim_error = predictions.apply(lambda x: error(azim_binned,x))
                rms_folded = math.sqrt((error_folded**2).mean())
                rms = math.sqrt((azim_error**2).mean())
                df = df.append({
                        "rms_folded": rms_folded,
                        "azim": azim_binned,
                        "rms": rms,
                        "algorithm": algo_name,
                        "arch_index": arch_index
                          }, ignore_index=True)
    return df
        
def make_azim_vs_folded_rms_plot(df_rms):
    plt.clf()
    if 'azim' in df_rms:
        df_rms['azim'] = df_rms['azim'].map(lambda x: x*5)
        df_rms['rms_folded'] = df_rms['rms_folded'].map(lambda x: x*5)
        df_rms.columns = ['Azimuth (Degrees)', 'DOA Algorithm', 
                          'Folded RMS (Degrees)', 'RMS (Degrees)',
                          'Architecture Index']
    sns.lineplot(x="Azimuth (Degrees)", y="Folded RMS (Degrees)", hue="DOA Algorithm", style="DOA Algorithm", data=df_rms)
    
def count_front_back_mistakes(pd_localization_results):
    is_mistake = lambda azim_binned,pred_loc : True if ((18 < azim_binned < 54)
                                                != (18 < pred_loc < 54) and 
                                                (azim_binned not in [18,54])) else False
    df = pd.DataFrame(columns=["azim","algorithm","front_back_mistakes","total_predictions"])
    for azim in range(0,360,30):
        azim_binned = azim/5
        if azim_binned in [18,54]:
            continue
        algo_names = pd_localization_results['algorithm'].unique()
        for algo_name in algo_names:
            query_string = 'algorithm == "{}" & azim == {}'.format(algo_name,
                                                                   int(azim_binned))
            prediction_counts = dict(pd_localization_results.query(query_string)['predicted'].value_counts())
            front_back_mistakes =  sum([v for k,v in
                                        prediction_counts.items() if
                                        is_mistake(azim_binned,k)])
            total_count = sum([v for k,v in prediction_counts.items()])
            df = df.append({
                    "front_back_mistakes": front_back_mistakes,
                    "azim": azim_binned,
                    "algorithm": algo_name,
                    "total_predictions": total_count
                      }, ignore_index=True)
    return df

def sum_over_azim(df_front_back_mistakes):
    df = pd.DataFrame(columns=["algorithm","front_back_mistakes","total_predictions"])
    algo_names = df_front_back_mistakes['algorithm'].unique()
    for algo_name in algo_names:
        query_string = 'algorithm == "{}"'.format(algo_name)
        front_back_total_counts = df_front_back_mistakes.query(query_string)['front_back_mistakes'].sum()
        prediction_total_counts = df_front_back_mistakes.query(query_string)['total_predictions'].sum()
        df = df.append({
                "front_back_mistakes": front_back_total_counts,
                "algorithm": algo_name,
                "total_predictions": prediction_total_counts
                  }, ignore_index=True)
    return df

def make_algorithm_vs_front_back_mistakes_plot(df_rms):
    plt.clf()
    if 'algorithm' in df_rms:
        df_rms['proportion'] = \
                df_rms['front_back_mistakes']/df_rms['total_predictions']
        df_rms.columns = ['DOA Algorithm','front_back_mistakes' ,'total_predictions',
                          'Proportion of Front/Back Mistakes']
    sns.barplot(x="DOA Algorithm", y="Proportion of Front/Back Mistakes",
                 data=df_rms)

def fold_pd_data_for_plotting(df_rms,algo_names = ['SRP', 'MUSIC', 'TOPS','CSSM','WAVES']):
    if 'azim' in df_rms:
        df_rms['azim'] = df_rms['azim'].map(lambda x: x*5)
        df_rms['rms_folded'] = df_rms['rms_folded'].map(lambda x: x*5)
    df = pd.DataFrame(columns=['Azimuth (Degrees)', 'DOA Algorithm', 'Folded RMS (Degrees)','RMS (Degrees)'])
    for algo_name in algo_names:
        query_string = 'algorithm == "{}"'.format(algo_name)
        df_rms_algo = df_rms.query(query_string)
        for azim_pair in [[0,180],[30,150],[60,120],[330,210],[300,240],[90],[270]]:
            rows = df_rms_algo[df_rms_algo['azimh'].isin(azim_pair)]
            rms_avg = rows['rms'].mean() 
            rms_folded_avg = rows['rms_folded'].mean()
            azim = azim_pair[0] if azim_pair[0] <= 91 else azim_pair[0]-360
            df = df.append({
                    "Folded RMS (Degrees)": rms_folded_avg,
                    "DOA Algorithm": algo_name,
                    "Azimuth (Degrees)": azim,
                    "RMS (Degrees)": rms_avg
                      }, ignore_index=True)
    return df
            


def fold_pd_error_data_for_plotting(df_rms,algo_names = ['SRP', 'MUSIC', 'TOPS','CSSM','WAVES']):
    if 'azim' in df_rms:
        df_rms['azim'] = df_rms['azim'].map(lambda x: x*5)
        df_rms['rms_folded'] = df_rms['rms_folded'].map(lambda x: x*5)
    df = pd.DataFrame(columns=['Azimuth (Degrees)', 'DOA Algorithm', 'Folded RMS (Degrees)','RMS (Degrees)','Architecture Index'])
    arch_index_set = df_rms['arch_index'].unique()    
    for arch_index in arch_index_set:
        for algo_name in algo_names:
            query_string = 'algorithm == "{}" & arch_index == {}'.format(algo_name,int(arch_index))
            df_rms_algo = df_rms.query(query_string)
            for azim_pair in [[0,180],[30,150],[60,120],[330,210],[300,240],[90],[270]]:
                rows = df_rms_algo[df_rms_algo['azim'].isin(azim_pair)]
                rms_avg = rows['rms'].mean() 
                rms_folded_avg = rows['rms_folded'].mean()
                azim = azim_pair[0] if azim_pair[0] <= 91 else azim_pair[0]-360
                df = df.append({
                        "Folded RMS (Degrees)": rms_folded_avg,
                        "DOA Algorithm": algo_name,
                        "Azimuth (Degrees)": azim,
                        "RMS (Degrees)": rms_avg,
                        "Architecture Index": arch_index
                          }, ignore_index=True)
    return df

    
def make_azim_vs_folded_rms_plot_all_algorithms(df_rms):
    sns.lineplot(x="Azimuth (Degrees)", y="Folded RMS (Degrees)", ci=68,err_style='bars',hue="DOA Algorithm", data=df_rms)

def get_folded_label_idx_5deg(azim_label):
    azim_degrees = azim_label*5
    if azim_degrees > 270 or azim_degrees < 90:
        reversal_point = round(math.degrees(math.acos(-math.cos(math.radians((azim_degrees+azim_degrees//180*180)%360))))+azim_degrees//180*180)
    else:
        reversal_point =  azim_degrees
    #folded = fold_locations(averages)
    reversal_idx = int((reversal_point - 90)/5)
    return reversal_idx

def CIPIC_azim_folding(azim):
    folded_idx= get_folded_label_idx_5deg(azim)
    folded_degrees = 90-folded_idx*5
    return folded_degrees

convert_from_numpy = lambda x: x[0] if len(x.tolist()) ==1 else x.tolist()

def get_algorithm_error(pd_data):
    #load data and set algorithms, conacat arrays, use this function
    pd_data = pd_data.convert_objects(convert_numeric=True)
    error = lambda x: min(72-(abs(x['azim']-x['predicted_folded'])),abs(x['azim']-x['predicted_folded']))
    if pd_data['azim'].dtype != 'int64' and pd_data['azim'].dtype != 'float64':
        pd_data['azim'] = pd_data['azim'].apply(convert_from_numpy)
        pd_data['elev'] = pd_data['elev'].apply(convert_from_numpy)
    if 'DOA Algorithm' not in pd_data.columns:
        pd_data['squared_error'] = (5*pd_data.apply(error,axis=1))**2
        pd_data['azim_folded'] = pd_data['azim'].apply(CIPIC_azim_folding)
        pd_data.rename(columns={'squared_error':'RMS Error (Degrees)',
                        'azim_folded':'Azimuth (Degrees)','algorithm':'DOA Algorithm'},
                       inplace=True)
    plt.clf()
    plt.figure()
    sns.lineplot(x='Azimuth (Degrees)',y='RMS Error (Degrees)',
                 hue='DOA Algorithm',estimator=lambda x:math.sqrt(np.mean(x)), 
                 ci=68,err_style='bars',legend=False,data=pd_data)
    plt.xticks([-75,-50,-25,0,25,50,75])
    plt.tight_layout()
    return pd_data

def get_algorithm_front_back_error(pd_data):
    is_mistake = lambda row: True if ((17 < row['azim'] < 54)
                                    != (17 < row['predicted'] < 54) and 
                                    (row['azim'] not in [17,54])) else False
    if 'DOA Algorithm' not in pd_data.columns:
        pd_data['front_back_error'] = pd_data.apply(is_mistake,axis=1)
        pd_data.rename(columns={'front_back_error':'Proportion of Front/Back Msitakes',
                        'azim_folded':'Azimuth (Degrees)','algorithm':'DOA Algorithm'},
                       inplace=True)
    plt.clf()
    plt.figure(figsize=(8,8))
    sns.barplot(x='DOA Algorithm',y='Proportion of Front/Back Msitakes',
                 estimator=lambda x:np.sum(x)*100.0/len(x), 
                 ci=67,data=pd_data)
    plt.yticks([10,20,30,40,50,60])
    plt.ylabel("Front/Back Mistakes (%)")
    plt.xticks(rotation=40)
    plt.tight_layout()

def get_bootstrap_dist(data_to_bootstrap,model_choices):
    rmse_list = []
    for model_idx in model_choices:
        model_mean = sum([data_to_bootstrap[i] for i in model_idx])/len(model_idx)
        rmse_list.append(model_mean)
    return rmse_list


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

def fig_2f():
    #Network Data
    fname_pkl = '/om/user/francl/baseline_algorithm_results_0PosMic/pd_kemar_offset26'
    pd_data = pd.read_pickle(fname_pkl)
    pd_dataframe_folded = make_dataframe("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/arch_number_*[3,4,5,6,7,8,9][0-9]_init_0/batch_conditional_stimRecords_binaural_recorded_testset_4078_main_kemar_0elev_speech_label_iter200000.npy",fold_data=True)
    pd_dataframe_folded_2_mic = make_dataframe("/om2/user/francl/new_task_archs/new_task_archs_2_mic_with_reverberation_background_noise_5dB40dBSNR_training/arch_number_*[3,4,5,6,7,8,9][0-9]_init_0/batch_conditional_stimRecords_binaural_recorded_testset_4078_main_2_mic_0elev_speech_label_iter200000.npy",fold_data=True)
    pd_dataframe_folded['algorithm'] = "KEMAR CNN (Ours)"
    pd_dataframe_folded_2_mic['algorithm'] = "2 Microphone Array CNN (Ours)"
    #Human Data import
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
    error_func = lambda x: (x["Actual Position (Degrees)"] -
                            x["Predicted Position (Degrees)"])**2
    pd_yost_distribution["error"] = pd_yost_distribution.apply(error_func,axis=1)
    pd_yost_distribution = (pd_yost_distribution.groupby(["Actual Position (Degrees)"])
                            .mean().pow(.5).reset_index()
                           .drop("Predicted Position (Degrees)",axis=1))
    pd_concat_data = pd.concat([pd_dataframe_folded,pd_dataframe_folded_2_mic,pd_data])
    plt.clf()
    get_algorithm_error(pd_concat_data)
    sns.lineplot(x='Actual Position (Degrees)',y='error',
                 data=pd_yost_distribution.astype('int32'),
                 palette=("black",),ci=None,legend=False)
    plt.yticks([0,20,40,60,80,100])
    #setting human line to black because seaborn isn't working correctly
    plt.gca().get_lines()[-1].set_color("black")
    plt.savefig(plots_folder+"/rms_vs_azim_by_algorithm_all_with_human.png",dpi=400)

def fig_2g():
    fname_pkl = '/om/user/francl/baseline_algorithm_results_0PosMic/pd_kemar_offset26'
    pd_data = pd.read_pickle(fname_pkl)
    pd_dataframe_folded = make_dataframe("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/arch_number_*[3,4,5,6,7,8,9][0-9]_init_0/batch_conditional_stimRecords_binaural_recorded_testset_4078_main_kemar_0elev_speech_label_iter200000.npy",fold_data=True)
    pd_dataframe_folded_2_mic = make_dataframe("/om2/user/francl/new_task_archs/new_task_archs_2_mic_with_reverberation_background_noise_5dB40dBSNR_training/arch_number_*[3,4,5,6,7,8,9][0-9]_init_0/batch_conditional_stimRecords_binaural_recorded_testset_4078_main_2_mic_0elev_speech_label_iter200000.npy",fold_data=True)
    pd_dataframe_folded['algorithm'] = "KEMAR CNN (Ours)"
    pd_dataframe_folded_2_mic['algorithm'] = "2 Microphone Array CNN (Ours)"
    pd_concat_data = pd.concat([pd_dataframe_folded,pd_dataframe_folded_2_mic,pd_data])
    get_algorithm_front_back_error(pd_concat_data)
    plt.savefig(plots_folder+"/front_back_by_algorithm_all.png",dpi=400)

    

def fig_8f():
    pd_dataframe_folded_normal = make_dataframe(("/om2/user/francl/grp-om2/gahlm/dataset_pipeline_test/"
                            "arch_number_*[0,1,3,4,5,6,7,8,9][0-9]_init_0/batch_conditional_stimRecords_binaural_recorded"
                            "_testset_4078_main_kemar_0elev_speech_label_iter100000.npy"),fold_data=True)
    pd_dataframe_folded_anechoic = make_dataframe(("/om2/user/francl/new_task_archs/new_task_archs_anechoic_training/"
                            "arch_number_*[0,1,3,4,5,6,7,8,9][0-9]_init_0/batch_conditional_stimRecords_binaural_recorded"
                            "_testset_4078_main_kemar_0elev_speech_label_iter100000.npy"),fold_data=True)
    pd_dataframe_folded_noiseless = make_dataframe(("/om2/user/francl/new_task_archs/"
                            "new_task_archs_no_background_noise_80dBSNR_training/"
                            "arch_number_*[0,1,3,4,5,6,7,8,9][0-9]_init_0/batch_conditional_stimRecords_binaural_recorded"
                            "_testset_4078_main_kemar_0elev_speech_label_iter100000.npy"),fold_data=True)
    pd_dataframe_folded_unnatural_sounds = make_dataframe(("/om2/user/francl/new_task_archs/"
                            "new_task_archs_logspaced_0.5octv_fluctuating_noise_training/"
                            "arch_number_*[0,1,3,4,5,6,7,8,9][0-9]_init_0/batch_conditional_stimRecords_binaural_recorded"
                            "_testset_4078_main_kemar_0elev_speech_label_iter150000.npy"),fold_data=True)
    pd_dataframe_folded_normal['algorithm'] = "Normal"
    pd_dataframe_folded_anechoic['algorithm'] = "Anechoic"
    pd_dataframe_folded_noiseless['algorithm'] = "Noiseless"
    pd_dataframe_folded_unnatural_sounds['algorithm'] = "Unnatural Sounds"
    pd_dataframe_folded_normal = get_algorithm_error(pd_dataframe_folded_normal)
    pd_dataframe_folded_anechoic = get_algorithm_error(pd_dataframe_folded_anechoic)
    pd_dataframe_folded_noiseless = get_algorithm_error(pd_dataframe_folded_noiseless)
    pd_dataframe_folded_unnatural_sounds = get_algorithm_error(pd_dataframe_folded_unnatural_sounds)
    pd_all_conditions = pd.concat([pd_dataframe_folded_normal,pd_dataframe_folded_anechoic,
                                   pd_dataframe_folded_noiseless,pd_dataframe_folded_unnatural_sounds])
    pd_data = pd_all_conditions.groupby(["DOA Algorithm","arch_index"]).mean().reset_index()
    bar_order = ["Normal","Anechoic","Noiseless","Unnatural Sounds"]
    plt.clf()
    new_palette = ["#97B4DE","#785EF0","#DC267F","#FE6100"]
    sns.barplot(x="DOA Algorithm",y="RMS Error (Degrees)",
                 estimator=lambda x:math.sqrt(np.mean(x)), 
                 ci=68,order=bar_order,
                 palette=sns.color_palette(new_palette),data=pd_data)
    plt.xticks(rotation=15)
    plt.ylim(0,35)
    plt.yticks([0,5,10,15,20,25,30,35])
    plt.tight_layout()
    plt.savefig(plots_folder+"/real_world_error_by_training_condition.png",dpi=400)




#        #Plot spatial maps
#        base = 1.
#        height = 10.
#        true_col = [0, 0, 0]
#        phi_plt = doa.grid.azimuth
#
#        fig, ax_list = plt.subplots(len(algo_names), 1, subplot_kw=dict(polar=True))
#        fig.set_size_inches(8, 12)
#        fig.tight_layout()
#        plt.subplots_adjust(hspace = 0.5,top=0.95)
#        for ax,algo_name in zip(ax_list,algo_names):
    #            # plot
    #            c_phi_plt = np.r_[phi_plt, phi_plt[0]]
    #            c_dirty_img = np.r_[spatial_resp[algo_name], spatial_resp[algo_name][0]]
    #            ax.plot(c_phi_plt, base + height * c_dirty_img, linewidth=3,
    #                    alpha=0.55, linestyle='-',
    #                    label="spatial spectrum")
    #            ax.set_title(algo_name,pad=20)
    #            
    #            #plt.legend()
    #            #handles, labels = ax.get_legend_handles_labels()
    #            #ax.legend(handles, labels, framealpha=0.5,
    #            #          scatterpoints=1, loc='center right', fontsize=16,
    #            #          ncol=1, bbox_to_anchor=(1.6, 0.5),
    #            #          handletextpad=.2, columnspacing=1.7, labelspacing=0.1)
    #
    #            ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
    #            ax.xaxis.set_label_coords(0.5, -0.11)
    #            ax.set_yticks(np.linspace(0, 1, 2))
    #            ax.xaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle=':')
    #            ax.yaxis.grid(b=True, color=[0.3, 0.3, 0.3], linestyle='--')
    #            ax.set_ylim([0, 1.05 * (base + height)]);
    #
    #
    #        plt.savefig(plots_folder+'/'+'spatial_map_{}_azim_{}.png'.format(azim,source_name))

