import os
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'
from math import sqrt,ceil
import numpy as np 
import  tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.contrib import ffmpeg
import sys
import glob
import time
import json
import pdb
from NetBuilder_valid_pad import NetBuilder
from layer_generator import generate
from tfrecords_iterator_get_power import build_tfrecords_iterator
from google.protobuf.json_format import MessageToJson
from parse_nested_dictionary import parse_nested_dictionary
import collections
import scipy.signal as signallib
from pycochleagram import erbfilter as erb
from pycochleagram import subband as sb
from scipy.io.wavfile import write
from get_tensor_metrics import *


import memory_saving_gradients
from tensorflow.python.ops import gradients
#import mem_util

# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='memory', **kwargs)
gradients.__dict__["gradients"] = memory_saving_gradients.gradients_speed

def tf_record_CNN_spherical(all_positions_bkgd,background_textures,testing,zero_padded,stacked_channel,num_epochs,train_path_pattern,bkgd_train_path_pattern,newpath,SNR_max=40,SNR_min=5):

    bkgd_training_paths = glob.glob(bkgd_train_path_pattern)
    training_paths = glob.glob(train_path_pattern)

    ###Do not change parameters below unless altering network###
    narrowband_noise = False
    manually_added = False

    BKGD_SIZE = [78,48000]
    STIM_SIZE = [78,89999]
    TONE_SIZE = [78,59099]
    ITD_TONE_SIZE = [78,39690]
    if zero_padded:
        STIM_SIZE = [78,48000]

    if stacked_channel:
        STIM_SIZE = [39,48000, 2]
        BKGD_SIZE = [39,48000,2]
    n_classes_localization=504
    n_classes_recognition=780
    localization_bin_resolution=5

    #Optimization Params
    batch_size=16
    learning_rate = 1e-3
    loss_exponent = 12
    loss_scale = 2**loss_exponent
    bn_training_state = True
    dropout_training_state = True
    training_state = True 
    #Queue parameters
    dequeue_min = 8
    dequeue_min_main = 10
    #num_epochs = None
    #Change for network precision,must match input data type
    filter_dtype = tf.float32
    padding='VALID'

    #Downsampling Params
    sr=48000
    cochleagram_sr=8000
    post_rectify=True

    #Display interval training statistics
    display_step = 1000

    if testing:
        bn_training_state = False
        dropout_training_state = False
        training_state= False
        num_epochs = 1
        #Using these values because 5/40 are the standard training SNRs
        if not (SNR_min > 30 or SNR_max > 40):
            SNR_max = 35.0
            SNR_min = 30.0
        print("Testing SNR(dB): Max: "+str(SNR_max)+"Min: "+str(SNR_min))


    #mean_subbands = np.load("mean_subband_51400.npy")/51400
    #tf_mean_subbands = tf.constant(mean_subbands,dtype=filter_dtype)
    def check_speed():
        for i in range(30):
            sess.run(subbands_batch)
        start_time = time.time()
        for _ in range(30):
            time.sleep(0.5)
            print(time.time() - start_time)
            print("Len:",sess.run('example_queue/random_shuffle_queue_Size:0'))

    with tf.device("/cpu:0"):
        ###READING QUEUE MACHINERY###

        def add_labels(paths):
            return [(stim,stim.split('_')[-1].split('a')[0]) for stim in
                    paths]

        def rms(wav):
            square = tf.square(wav) 
            mean_val = tf.reduce_mean(square)
            return tf.sqrt(mean_val)



        def combine_signal_and_noise_stacked_channel(signals,backgrounds,delay,
                                                     sr,cochleagram_sr,post_rectify):
            tensor_dict_fg = {}
            tensor_dict_bkgd = {}
            tensor_dict = {}
            snr = tf.random_uniform([],minval=SNR_min,maxval=SNR_max,name="snr_gen")
            for path1 in backgrounds:
                if path1 == 'train/image':
                    background = backgrounds['train/image']
                else:
                    tensor_dict_bkgd[path1] = backgrounds[path1]
            for path in signals:
                if path == 'train/image':
                    signal = signals['train/image']
                    sig_len= signal.shape[1] - delay
                    sig = tf.slice(signal,[0,0,0],[39,sig_len,2])
                    max_val = tf.reduce_max(sig)
                    sig_rms = rms(tf.reduce_sum(sig,[0,2]))
                    sig = tf.div(sig,sig_rms)
                    #sig = tf.Print(sig, [tf.reduce_max(sig)],message="\nMax SIG:")
                    sf = tf.pow(tf.constant(10,dtype=tf.float32),
                                tf.div(snr,tf.constant(20,dtype=tf.float32)))
                    bak_rms = rms(tf.reduce_sum(background,[0,2]))
                    #bak_rms = tf.Print(bak_rms, [tf.reduce_max(bak_rms)],message="\nNoise RMS:")
                    sig_rms = rms(tf.reduce_sum(sig,[0,2]))
                    scaling_factor = tf.div(tf.div(sig_rms,bak_rms),sf)
                    #scaling_factor = tf.Print(scaling_factor, [scaling_factor],message="\nScaling Factor:")
                    noise = tf.scalar_mul(scaling_factor,background)
                    #noise = tf.Print(noise, [tf.reduce_max(noise)],message="\nMax Noise:")
                    front = tf.slice(noise,[0,0,0],[39,delay,2])
                    middle = tf.slice(noise,[0,delay,0],[39,sig_len,2])
                    end = tf.slice(noise,[0,(delay+int(sig_len)),0],[39,-1,2])
                    middle_added = tf.add(middle,sig)
                    new_sig = tf.concat([front,middle_added,end],1)
                    #new_sig = sig
                    rescale_factor = tf.div(max_val,tf.reduce_max(new_sig))
                    #rescale_factor = tf.Print(rescale_factor, [rescale_factor],message="\nRescaling Factor:")
                    new_sig = tf.scalar_mul(rescale_factor,new_sig)
                    new_sig_rectified = tf.nn.relu(new_sig)
                    new_sig_reshaped = tf.reshape(new_sig_rectified,[39,48000,2])
                    #new_sig_reshaped = tf.reshape(new_sig,[72,30000,1])
                    #return (signal, background,noise,new_sig_reshaped)
                    tensor_dict_fg[path] = new_sig_reshaped
                else:
                    tensor_dict_fg[path] = signals[path]
            tensor_dict[0] = tensor_dict_fg
            tensor_dict[1] = tensor_dict_bkgd
            return tensor_dict

        def combine_signal_and_noise(signals,backgrounds,delay,
                                     sr,cochleagram_sr,post_rectify):
            tensor_dict_fg = {}
            tensor_dict_bkgd = {}
            tensor_dict = {}
            snr = tf.random_uniform([],minval=SNR_min,maxval=SNR_max,name="snr_gen")
            for path1 in backgrounds:
                if path1 == 'train/image':
                    background = backgrounds['train/image']
                else:
                    tensor_dict_bkgd[path1] = backgrounds[path1]
            for path in signals:
                if path == 'train/image':
                    signal = signals['train/image']
                    sig_len= signal.shape[1] - delay
                    sig = tf.slice(signal,[0,0],[78,sig_len])
                    max_val = tf.reduce_max(sig)
                    sig_rms = rms(tf.reduce_sum(sig,0))
                    sig = tf.div(sig,sig_rms)
                    #sig = tf.Print(sig, [tf.reduce_max(sig)],message="\nMax SIG:")
                    sf = tf.pow(tf.constant(10,dtype=tf.float32),tf.div(snr,tf.constant(20,dtype=tf.float32)))
                    bak_rms = rms(tf.reduce_sum(background,0))
                    #bak_rms = tf.Print(bak_rms, [tf.reduce_max(bak_rms)],message="\nNoise RMS:")
                    sig_rms = rms(tf.reduce_sum(sig,0))
                    scaling_factor = tf.div(tf.div(sig_rms,bak_rms),sf)
                    #scaling_factor = tf.Print(scaling_factor, [scaling_factor],message="\nScaling Factor:")
                    noise = tf.scalar_mul(scaling_factor,background)
                    #noise = tf.Print(noise, [tf.reduce_max(noise)],message="\nMax Noise:")
                    front = tf.slice(noise,[0,0],[78,delay])
                    middle = tf.slice(noise,[0,delay],[78,sig_len])
                    end = tf.slice(noise,[0,(delay+int(sig_len))],[78,-1])
                    middle_added = tf.add(middle,sig)
                    new_sig = tf.concat([front,middle_added,end],1)
                    #new_sig = sig
                    rescale_factor = tf.div(max_val,tf.reduce_max(new_sig))
                    #rescale_factor = tf.Print(rescale_factor, [rescale_factor],message="\nRescaling Factor:")
                    new_sig = tf.scalar_mul(rescale_factor,new_sig)
                    new_sig_rectified = tf.nn.relu(new_sig)
                    new_sig_reshaped = tf.reshape(new_sig_rectified,[72,48000,1])
                    #new_sig_reshaped = tf.reshape(new_sig,[72,30000,1])
                    #return (signal, background,noise,new_sig_reshaped)
                    tensor_dict_fg[path] = new_sig_reshaped
                else:
                    tensor_dict_fg[path] = signals[path]
            tensor_dict[0] = tensor_dict_fg
            tensor_dict[1] = tensor_dict_bkgd
            return tensor_dict

        #Best to read https://www.tensorflow.org/api_guides/python/reading_data#Reading_from_files
        ###READING QUEUE MACHINERY###
        #Best to read https://www.tensorflow.org/api_guides/python/reading_data#Reading_from_files


        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        is_bkgd = False       
        first = training_paths[0]
        for example in tf.python_io.tf_record_iterator(first,options=options):
            result = tf.train.Example.FromString(example)
            break

        jsonMessage = MessageToJson(tf.train.Example.FromString(example))
        jsdict = json.loads(jsonMessage)
        feature = parse_nested_dictionary(jsdict,is_bkgd)

        dataset = build_tfrecords_iterator(num_epochs, train_path_pattern, is_bkgd, feature, narrowband_noise, manually_added, STIM_SIZE, localization_bin_resolution,stacked_channel)





        ###READING QUEUE MACHINERY###


        # Create a list of filenames and pass it to a queue
        bkgd_filename_queue = tf.train.string_input_producer(bkgd_training_paths,
                                                        shuffle=True,
                                                        capacity=len(bkgd_training_paths))
        # Define a reader and read the next record
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        bkgd_reader = tf.TFRecordReader(options=options)
        _, bkgd_serialized_example = bkgd_reader.read(bkgd_filename_queue)


        is_bkgd = True
        bkgd_first = bkgd_training_paths[0]
        for bkgd_example in tf.python_io.tf_record_iterator(bkgd_first,options=options):
            bkgd_result = tf.train.Example.FromString(bkgd_example)
            break

        bkgd_jsonMessage = MessageToJson(tf.train.Example.FromString(bkgd_example))
        bkgd_jsdict = json.loads(bkgd_jsonMessage)
        bkgd_feature = parse_nested_dictionary(bkgd_jsdict,is_bkgd)


        dataset_bkgd = build_tfrecords_iterator(num_epochs, bkgd_train_path_pattern, is_bkgd, bkgd_feature, narrowband_noise, manually_added, BKGD_SIZE, localization_bin_resolution, stacked_channel)

        dataset_iter = dataset_bkgd.make_initializable_iterator()
        dataset_dict = dataset_iter.get_next()
        source_cochleagram = dataset_dict['train/image']
        mean_cochleagram, mean_cochleagram_update = record_tensor_mean(source_cochleagram)

        #new_dataset = tf.data.Dataset.zip((dataset, dataset_bkgd))



        ##SNR = tf.random_uniform([],minval=SNR_min,maxval=SNR_max,name="snr_gen")
        #
        #
        #if stacked_channel:
        #    new_dataset = new_dataset.map(lambda x,y: combine_signal_and_noise_stacked_channel(x,y,0,48000,8000,post_rectify=True))
        #else:
        #    new_dataset = new_dataset.map(lambda x,y: combine_signal_and_noise(x,y,0,48000,8000,post_rectify=True))
        #batch_sizes = tf.constant(16,dtype=tf.int64)
        #new_dataset = new_dataset.shuffle(buffer_size=200).batch(batch_size=batch_sizes,drop_remainder=True)
        ##combined_iter = new_dataset.make_one_shot_iterator()
        #combined_iter = new_dataset.make_initializable_iterator()
        #combined_iter_dict = collections.OrderedDict()
        #combined_iter_dict = combined_iter.get_next()

        #if background_textures:
        #    bkgd_metadata = [combined_iter_dict[1]['train/azim'],combined_iter_dict[1]['train/elev']]


    ###END READING QUEUE MACHINERY###


    def make_downsample_filt_tensor(SR=16000, ENV_SR=200, WINDOW_SIZE=1001, beta=5.0, pycoch_downsamp=False):
        """
        Make the sinc filter that will be used to downsample the cochleagram
        Parameters
        ----------
        SR : int
            raw sampling rate of the audio signal
        ENV_SR : int
            end sampling rate of the envelopes
        WINDOW_SIZE : int
            the size of the downsampling window (should be large enough to go to zero on the edges).
        beta : float
            kaiser window shape parameter
        pycoch_downsamp : Boolean
            if true, uses a slightly different downsampling function
        Returns
        -------
        downsample_filt_tensor : tensorflow tensor, tf.float32
            a tensor of shape [0 WINDOW_SIZE 0 0] the sinc windows with a kaiser lowpass filter that is applied while downsampling the cochleagram
        """
        DOWNSAMPLE = SR/ENV_SR
        if not pycoch_downsamp: 
            downsample_filter_times = np.arange(-WINDOW_SIZE/2,int(WINDOW_SIZE/2))
            downsample_filter_response_orig = np.sinc(downsample_filter_times/DOWNSAMPLE)/DOWNSAMPLE
            downsample_filter_window = signallib.kaiser(WINDOW_SIZE, beta)
            downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
        else: 
            max_rate = DOWNSAMPLE
            f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
            half_len = 10 * max_rate  # reasonable cutoff for our sinc-like function
            if max_rate!=1:    
                downsample_filter_response = signallib.firwin(2 * half_len + 1, f_c, window=('kaiser', beta))
            else:  # just in case we aren't downsampling -- I think this should work? 
                downsample_filter_response = zeros(2 * half_len + 1)
                downsample_filter_response[half_len + 1] = 1
                
            # Zero-pad our filter to put the output samples at the center
            # n_pre_pad = int((DOWNSAMPLE - half_len % DOWNSAMPLE))
            # n_post_pad = 0
            # n_pre_remove = (half_len + n_pre_pad) // DOWNSAMPLE
            # We should rarely need to do this given our filter lengths...
            # while _output_len(len(h) + n_pre_pad + n_post_pad, x.shape[axis],
            #                  up, down) < n_out + n_pre_remove:
            #     n_post_pad += 1
            # downsample_filter_response = np.concatenate((np.zeros(n_pre_pad), downsample_filter_response, np.zeros(n_post_pad)))
                
        downsample_filt_tensor = tf.constant(downsample_filter_response, tf.float32)
        downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 0)
        downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 2)
        downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 3)

        return downsample_filt_tensor 

    def downsample(signal,current_rate,new_rate,window_size,
                   beta,post_rectify=True):
        downsample = current_rate/new_rate
        message = ("The current downsample rate {} is "
                   "not an integer. Only integer ratios "
                   "between current and new sampling rates "
                   "are supported".format(downsample))

        assert (current_rate%new_rate == 0), message
        message = ("New rate must be less than old rate for this "
                    "implementation to work!")
        assert (new_rate < current_rate), message
        # make the downsample tensor
        downsample_filter_tensor = make_downsample_filt_tensor(current_rate, new_rate,
                                                                window_size, pycoch_downsamp=False)
        downsampled_signal  = tf.nn.conv2d(signal, downsample_filter_tensor,
                                           strides=[1, 1, downsample, 1], padding='SAME',
                                           name='conv2d_cochleagram_raw')
        if post_rectify:
            downsampled_signal = tf.nn.relu(downsampled_signal)
        
        return downsampled_signal


    
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Launch the graph
    #with tf.Session() as sess:
    #run_metadata = tf.RunMetadata()
    config = tf.ConfigProto(allow_soft_placement=True,
                            inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
    sess = tf.Session(config=config)
    sess.run(init_op)

#     ##This code allows for tracing ops acorss GPUs, you often have to run it twice
#     ##to get sensible traces
# 
#     #sess.run(optimizer,options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
#     #                     run_metadata=run_metadata)
#     #from tensorflow.python.client import timeline
#     #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
#     #trace_file.close()

##Used to write out stimuli examples
#
#    low_lim=30
#    hi_lim=20000
#    sr=48000
#    sample_factor=1
#    scale = 0.1
#    i=0
#    pad_factor = None
#    #invert subbands
#    n = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)
#    sess.run(combined_iter.initializer)
#    subbands_test,az_label,elev_label = sess.run([combined_iter_dict[0]['train/image'],combined_iter_dict[0]['train/azim'],combined_iter_dict[0]['train/elev']])
#
#    filts, hz_cutoffs, freqs=erb.make_erb_cos_filters_nx(subbands_test.shape[2],sr, n,low_lim,hi_lim, sample_factor,pad_factor=pad_factor,full_filter=True)
#
#    filts_no_edges = filts[1:-1]
#    for batch_iter in range(3):
#        for stim_iter in range(16):
#            subbands_l=subbands_test[stim_iter,:,:,0]
#            subbands_r=subbands_test[stim_iter,:,:,1]
#            wavs = np.zeros([subbands_test.shape[2],2])
#            wavs[:,0] = sb.collapse_subbands(subbands_l,filts_no_edges).astype(np.float32)
#            wavs[:,1] = sb.collapse_subbands(subbands_r,filts_no_edges).astype(np.float32)
#            max_val = wavs.max()
#            rescaled_wav = wavs/max_val*scale
#            name = "stim_{}_{}az_{}elev.wav".format(stim_iter+batch_iter*16,int(az_label[stim_iter])*5,int(elev_label[stim_iter])*5)
#            name_with_path = newpath+'/'+name
#            write(name_with_path,sr,rescaled_wav)
#        pdb.set_trace()
#        subbands_test,az_label,elev_label = sess.run([combined_iter_dict[0]['train/image'],combined_iter_dict[0]['train/azim'],combined_iter_dict[0]['train/elev']])

    sess.run(dataset_iter.initializer)
    stimuli_name = train_path_pattern.split("/")[-2]
    step = 0
    try:
        while True:
            sess.run(mean_cochleagram_update)
            step+=1
            if step % display_step ==0:
                print("Iter "+str(step))
    except tf.errors.ResourceExhaustedError:
        print("Out of memory error")
        error= "Out of memory error"
        with open(newpath+'/test_error_{}.json'.format(stim),'w') as f:
            json.dump(arch_ID,f)
            json.dump(error,f)
    except tf.errors.OutOfRangeError:
        print("Out of Range Error. Calculation Finished")
        
    finally:
        np_dataset_mean = sess.run(mean_cochleagram)
        np.save(newpath+'/dataset_mean_{}.npy'.format(stimuli_name),np_dataset_mean)
 
     #acc= sess.run(test_acc)
     #print("Test Accuracy= "+"{:.5f}".format(acc))
     #customs = sess.run(custom_test_acc)
     #correct_pred = sess.run(custom_correct_pred)
     #with open('custom_out2.json', 'w') as f:
     #    json.dump([test_data_img,correct_pred.tolist()],f)
     #print("ACC for special cases:")
     #print(customs) 
     #first_layer = sess.run(weights['wc1'])
     #activation1, activation2 = sess.run([conv1,conv3])
     #with open('activations.json','w') as f:
     #    json.dump([activation1.tolist(),activation2.tolist()],f)
     #tf.get_variable_scope().reuse_variables()
     #first_layer = [var for var in tf.global_variables() if var.op.name=="wc1"][0]
     #second_layer = [var for var in tf.global_variables() if var.op.name=="wc2"][0]
     #weights_image = put_kernels_on_grid(first_layer)
     #weights_image2 = put_kernels_on_grid(second_layer)
     #np_weights1, np_weights2 = sess.run([weights_image,weights_image2])
     #with open('conv1weights.json','w') as f:
     #    json.dump([np_weights1.tolist(),np_weights2.tolist()],f)
     #
    sess.close()
    tf.reset_default_graph()

