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
from tfrecords_iterator import build_tfrecords_iterator
from google.protobuf.json_format import MessageToJson
from parse_nested_dictionary import parse_nested_dictionary
import collections
import scipy.signal as signallib

import memory_saving_gradients
from tensorflow.python.ops import gradients
#import mem_util

# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='memory', **kwargs)
gradients.__dict__["gradients"] = memory_saving_gradients.gradients_speed

def tf_record_CNN_spherical(tone_version,itd_tones,ild_tones,manually_added,freq_label,sam_tones,transposed_tones,precedence_effect,narrowband_noise,all_positions_bkgd,background_textures,testing,branched,zero_padded,stacked_channel,model_version,num_epochs,train_path_pattern,bkgd_train_path_pattern,arch_ID,config_array,files,num_files,newpath):

    bkgd_training_paths = glob.glob(bkgd_train_path_pattern)
    training_paths = glob.glob(train_path_pattern)

    ###Do not change parameters below unless altering network###

    BKGD_SIZE = [78,48000]
    STIM_SIZE = [78,89999]
    TONE_SIZE = [78,59099]
    ITD_TONE_SIZE = [78,39690]
    if zero_padded:
        STIM_SIZE = [78,48000]

    if stacked_channel:
        STIM_SIZE = [39,48000, 2]
        BKGD_SIZE = [39,48000,2]
    SNR_max = 40
    SNR_min = 5
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
    display_step = 25

    if itd_tones:
        TONE_SIZE = ITD_TONE_SIZE

    if ild_tones:
        itd_tones = True

    if testing:
        bn_training_state = False
        dropout_training_state = False
        training_state= False
        num_epochs = 1
        SNR_max = 35.0
        SNR_min = 30.0


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

        new_dataset = tf.data.Dataset.zip((dataset, dataset_bkgd))



        #SNR = tf.random_uniform([],minval=SNR_min,maxval=SNR_max,name="snr_gen")
        
        
        if stacked_channel:
            new_dataset = new_dataset.map(lambda x,y: combine_signal_and_noise_stacked_channel(x,y,0,48000,8000,post_rectify=True))
        else:
            new_dataset = new_dataset.map(lambda x,y: combine_signal_and_noise(x,y,0,48000,8000,post_rectify=True))
        batch_sizes = tf.constant(16,dtype=tf.int64)
        new_dataset = new_dataset.shuffle(buffer_size=200).batch(batch_size=batch_sizes,drop_remainder=True)
        #combined_iter = new_dataset.make_one_shot_iterator()
        combined_iter = new_dataset.make_initializable_iterator()
        combined_iter_dict = collections.OrderedDict()
        combined_iter_dict = combined_iter.get_next()

        if background_textures:
            bkgd_metadata = [combined_iter_dict[1]['train/azim'],combined_iter_dict[1]['train/elev']]


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


    def put_kernels_on_grid (kernel, pad = 1):

      '''Visualize conv. filters as an image (mostly for the 1st layer).
      Arranges filters into a grid, with some paddings between adjacent filters.
      Args:
        kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
        pad:               number of black pixels around each filter (between them)
      Return:
        Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
      '''
      # get shape of the grid. NumKernels == grid_Y * grid_X
      def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
          if n % i == 0:
            if i == 1: print('Who would enter a prime number of filters')
            return (i, int(n / i))
      (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
      print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

      x_min = tf.reduce_min(kernel)
      x_max = tf.reduce_max(kernel)
      kernel = (kernel - x_min) / (x_max - x_min)

      # pad X and Y
      x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

      # X and Y dimensions, w.r.t. padding
      Y = kernel.get_shape()[0] + 2 * pad
      X = kernel.get_shape()[1] + 2 * pad
      x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

      # X and Y dimensions, w.r.t. padding
      Y = kernel.get_shape()[0] + 2 * pad
      X = kernel.get_shape()[1] + 2 * pad

      channels = kernel.get_shape()[2]

      # put NumKernels to the 1st dimension
      x = tf.transpose(x, (3, 0, 1, 2))
      # organize grid on Y axis
      x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

      # switch X and Y axes
      x = tf.transpose(x, (0, 2, 1, 3))
      # organize grid on X axis
      x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

      # back to normal order (not combining with the next step for clarity)
      x = tf.transpose(x, (2, 1, 3, 0))

      # to tf.image_summary order [batch_size, height, width, channels],
      #   where in this case batch_size == 1
      x = tf.transpose(x, (3, 0, 1, 2))

      # scaling to [0, 255] is not necessary for tensorboard
      return x


    #Many lines are commented out to allow for quick architecture changes
    #TODO:This should be abstracted to arcitectures are defined by some sort of
    #config dictionary or file

    def gradients_with_loss_scaling(loss, loss_scale):
        """Gradient calculation with loss scaling to improve numerical stability
        when training with float16.
        """

        grads = [(grad[0] / loss_scale,grad[1]) for grad in
                 tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=1e-4).
                 compute_gradients(loss * loss_scale,colocate_gradients_with_ops=True)]
        return grads

    def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                        regularizer=None,trainable=True,*args, **kwargs):
        storage_dtype = tf.float32 if trainable else dtype
        variable = getter(name, shape, dtype=storage_dtype,initializer=initializer,
                          regularizer=regularizer,trainable=trainable,*args, **kwargs)
        if trainable and dtype != tf.float32:
            variable = tf.cast(variable, dtype)
        return variable

    strides =1
    time_stride =1
    freq_stride=2
    time_pool = 4
    freq_pool =1
    k=2
    k_wide =8
    
    
#    config_array=[[["/gpu:0"],['conv',[2,50,32],[2,1]],['relu'],['pool',[1,4]]],[["/gpu:1"],['conv',[4,20,64],[1,1]],['bn'],['relu'],['pool',[1,4]],['conv',[8,8,128],[1,1]],['bn'],['relu'],['pool',[1,4]],['conv',[8,8,256],[1,1]],['bn'],['relu'],['pool',[1,8]],['fc',512],['fc_bn'],['fc_relu'],['dropout'],['out',]]]



    #[L_channel,R_channel] = tf.unstack(subbands_batch,axis=3)
    [L_channel,R_channel] = tf.unstack(combined_iter_dict[0]['train/image'],axis=3)
    concat_for_downsample = tf.concat([L_channel,R_channel],axis=0)
    reshaped_for_downsample = tf.expand_dims(concat_for_downsample,axis=3)
    
    #hard coding filter shape based on previous experimentation
    new_sig_downsampled = downsample(reshaped_for_downsample,sr,cochleagram_sr,
                                     window_size=4097,beta=10.06,
                                     post_rectify=post_rectify)
    downsampled_squeezed = tf.squeeze(new_sig_downsampled)
    [L_channel_downsampled, R_channel_downsampled] = tf.split(downsampled_squeezed,
                                                              num_or_size_splits=2,
                                                              axis=0)
    downsampled_reshaped = tf.stack([L_channel_downsampled, R_channel_downsampled],axis=3)
    new_sig_nonlin = tf.pow(downsampled_reshaped,0.3)
    # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='fp32_storage'))
    # print(subbands_batch)
    
    ####TMEPORARY OVERRIDE####
    
    #branched = False
    net=NetBuilder()
    if branched:
        out,out2=net.build(config_array,new_sig_nonlin,training_state,dropout_training_state,filter_dtype,padding,n_classes_localization,n_classes_recognition,branched)
    else:
        out=net.build(config_array,new_sig_nonlin,training_state,dropout_training_state,filter_dtype,padding,n_classes_localization,n_classes_recognition,branched)
    
    
    combined_dict = collections.OrderedDict()
    combined_dict_fg = collections.OrderedDict()
    combined_dict_bkgd = collections.OrderedDict()
    for k,v in combined_iter_dict[0].items():
        if k != 'train/image' and k != 'train/image_height' and k != 'train/image_width': 
            combined_dict_fg[k] = combined_iter_dict[0][k]
    for k,v in combined_iter_dict[1].items():
        if k != 'train/image' and k != 'train/image_height' and k != 'train/image_width': 
            combined_dict_bkgd[k] = combined_iter_dict[1][k]
    combined_dict[0] = combined_dict_fg
    combined_dict[1] = combined_dict_bkgd


    ##Fully connected Layer 2
    #wd2 = tf.get_variable('wd2',[512,512],filter_dtype)
    #dense_bias2 = tf.get_variable('wb6',[512],filter_dtype)
    #fc2 = tf.add(tf.matmul(fc1_do, wd2), dense_bias2)
    #fc2 = tf.nn.relu(fc2)
    #fc2_do = tf.layers.dropout(fc2,training=dropout_training_state)


    # Construct model
    #fix labels dimension to be one less that logits dimension

    #Testing small subbatch
    if sam_tones or transposed_tones:
        labels_batch_cost_sphere = tf.squeeze(tf.zeros_like(combined_dict[0]['train/carrier_freq']))
    elif precedence_effect:
        labels_batch_cost_sphere = tf.squeeze(tf.zeros_like(combined_dict[0]['train/start_sample']))
    else:
        labels_batch_cost = tf.squeeze(combined_dict[0]['train/azim'])
        #labels_batch_cost = tf.squeeze(subbands_batch_labels,axis=[1,2])
        if not tone_version:
            labels_batch_sphere =tf.add(tf.scalar_mul(tf.constant(36,dtype=tf.int32),combined_dict[0]['train/elev']),combined_dict[0]['train/azim'])
        else:
            labels_batch_sphere = combined_dict[0]['train/azim'] 
        labels_batch_cost_sphere = tf.squeeze(labels_batch_sphere)
    
    # Define loss and optimizer
    # On r1.1 reduce mean doees not work(returns nans) with float16 vals
    
    if branched:
        cost1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=labels_batch_cost_sphere))
        cost2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out2,labels=combined_dict[0]['train/class_num']))
        cost = cost1 +cost2
    else:
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=labels_batch_cost_sphere))
    
    
    #cost = tf.Print(cost, [labels],message="\nLabel:",summarize=32)

    cond_dist = tf.nn.softmax(out)
    if branched:
        cond_dist2 = tf.nn.softmax(out2)
    
    #cost = tf.Print(cost, [tf.argmax(out, 1)],message="\nOut:",summarize=32)
    
#     init_op = tf.group(tf.global_variables_initializer(),
#                        tf.local_variables_initializer())
#     config = tf.ConfigProto(allow_soft_placement=True,
#                             inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
#     sess = tf.Session(config=config)
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#     print(sess.run(cost))
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        update_grads = tf.train.AdamOptimizer(learning_rate=learning_rate,epsilon=1e-4).minimize(cost)



    # Evaluate model
    correct_pred = tf.equal(tf.argmax(out, 1), tf.cast(labels_batch_cost_sphere,tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    top_k = tf.nn.top_k(out,5)
    
    if branched:
        correct_pred2 = tf.equal(tf.argmax(out2, 1), tf.cast(combined_dict[0]['train/class_num'],tf.int64))
        accuracy2 = tf.reduce_mean(tf.cast(correct_pred2, tf.float32))

        top_k2 = tf.nn.top_k(out2,5)
    #test_pred = conv_net(tf.cast(test_images,tf.float32),weights,biases)
    #correct_pred = tf.equal(tf.argmax(test_pred, 1), tf.cast(test_labels,tf.int64)) 
    #test_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    ##Check special cases(made by hand in testdata.json
    #custom_pred = conv_net(tf_test_data,weights,biases)
    #custom_correct_pred = tf.equal(tf.argmax(custom_pred, 1), tf.cast(tf_test_label,tf.int64)) 
    #custom_test_acc = tf.reduce_mean(tf.cast(custom_correct_pred, tf.float32))

    # Initializing the variables
    #
    # Check_op seems to take up a lot of space on the GPU
    check_op = tf.add_check_numerics_ops()
    
    
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Launch the graph
    #with tf.Session() as sess:
    #run_metadata = tf.RunMetadata()
    config = tf.ConfigProto(allow_soft_placement=True,
                            inter_op_parallelism_threads=0, intra_op_parallelism_threads=0)
    sess = tf.Session(config=config)
    sess.run(init_op)
    if branched:
        print("Class Labels:" + str(sess.run(combined_dict[0]['train/class_num'])))


#     ##This code allows for tracing ops acorss GPUs, you often have to run it twice
#     ##to get sensible traces
# 
#     #sess.run(optimizer,options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
#     #                     run_metadata=run_metadata)
#     #from tensorflow.python.client import timeline
#     #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
#     #trace_file.close()
    if not testing:
        sess.run(combined_iter.initializer)
        saver = tf.train.Saver(max_to_keep=None)
        learning_curve = []
        errors_count =0
        try:
             step = 1
             sess.graph.finalize()
             while True:
                 #sess.run([optimizer,check_op])
                 try:
                     if step ==1:
                         if not num_files == 1:
                             latest_addition = max(files, key=os.path.getctime)
                             latest_addition_name = latest_addition.split(".") [1]
                             saver.restore(sess,newpath+"/model."+latest_addition_name)
                             step=int(latest_addition_name.split("-") [1])
                         else:
                             sess.run(update_grads)
                     else:
                         sess.run(update_grads)
 #                    sess.run(update_grads)
                 except tf.errors.InvalidArgumentError as e:
                     print(e.message)
                     errors_count+=1
                     continue
                 if step % display_step == 0:
                     # Calculate batch loss and accuracy
                     loss, acc, az= sess.run([cost,accuracy,combined_dict[0]['train/azim']])
                     #print("Batch Labels: ",az)
                     print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                           "{:.6f}".format(loss) + ", Training Accuracy= " + \
                           "{:.5f}".format(acc))
                 if step%5000 ==0:
                     print("Checkpointing Model...")
                     saver.save(sess,newpath+'/model.ckpt',global_step=step,write_meta_graph=False)
                     learning_curve.append([int(step*batch_size),float(acc)])
                     print("Checkpoint Complete")
 
                 #Just for testing the model/call_model
                 if step == 200000:
                     print("Break!")
                     break
                 step += 1
        except tf.errors.OutOfRangeError:
             print("Out of Range Error. Optimization Finished")
        except tf.errors.DataLossError as e:
             print("Corrupted file found!!")
             pdb.set_trace()
        except tf.errors.ResourceExhaustedError as e:
             gpu=e.message
             print("Out of memory error")
             error= "Out of memory error"
             with open(newpath+'/train_error.json','w') as f:
                 json.dump(arch_ID,f)
                 json.dump(error,f)
                 json.dump(gpu,f)
             return False
        finally:
             print(errors_count)
             print("Training stopped.")
 
        with open(newpath+'/curve_no_resample_w_cutoff_vary_loc.json', 'w') as f:
             json.dump(learning_curve,f)
 

    if testing:
        ##Testing loop
        for stim in model_version:
            sess.run(combined_iter.initializer)
            print ("Starting model version: ", stim)
            batch_acc = []
            batch_acc2 = []
            batch_conditional = []
            batch_conditional2 = []
            saver = tf.train.Saver(max_to_keep=None)
            #saver.restore(sess,newpath+"/model.ckpt-"+str(model_version))
            saver.restore(sess,newpath+"/model.ckpt-"+str(stim))
            step = 0
            try:
                eval_vars = list(combined_dict[0].values())
                eval_keys = list(combined_dict[0].keys())
                while True:
                    pred, cd, e_vars = sess.run([correct_pred, cond_dist, eval_vars])
                    e_vars = np.squeeze(e_vars)
                    array_len = len(e_vars)
                    split = np.vsplit(e_vars,array_len)
                    batch_conditional += [(cond,var) for cond, var in zip(cd,e_vars.T)]
                    split.insert(0,pred)
                    batch_acc += np.dstack(split).tolist()[0]

                    if branched:
                        pred2, cd2, e_vars2 = sess.run(correct_pred2,cond_dist2,eval_vars)
                        e_vars2 = np.squeeze(e_vars2)
                        array_len2 = len(e_vars2)
                        split2 = np.vsplit(e_vars2,array_len2)
                        split2.insert(0,pred2)
                        batch_conditional2 += [(cond,var) for cond,var in zip(cd2,e_vars,T)]
                        batch_acc2 += np.dstack(split2).tolist()[0]

                    step+=1
                    if step % display_step ==0:
                        print("Iter "+str(step*batch_size))
                        #if not tone_version:
                        #    print("Current Accuracy:",sum(batch_acc)/len(batch_acc))
                    if step == 65000:
                        print ("Break!")
                        break
            except tf.errors.ResourceExhaustedError:
                print("Out of memory error")
                error= "Out of memory error"
                with open(newpath+'/test_error_{}.json'.format(stim),'w') as f:
                    json.dump(arch_ID,f)
                    json.dump(error,f)
            except tf.errors.OutOfRangeError:
                print("Out of Range Error. Optimization Finished")
                
            finally:
                if tone_version:
                    np.save(newpath+'/plot_array_test_{}.npy'.format(stim),batch_acc)
                    np.save(newpath+'/batch_conditional_test_{}.npy'.format(stim),batch_conditional)
                    acc_corr=[pred[0] for pred in batch_acc]
                    acc_accuracy=sum(acc_corr)/len(acc_corr)
                    if branched:
                        np.save(newpath+'/plot_array_test_{}_2.npy'.format(stim),batch_acc2)
                        np.save(newpath+'/batch_conditional_test_{}_2.npy'.format(stim),batch_conditional2)
                        acc_corr2=[pred2[0] for pred2 in batch_acc2]
                        acc_accuracy2=sum(acc_corr2)/len(acc_corr2)
                    with open(newpath+'/accuracies_itd_{}.json'.format(stim),'w') as f:
                        json.dump(acc_accuracy,f)
                        if branched:
                            json.dump(acc_accuracy2,f)
                elif (sam_tones or transposed_tones or 
                      precedence_effect or narrowband_noise):
                    stimuli_name = train_path_pattern.split("/")[-2]
                    np.save(newpath+'/batch_array_{}_iter{}.npy'.format(stimuli_name,stim),batch_acc)
                    np.save(newpath+'/batch_conditional_{}_iter{}.npy'.format(stimuli_name,stim),batch_conditional)
                    acc_corr=[pred[0] for pred in batch_acc]
                    acc_accuracy=sum(acc_corr)/len(acc_corr)
                    if branched:
                        np.save(newpath+'/plot_array_test_{}_2.npy'.format(stim),batch_acc2)
                        np.save(newpath+'/batch_conditional_test_{}_2.npy'.format(stim),batch_conditional2)
                        acc_corr2=[pred2[0] for pred2 in batch_acc2]
                        acc_accuracy2=sum(acc_corr2)/len(acc_corr2)
                    with open(newpath+'/accuracies_test_{}_iter{}.json'.format(stimuli_name,stim),'w') as f:
                        json.dump(acc_accuracy,f)
                        if branched:
                            json.dump(acc_accuracy2,f)
                    with open(newpath+'/keys_test_{}_iter{}.json'.format(stimuli_name,stim),'w') as f:
                        json.dump(eval_keys,f)

                else:
                    stimuli_name = train_path_pattern.split("/")[-2]
                    np.save(newpath+'/plot_array_padded_{}_iter{}.npy'.format(stimuli_name,stim),batch_acc)
                    np.save(newpath+'/batch_conditional_{}_iter{}.npy'.format(stimuli_name,stim),batch_conditional)
                    acc_corr=[pred[0] for pred in batch_acc]
                    acc_accuracy=sum(acc_corr)/len(acc_corr)
                    if branched:
                        np.save(newpath+'/plot_array_stim_vary_env_{}_2.npy'.format(stim),batch_acc2)
                        np.save(newpath+'/batch_conditional_test_{}_2.npy'.format(stim),batch_conditional2)
                        acc_corr2=[pred2[0] for pred2 in batch_acc2]
                        acc_accuracy2=sum(acc_corr2)/len(acc_corr2)
                    with open(newpath+'/accuracies_test_{}_iter{}.json'.format(stimuli_name,stim),'w') as f:
                        json.dump(acc_accuracy,f)
                        if branched:
                            json.dump(acc_accuracy2,f)
                    with open(newpath+'/keys_test_{}_iter{}.json'.format(stimuli_name,stim),'w') as f:
                        json.dump(eval_keys,f)
 
 
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

