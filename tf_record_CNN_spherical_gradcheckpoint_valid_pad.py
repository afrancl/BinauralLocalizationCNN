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



        def combine_signal_and_noise_stacked_channel(signal,background,snr,delay,
                                                     sr,cochleagram_sr,post_rectify):
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
            return new_sig_reshaped

        def combine_signal_and_noise(signal,background,snr,delay,
                                     sr,cochleagram_sr,post_rectify):
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
            return new_sig_reshaped

        #Best to read https://www.tensorflow.org/api_guides/python/reading_data#Reading_from_files
        ###READING QUEUE MACHINERY###
        #Best to read https://www.tensorflow.org/api_guides/python/reading_data#Reading_from_files

        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/azim': tf.FixedLenFeature([], tf.int64),
                   'train/elev': tf.FixedLenFeature([], tf.int64),
                   'train/image_height': tf.FixedLenFeature([], tf.int64),
                   'train/image_width': tf.FixedLenFeature([], tf.int64) 
                  }
        if tone_version:
            feature = {'train/image': tf.FixedLenFeature([], tf.string),
                       'train/label': tf.FixedLenFeature([], tf.int64),
                       'train/image_height': tf.FixedLenFeature([], tf.int64),
                       'train/image_width': tf.FixedLenFeature([], tf.int64),
                       'train/freq': tf.FixedLenFeature([], tf.int64)
                      }
        if freq_label:
            feature = {'train/azim': tf.FixedLenFeature([], tf.int64),
                       'train/elev': tf.FixedLenFeature([], tf.int64),
                       'train/image': tf.FixedLenFeature([], tf.string),
                       'train/image_height': tf.FixedLenFeature([], tf.int64),
                       'train/image_width': tf.FixedLenFeature([], tf.int64),
                       'train/freq': tf.FixedLenFeature([], tf.int64)
                      }
        if itd_tones:
            feature = {'train/azim': tf.FixedLenFeature([], tf.int64),
                       'train/elev': tf.FixedLenFeature([], tf.int64),
                       'train/image': tf.FixedLenFeature([], tf.string),
                       'train/image_height': tf.FixedLenFeature([], tf.int64),
                       'train/image_width': tf.FixedLenFeature([], tf.int64),
                       'train/freq': tf.FixedLenFeature([], tf.int64)
                      }
        if sam_tones:
            feature = {'train/carrier_freq': tf.FixedLenFeature([], tf.int64),
                       'train/modulation_freq': tf.FixedLenFeature([], tf.int64),
                       'train/carrier_delay': tf.FixedLenFeature([], tf.float32),
                       'train/modulation_delay': tf.FixedLenFeature([], tf.float32),
                       'train/flipped': tf.FixedLenFeature([], tf.int64),
                       'train/image': tf.FixedLenFeature([], tf.string),
                       'train/image_height': tf.FixedLenFeature([], tf.int64),
                       'train/image_width': tf.FixedLenFeature([], tf.int64),
                      }
        if transposed_tones:
            feature = {'train/carrier_freq': tf.FixedLenFeature([], tf.int64),
                       'train/modulation_freq': tf.FixedLenFeature([], tf.int64),
                       'train/delay': tf.FixedLenFeature([], tf.float32),
                       'train/flipped': tf.FixedLenFeature([], tf.int64),
                       'train/image': tf.FixedLenFeature([], tf.string),
                       'train/image_height': tf.FixedLenFeature([], tf.int64),
                       'train/image_width': tf.FixedLenFeature([], tf.int64),
                      }
        if precedence_effect:
            feature = {'train/delay': tf.FixedLenFeature([], tf.float32),
                       'train/start_sample': tf.FixedLenFeature([], tf.int64),
                       'train/lead_level': tf.FixedLenFeature([], tf.float32),
                       'train/lag_level': tf.FixedLenFeature([], tf.float32),
                       'train/flipped': tf.FixedLenFeature([], tf.int64),
                       'train/image': tf.FixedLenFeature([], tf.string),
                       'train/image_height': tf.FixedLenFeature([], tf.int64),
                       'train/image_width': tf.FixedLenFeature([], tf.int64),
                      }
        if narrowband_noise:
            feature = {'train/azim': tf.FixedLenFeature([], tf.int64),
                       'train/elev': tf.FixedLenFeature([], tf.int64),
                       'train/bandwidth': tf.FixedLenFeature([], tf.float32),
                       'train/center_freq': tf.FixedLenFeature([], tf.int64),
                       'train/image': tf.FixedLenFeature([], tf.string),
                       'train/image_height': tf.FixedLenFeature([], tf.int64),
                       'train/image_width': tf.FixedLenFeature([], tf.int64),
                      }
        if branched:
            feature = {'train/image': tf.FixedLenFeature([], tf.string),
                       'train/azim': tf.FixedLenFeature([], tf.int64),
                       'train/elev': tf.FixedLenFeature([], tf.int64),
                       'train/class_num': tf.FixedLenFeature([], tf.int64),
                       'train/image_height': tf.FixedLenFeature([], tf.int64),
                       'train/image_width': tf.FixedLenFeature([], tf.int64) 
                      }

        # Define a reader and read the next record
        def parse_tfrecord_example(record):
            # Decode the record read by the reader
            features = tf.parse_single_example(record, features=feature)
            # Convert the image data from string back to the numbers
            image = tf.decode_raw(features['train/image'], tf.float32)
            #shape = tf.cast(features['train/image_shape'],tf.int32)
            height = tf.cast(features['train/image_height'],tf.int32)
            width = tf.cast(features['train/image_width'],tf.int32)

            # Cast label data into int32
            if tone_version:
                image = tf.reshape(image, TONE_SIZE)
                tone = tf.cast(features['train/freq'],tf.int32)
                if itd_tones:
                    azim = tf.cast(features['train/azim'], tf.int32)
                    elev = tf.cast(features['train/elev'], tf.int32)
                    label_div_const = tf.constant([localization_bin_resolution])
                    if not manually_added:
                        azim = tf.div(azim,label_div_const)
                        elev = tf.div(elev,label_div_const)
                    return image, azim, elev, tone
                else:
                    label = tf.cast(features['train/label'], tf.int32)
                    label_div_const = tf.constant([10])
                    label = tf.div(label,label_div_const)
                return image,label, tone
            elif sam_tones or transposed_tones:
                image = tf.reshape(image,STIM_SIZE)
                carrier_freq = tf.cast(features['train/carrier_freq'], tf.int32)
                modulation_freq = tf.cast(features['train/modulation_freq'], tf.int32)
                flipped = tf.cast(features['train/flipped'], tf.int32)
                if sam_tones:
                    carrier_delay = tf.cast(features['train/carrier_delay'], tf.float32)
                    modulation_delay = tf.cast(features['train/modulation_delay'], tf.float32)
                    return (image, carrier_freq, modulation_freq,
                            carrier_delay, modulation_delay, flipped)
                elif transposed_tones:
                    delay = tf.cast(features['train/delay'], tf.float32)
                    return (image, carrier_freq, modulation_freq,
                            delay, flipped)
            elif precedence_effect:
                image = tf.reshape(image,STIM_SIZE)
                delay = tf.cast(features['train/delay'], tf.float32)
                start_sample = tf.cast(features['train/start_sample'], tf.int32)
                lead_level = tf.cast(features['train/lead_level'], tf.float32)
                lag_level = tf.cast(features['train/lag_level'], tf.float32)
                flipped = tf.cast(features['train/flipped'], tf.int32)
                return (image, delay, start_sample, lead_level,
                        lag_level, flipped)
            elif narrowband_noise:
                image = tf.reshape(image,STIM_SIZE)
                azim = tf.cast(features['train/azim'], tf.int32)
                elev = tf.cast(features['train/elev'], tf.int32)
                label_div_const = tf.constant([localization_bin_resolution])
                azim = tf.div(azim,label_div_const)
                elev = tf.div(elev,label_div_const)
                bandwidth = tf.cast(features['train/bandwidth'], tf.float32)
                center_freq = tf.cast(features['train/center_freq'], tf.int32)
                return (image, azim, elev, bandwidth, center_freq)
            else:
                azim = tf.cast(features['train/azim'], tf.int32)
                elev = tf.cast(features['train/elev'], tf.int32)
                label_div_const = tf.constant([localization_bin_resolution])
                if not manually_added:
                    azim = tf.div(azim,label_div_const)
                    elev = tf.div(elev,label_div_const)
                image = tf.reshape(image,STIM_SIZE)
                # Reshape image data into the original shape
                if branched:
                    class_num = tf.cast(features['train/class_num'], tf.int32)
                    return image, azim, elev, class_num
                if freq_label:
                    tone = tf.cast(features['train/freq'],tf.int32)
                    return image, azim, elev, tone
                return image, azim, elev

        # Creates batches by randomly shuffling tensors
        dataset = tf.data.Dataset.list_files(train_path_pattern).shuffle(len(training_paths))
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(lambda x:tf.data.TFRecordDataset(x,
                                    compression_type="GZIP").map(parse_tfrecord_example,num_parallel_calls=1),
                                    cycle_length=10, block_length=16))
        dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.prefetch(100)
        iterator = dataset.make_one_shot_iterator()
        if itd_tones:
            images, azims, elevs, tones = iterator.get_next()
        elif tone_version:
            images,labels,tones = iterator.get_next()
        elif sam_tones:
            (images, carrier_freq, modulation_freq, carrier_delay,
             modulation_delay, flipped) = iterator.get_next()
        elif transposed_tones:
            (images, carrier_freq, modulation_freq, delay,
             flipped) = iterator.get_next()
        elif precedence_effect:
            (images, delay, start_sample, lead_level,
             lag_level, flipped) = iterator.get_next()
        elif narrowband_noise:
            (images, azim, elev, bandwidth,
             center_freq) = iterator.get_next()
        else:
            if branched:
                images,azims,elevs,class_num= iterator.get_next()
            elif freq_label:
                images, azims, elevs, tones = iterator.get_next()
            else:
                images,azims,elevs= iterator.get_next()
        ###READING QUEUE MACHINERY###

        if all_positions_bkgd:
            bkgd_feature = {'train/image': tf.FixedLenFeature([], tf.string),
                       'train/image_height': tf.FixedLenFeature([], tf.int64),
                       'train/image_width': tf.FixedLenFeature([], tf.int64) 
                      }
        elif background_textures:
            bkgd_feature = {'train/azim': tf.VarLenFeature(tf.int64),
                            'train/elev': tf.VarLenFeature(tf.int64),
                            'train/image': tf.FixedLenFeature([], tf.string),
                            'train/image_height': tf.FixedLenFeature([], tf.int64),
                            'train/image_width': tf.FixedLenFeature([], tf.int64) 
                      }
        else:
            bkgd_feature = {'train/label': tf.FixedLenFeature([], tf.int64),
                       'train/image': tf.FixedLenFeature([], tf.string),
                       'train/image_height': tf.FixedLenFeature([], tf.int64),
                       'train/image_width': tf.FixedLenFeature([], tf.int64) 
                      }

        # Create a list of filenames and pass it to a queue
        bkgd_filename_queue = tf.train.string_input_producer(bkgd_training_paths,
                                                        shuffle=True,
                                                        capacity=len(bkgd_training_paths))
        # Define a reader and read the next record
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        bkgd_reader = tf.TFRecordReader(options=options)
        _, bkgd_serialized_example = bkgd_reader.read(bkgd_filename_queue)
        # Decode the record read by the reader
        def parse_tfrecord_background(record):
            bkgd_features = tf.parse_single_example(record, features=bkgd_feature)
            # Convert the image data from string back to the numbers
            bkgd_image = tf.decode_raw(bkgd_features['train/image'], tf.float32)
            bkgd_height = tf.cast(bkgd_features['train/image_height'],tf.int32)
            bkgd_width = tf.cast(bkgd_features['train/image_width'],tf.int32)
            # Reshape image data into the original shape
            bkgd_image = tf.reshape(bkgd_image, BKGD_SIZE)
            if background_textures:
                bkgd_azim = tf.cast(bkgd_features['train/azim'],tf.int32)
                bkgd_elev = tf.cast(bkgd_features['train/elev'],tf.int32)
                return bkgd_image,bkgd_azim,bkgd_elev
            if not all_positions_bkgd:
                bkgd_label = tf.cast(bkgd_features['train/label'], tf.int32)
                return bkgd_image,bkgd_label
            return bkgd_image

        dataset_bkgd = tf.data.Dataset.list_files(bkgd_train_path_pattern).shuffle(len(bkgd_training_paths))
        dataset_bkgd = dataset_bkgd.apply(tf.contrib.data.parallel_interleave(lambda x:tf.data.TFRecordDataset(x,
                                    compression_type="GZIP").map(parse_tfrecord_background,num_parallel_calls=1),
                                    cycle_length=10, block_length=16))
        dataset_bkgd = dataset_bkgd.shuffle(buffer_size=200)
        dataset_bkgd = dataset_bkgd.repeat()
        dataset_bkgd = dataset_bkgd.prefetch(100)
        iterator_bkgd = dataset_bkgd.make_one_shot_iterator()
        if all_positions_bkgd:
            bkgd_images = iterator_bkgd.get_next()
        elif background_textures:
            bkgd_images, bkgd_azim, bkgd_elev = iterator_bkgd.get_next()
            bkgd_metadata = [bkgd_azim,bkgd_elev]
        else:
            bkgd_images, bkgd_labels = iterator_bkgd.get_next()
        SNR = tf.random_uniform([],minval=SNR_min,maxval=SNR_max,name="snr_gen")
        if stacked_channel:
            images = tf.slice(images,[0,0,0],[39,48000,2])
            bkgd_images = tf.slice(bkgd_images,[0,0,0],[39,48000,2])
            combined_subbands = combine_signal_and_noise_stacked_channel(images,bkgd_images,SNR,0,48000,8000,post_rectify=True)
        else:
            images = tf.slice(images,[0,0],[78,48000])
            bkgd_images = tf.slice(bkgd_images,[0,0],[78,48000])
            combined_subbands =  combine_signal_and_noise(images,bkgd_images,SNR,0,48000,8000,post_rectify=True)
        combined_subbands = tf.cast(combined_subbands, filter_dtype)

        if itd_tones:
            subbands_batch, azims_batch,elevs_batch,tones_batch = tf.train.shuffle_batch([combined_subbands,azims,elevs,tones],batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")

        elif tone_version:
            subbands_batch, azims_batch,tones_batch = tf.train.shuffle_batch([combined_subbands,labels,tones],batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")
        elif sam_tones:
            queue_input_list = [combined_subbands, carrier_freq, modulation_freq, 
                                 carrier_delay, modulation_delay, flipped]
            (subbands_batch, carrier_freq_batch, modulation_freq_batch, 
             carrier_delay_batch, modulation_delay_batch,
             flipped_batch) = tf.train.shuffle_batch(queue_input_list,batch_size=batch_size,
                                                     capacity=2000+batch_size*4,num_threads=5,
                                                     min_after_dequeue=dequeue_min_main,
                                                     name="example_queue")

        elif transposed_tones:
            queue_input_list = [combined_subbands, carrier_freq, modulation_freq, 
                                delay, flipped]

            (subbands_batch, carrier_freq_batch,
             modulation_freq_batch, delay_batch,
             flipped_batch) = tf.train.shuffle_batch(queue_input_list,batch_size=batch_size,
                                                     capacity=2000+batch_size*4,num_threads=5,
                                                     min_after_dequeue=dequeue_min_main,
                                                     name="example_queue")
        elif precedence_effect:
            queue_input_list = [combined_subbands, delay, start_sample, lead_level,
                                lag_level, flipped]

            (subbands_batch, delay_batch,
             start_sample_batch, lead_level_batch,
             lag_level_batch, flipped_batch) = tf.train.shuffle_batch(queue_input_list,batch_size=batch_size,
                                                     capacity=2000+batch_size*4,num_threads=5,
                                                     min_after_dequeue=dequeue_min_main,
                                                     name="example_queue")
        elif narrowband_noise:
            queue_input_list = [combined_subbands, azim, elev, bandwidth,
                                center_freq]

            (subbands_batch, azims_batch,
             elevs_batch, bandwidths_batch,
             center_freqs_batch) = tf.train.shuffle_batch(queue_input_list,batch_size=batch_size,
                                                     capacity=2000+batch_size*4,num_threads=5,
                                                     min_after_dequeue=dequeue_min_main,
                                                     name="example_queue")
        elif branched:
            subbands_batch, azims_batch, elevs_batch, class_num_batch= tf.train.shuffle_batch([combined_subbands,azims,elevs,class_num],batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")
        elif freq_label:
            subbands_batch, azims_batch,elevs_batch,tones_batch = tf.train.shuffle_batch([combined_subbands,azims,elevs,tones],batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")
        else:
            if background_textures:
                queue_input_list = [combined_subbands,azims,elevs] + bkgd_metadata
                subbands_batch, azims_batch, elevs_batch, bkgd_azim, bkgd_elev = tf.train.shuffle_batch(queue_input_list,batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")
            else:
                queue_input_list = [combined_subbands,azims,elevs]
                subbands_batch, azims_batch, elevs_batch = tf.train.shuffle_batch(queue_input_list,batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")
        print("queues created")

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



    [L_channel,R_channel] = tf.unstack(subbands_batch,axis=3)
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
    
    branched = False
    net=NetBuilder()
    if branched:
        out,out2=net.build(config_array,new_sig_nonlin,training_state,dropout_training_state,filter_dtype,padding,n_classes_localization,n_classes_recognition,branched)
    else:
        out=net.build(config_array,new_sig_nonlin,training_state,dropout_training_state,filter_dtype,padding,n_classes_localization,n_classes_recognition,branched)
    
    
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
        labels_batch_cost_sphere = tf.squeeze(tf.zeros_like(carrier_freq_batch))
    elif precedence_effect:
        labels_batch_cost_sphere = tf.squeeze(tf.zeros_like(start_sample_batch))
    else:
        labels_batch_cost = tf.squeeze(azims_batch)
        #labels_batch_cost = tf.squeeze(subbands_batch_labels,axis=[1,2])
        if not tone_version:
            labels_batch_sphere = tf.add(tf.scalar_mul(tf.constant(36,dtype=tf.int32),elevs_batch),
                   azims_batch)
        else:
            labels_batch_sphere = azims_batch
        labels_batch_cost_sphere = tf.squeeze(labels_batch_sphere)
    # Define loss and optimizer
    # On r1.1 reduce mean doees not work(returns nans) with float16 vals
    
    if branched:
        cost1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=labels_batch_cost_sphere))
        cost2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out2,labels=class_num_batch))
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
        correct_pred2 = tf.equal(tf.argmax(out2, 1), tf.cast(class_num_batch,tf.int64))
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
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    print("Filling Queues...")
    if branched:
        print("Class Labels:" + str(sess.run(class_num_batch)))
    #sess.run(cost)
    time.sleep(200)
    print("Examples in Queue:",sess.run('example_queue/random_shuffle_queue_Size:0'))


#     ##This code allows for tracing ops acorss GPUs, you often have to run it twice
#     ##to get sensible traces
# 
#     #sess.run(optimizer,options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
#     #                     run_metadata=run_metadata)
#     #from tensorflow.python.client import timeline
#     #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
#     #trace_file.close()
    if not testing:
         saver = tf.train.Saver(max_to_keep=None)
         learning_curve = []
         errors_count =0
         try:
             step = 1
             sess.graph.finalize()
             while not coord.should_stop():
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
                     loss, acc, az= sess.run([cost, accuracy,azims_batch])
                     print("Examples in Queue:",sess.run('example_queue/random_shuffle_queue_Size:0'))
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
             coord.request_stop()
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
             coord.request_stop()
             return False
         finally:
             print(errors_count)
             print("Training stopped.")
 
         with open(newpath+'/curve_no_resample_w_cutoff_vary_loc.json', 'w') as f:
             json.dump(learning_curve,f)
 

    if testing:
        ##Testing loop
        batch_acc = []
        batch_acc2 = []
        batch_conditional = []
        batch_conditional2 = []
        saver = tf.train.Saver(max_to_keep=None)
        saver.restore(sess,newpath+"/model.ckpt-"+str(model_version))
        step = 0
        try:
            while not coord.should_stop():
                if tone_version or freq_label:
                    pred, ts, ls, cd = sess.run([correct_pred,tones_batch,azims_batch,cond_dist])
                    batch_acc += np.dstack((pred,np.squeeze(ts),np.squeeze(ls))).tolist()[0]
                    batch_conditional += [(cond,label,freq) for cond,label,freq in zip(cd,np.squeeze(ls),np.squeeze(ts))]
                    if branched:
                        pred2, ts, ls, cd2 = sess.run([correct_pred2,tones_batch,azims_batch,cond_dist2])
                        batch_acc2 += np.dstack((pred2,np.squeeze(ts),np.squeeze(ls))).tolist()[0]
                        batch_conditional2 += [(cond,label,freq) for cond,label,freq in zip(cd2,np.squeeze(ls),np.squeeze(ts))]
                elif sam_tones:
                    eval_vars = [carrier_freq_batch, modulation_freq_batch,
                                 carrier_delay_batch, modulation_delay_batch,
                                 flipped_batch,cond_dist]
                    cf, mf, cdel,mdel,flip,cond_pred = sess.run(eval_vars)
                    batch_conditional += [(cond,c_freq,m_freq,c_delay,m_delay,flip_sig) for
                                          cond,c_freq,m_freq,c_delay,m_delay,flip_sig
                                          in zip(cond_pred,np.squeeze(cf),np.squeeze(mf),
                                                 np.squeeze(cdel),np.squeeze(mdel),np.squeeze(flip))]
                elif transposed_tones:
                    eval_vars = [carrier_freq_batch, modulation_freq_batch,
                                 delay_batch, flipped_batch, cond_dist]
                    cf, mf, delay_eval,flip,cond_pred = sess.run(eval_vars)
                    batch_conditional += [(cond,c_freq,m_freq,delay,flip_sig) for
                                          cond,c_freq,m_freq,delay,flip_sig
                                          in zip(cond_pred,np.squeeze(cf),np.squeeze(mf),
                                                 np.squeeze(delay_eval),np.squeeze(flip))]
                elif precedence_effect:
                    eval_vars = [delay_batch,start_sample_batch,
                                 lead_level_batch,lag_level_batch, flipped_batch, cond_dist]
                    db, ssb, lead_lev_b,lag_lev_b,flip,cond_pred = sess.run(eval_vars)
                    batch_conditional += [(cond,del_b,start_s,leadlb,laglb,flip_sig) for
                                          cond,del_b,start_s,leadlb,laglb,flip_sig
                                          in zip(cond_pred,np.squeeze(db),np.squeeze(ssb),
                                                 np.squeeze(lead_lev_b),np.squeeze(lag_lev_b)
                                                 ,np.squeeze(flip))]
                elif narrowband_noise:
                    eval_vars = [azims_batch,elevs_batch, bandwidths_batch,
                                 center_freqs_batch,cond_dist]
                    az, el, bw, cf, cond_pred = sess.run(eval_vars)
                    batch_conditional += [(cond,az_b,el_b,bw_b,cf_b) for
                                          cond,az_b,el_b,bw_b,cf_b
                                          in zip(cond_pred,np.squeeze(az),np.squeeze(el),
                                                 np.squeeze(bw),np.squeeze(cf))]
                else:
                    pred, ts, cd = sess.run([correct_pred,azims_batch,cond_dist])
                    batch_acc += np.dstack((pred,np.squeeze(ts))).tolist()[0]
                    batch_conditional += [(cond,label) for cond,label in zip(cd,np.squeeze(ts))]
                    if branched:
                        pred2, ts, cd2 = sess.run([correct_pred2,class_num_batch,cond_dist2])
                        batch_acc2 += np.dstack((pred2,np.squeeze(ts))).tolist()[0]
                        batch_conditional2 += [(cond,label) for cond,label in zip(cd2,np.squeeze(ts))]
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
            with open(newpath+'/test_error.json','w') as f:
                json.dump(arch_ID,f)
                json.dump(error,f)
            coord.request_stop()
        except tf.errors.OutOfRangeError:
            print("Out of Range Error. Optimization Finished")
            
        finally:
            if tone_version:
                np.save(newpath+'/plot_array_test.npy',batch_acc)
                np.save(newpath+'/batch_conditional_test.npy',batch_conditional)
                acc_corr=[pred[0] for pred in batch_acc]
                acc_accuracy=sum(acc_corr)/len(acc_corr)
                if branched:
                    np.save(newpath+'/plot_array_test_2.npy',batch_acc2)
                    np.save(newpath+'/batch_conditional_test_2.npy',batch_conditional2)
                    acc_corr2=[pred2[0] for pred2 in batch_acc2]
                    acc_accuracy2=sum(acc_corr2)/len(acc_corr2)
                with open(newpath+'/accuracies_itd.json','w') as f:
                    json.dump(acc_accuracy,f)
                    if branched:
                        json.dump(acc_accuracy2,f)
            elif (sam_tones or transposed_tones or 
                  precedence_effect or narrowband_noise):
                stimuli_name = train_path_pattern.split("/")[-2]
                np.save(newpath+'/batch_conditional_{}_iter{}.npy'.format(stimuli_name,model_version),batch_conditional)
                if branched:
                    np.save(newpath+'/batch_conditional_test_2.npy',batch_conditional2)
            else:
                stimuli_name = train_path_pattern.split("/")[-2]
                np.save(newpath+'/plot_array_padded_{}_iter{}.npy'.format(stimuli_name,model_version),batch_acc)
                np.save(newpath+'/batch_conditional_{}_iter{}.npy'.format(stimuli_name,model_version),batch_conditional)
                acc_corr=[pred[0] for pred in batch_acc]
                acc_accuracy=sum(acc_corr)/len(acc_corr)
                if branched:
                    np.save(newpath+'/plot_array_stim_vary_env_2.npy',batch_acc2)
                    np.save(newpath+'/batch_conditional_test_2.npy',batch_conditional2)
                    acc_corr2=[pred2[0] for pred2 in batch_acc2]
                    acc_accuracy2=sum(acc_corr2)/len(acc_corr2)
                with open(newpath+'/accuracies_test.json','w') as f:
                    json.dump(acc_accuracy,f)
                    if branched:
                        json.dump(acc_accuracy2,f)
            coord.request_stop()
 
 
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

