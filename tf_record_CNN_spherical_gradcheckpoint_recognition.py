import os
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'
from math import sqrt
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
from NetBuilder import NetBuilder
from layer_generator import generate

import memory_saving_gradients
from tensorflow.python.ops import gradients
#import mem_util

# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
def gradients_memory(ys, xs, grad_ys=None, **kwargs):
    return memory_saving_gradients.gradients(ys, xs, grad_ys, checkpoints='memory', **kwargs)
gradients.__dict__["gradients"] = memory_saving_gradients.gradients_collection

def tf_record_CNN_spherical(tone_version,itd_tones,ild_tones,manually_added,freq_label,all_positions_bkgd,testing,branched,zero_padded,stacked_channel,model_version,num_epochs,train_path_pattern,bkgd_train_path_pattern,arch_ID,config_array,files,num_files,newpath):

    bkgd_training_paths = glob.glob(bkgd_train_path_pattern)
    training_paths = glob.glob(train_path_pattern)
    print (bkgd_training_paths)

    ###Do not change parameters below unless altering network###

    BKGD_SIZE = [72,59099]
    STIM_SIZE = [72,89999]
    TONE_SIZE = [72,59099]
    ITD_TONE_SIZE = [72,39690]
    if zero_padded:
        STIM_SIZE = [72,30000]

    if stacked_channel:
        STIM_SIZE = [36,30000, 2]
        BKGD_SIZE = [36,30000,2]
    SNR =10
    SNR_max = 40
    SNR_min = 5
    ###AGgressive monkeypathing###
    #n_classes_localization=252
    n_classes_localization=780
    n_classes_recognition=780

    #Optimization Params
    batch_size=16
    learning_rate = 1e-3
    loss_exponent = 12
    loss_scale = 2**loss_exponent
    bn_training_state = True
    dropout_training_state = True
    training_state = True 
    #Queue parameters
    dequeue_min = 80
    dequeue_min_main = 1000
    #num_epochs = None
    #Change for network precision,must match input data type
    filter_dtype = tf.float32

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

        def combine_signal_and_noise_stacked_channel(signal,background,snr,delay):
            sig_len= signal.shape[1] - delay
            sig = tf.slice(signal,[0,0,0],[36,sig_len,2])
            max_val = tf.reduce_max(sig)
            sig_rms = rms(tf.reduce_sum(sig,[0,2]))
            sig = tf.div(sig,sig_rms)
            #sig = tf.Print(sig, [tf.reduce_max(sig)],message="\nMax SIG:")
            sf = tf.pow(tf.constant(10,dtype=tf.float32),tf.div(snr,tf.constant(20,dtype=tf.float32)))
            bak_rms = rms(tf.reduce_sum(background,[0,2]))
            #bak_rms = tf.Print(bak_rms, [tf.reduce_max(bak_rms)],message="\nNoise RMS:")
            sig_rms = rms(tf.reduce_sum(sig,[0,2]))
            scaling_factor = tf.div(tf.div(sig_rms,bak_rms),sf)
            #scaling_factor = tf.Print(scaling_factor, [scaling_factor],message="\nScaling Factor:")
            noise = tf.scalar_mul(scaling_factor,background)
            #noise = tf.Print(noise, [tf.reduce_max(noise)],message="\nMax Noise:")
            front = tf.slice(noise,[0,0,0],[36,delay,2])
            middle = tf.slice(noise,[0,delay,0],[36,sig_len,2])
            end = tf.slice(noise,[0,(delay+int(sig_len)),0],[36,-1,2])
            middle_added = tf.add(middle,sig)
            new_sig = tf.concat([front,middle_added,end],1)
            #new_sig = sig
            rescale_factor = tf.div(max_val,tf.reduce_max(new_sig))
            #rescale_factor = tf.Print(rescale_factor, [rescale_factor],message="\nRescaling Factor:")
            new_sig = tf.scalar_mul(rescale_factor,new_sig)
            new_sig_rectified = tf.nn.relu(new_sig)
            new_sig_nonlin = tf.pow(new_sig_rectified,0.3)
            new_sig_reshaped = tf.reshape(new_sig_nonlin,[36,30000,2])
            #new_sig_reshaped = tf.reshape(new_sig,[72,30000,1])
            #return (signal, background,noise,new_sig_reshaped)
            return new_sig_reshaped

        def combine_signal_and_noise(signal,background,snr,delay):
            sig_len= signal.shape[1] - delay
            sig = tf.slice(signal,[0,0],[72,sig_len])
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
            front = tf.slice(noise,[0,0],[72,delay])
            middle = tf.slice(noise,[0,delay],[72,sig_len])
            end = tf.slice(noise,[0,(delay+int(sig_len))],[72,-1])
            middle_added = tf.add(middle,sig)
            new_sig = tf.concat([front,middle_added,end],1)
            #new_sig = sig
            rescale_factor = tf.div(max_val,tf.reduce_max(new_sig))
            #rescale_factor = tf.Print(rescale_factor, [rescale_factor],message="\nRescaling Factor:")
            new_sig = tf.scalar_mul(rescale_factor,new_sig)
            new_sig_rectified = tf.nn.relu(new_sig)
            new_sig_nonlin = tf.pow(new_sig_rectified,0.3)
            new_sig_reshaped = tf.reshape(new_sig_nonlin,[72,30000,1])
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
            if not tone_version:
                azim = tf.cast(features['train/azim'], tf.int32)
                elev = tf.cast(features['train/elev'], tf.int32)
                label_div_const = tf.constant([10])
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
            if tone_version:
                image = tf.reshape(image, TONE_SIZE)
                tone = tf.cast(features['train/freq'],tf.int32)
                if itd_tones:
                    azim = tf.cast(features['train/azim'], tf.int32)
                    elev = tf.cast(features['train/elev'], tf.int32)
                    label_div_const = tf.constant([10])
                    if not manually_added:
                        azim = tf.div(azim,label_div_const)
                        elev = tf.div(elev,label_div_const)
                    return image, azim, elev, tone
                else:
                    label = tf.cast(features['train/label'], tf.int32)
                    label_div_const = tf.constant([10])
                    label = tf.div(label,label_div_const)
                return image,label, tone

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
        else:
            bkgd_images, bkgd_labels = iterator_bkgd.get_next()

        SNR = tf.random_uniform([],minval=SNR_min,maxval=SNR_max,name="snr_gen")
        if stacked_channel:
            images = tf.slice(images,[0,0,0],[36,30000,2])
            bkgd_images = tf.slice(bkgd_images,[0,0,0],[36,30000,2])
            combined_subbands = combine_signal_and_noise_stacked_channel(images,bkgd_images,SNR,0)
        else:
            images = tf.slice(images,[0,0],[72,30000])
            bkgd_images = tf.slice(bkgd_images,[0,0],[72,30000])
            combined_subbands =  combine_signal_and_noise(images,bkgd_images,SNR,0)
        combined_subbands = tf.cast(combined_subbands, filter_dtype)

        if itd_tones:
            subbands_batch, azims_batch,elevs_batch,tones_batch = tf.train.shuffle_batch([combined_subbands,azims,elevs,tones],batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")

        elif tone_version:
            subbands_batch, azims_batch,tones_batch = tf.train.shuffle_batch([combined_subbands,labels,tones],batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")
        elif branched:
            subbands_batch, azims_batch, elevs_batch, class_num_batch= tf.train.shuffle_batch([combined_subbands,azims,elevs,class_num],batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")
        elif freq_label:
            subbands_batch, azims_batch,elevs_batch,tones_batch = tf.train.shuffle_batch([combined_subbands,azims,elevs,tones],batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")
        else:
            subbands_batch, azims_batch, elevs_batch = tf.train.shuffle_batch([combined_subbands,azims,elevs],batch_size=batch_size,capacity=2000+batch_size*4,num_threads=5,min_after_dequeue=dequeue_min_main,name="example_queue")
        print("queues created")

    ###END READING QUEUE MACHINERY###


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
    padding='SAME'
    
    
#    config_array=[[["/gpu:0"],['conv',[2,50,32],[2,1]],['relu'],['pool',[1,4]]],[["/gpu:1"],['conv',[4,20,64],[1,1]],['bn'],['relu'],['pool',[1,4]],['conv',[8,8,128],[1,1]],['bn'],['relu'],['pool',[1,4]],['conv',[8,8,256],[1,1]],['bn'],['relu'],['pool',[1,8]],['fc',512],['fc_bn'],['fc_relu'],['dropout'],['out',]]]



   # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='fp32_storage'))
   # print(subbands_batch)
   
   ####TMEPORARY OVERRIDE####
    
    branched = False
    net=NetBuilder()
    if branched:
        out,out2=net.build(config_array,subbands_batch,training_state,dropout_training_state,filter_dtype,padding,n_classes_localization,n_classes_recognition,branched)
    else:
        out=net.build(config_array,subbands_batch,training_state,dropout_training_state,filter_dtype,padding,n_classes_localization,n_classes_recognition,branched)
    
    
    ##Fully connected Layer 2
    #wd2 = tf.get_variable('wd2',[512,512],filter_dtype)
    #dense_bias2 = tf.get_variable('wb6',[512],filter_dtype)
    #fc2 = tf.add(tf.matmul(fc1_do, wd2), dense_bias2)
    #fc2 = tf.nn.relu(fc2)
    #fc2_do = tf.layers.dropout(fc2,training=dropout_training_state)


    # Construct model
    #fix labels dimension to be one less that logits dimension

    #Testing small subbatch
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
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=class_num_batch))
        #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=labels_batch_cost_sphere))
    
    
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
    correct_pred = tf.equal(tf.argmax(out, 1), tf.cast(class_num_batch,tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #correct_pred = tf.equal(tf.argmax(out, 1), tf.cast(labels_batch_cost_sphere,tf.int64))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
    #check_op = tf.add_check_numerics_ops()
    
    
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
    print("Class Labels:" + str(sess.run(class_num_batch)))
    #sess.run(cost)
    time.sleep(30)
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
                     loss, acc = sess.run([cost, accuracy])
                     print("Examples in Queue:",sess.run('example_queue/random_shuffle_queue_Size:0'))
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
            else:
                stimuli_name = train_path_pattern.split("/")[-2]
                np.save(newpath+'/plot_array_padded_{}_iter{}.npy'.format(stimuli_name, model_version),batch_acc)
                np.save(newpath+'/batch_conditional_{}_iter{}.npy'.format(stimuli_name, model_version),batch_conditional)
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

