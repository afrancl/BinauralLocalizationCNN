from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import h5py
import numpy as np
from math import ceil
from multiprocessing import Process, Queue, Pool

# from joblib import Parallel, delayed
import os
import scipy.io.wavfile as wav

from pycochleagram import cochleagram as cgram
from pycochleagram import utils
import tempfile

import keras.utils

# import ipdb


# shared_array_base = Array(ctypes.c_double, 10*10)
# shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
# shared_array = shared_array.reshape(10, 10)


# ######## francl ########
# # class IndexedFileReader(object):
# #   """Allows a dataset of a collection of filenames to be index into
# #   like a numpy array."""

# #   def __init__(self, data_src_list):
# #     self.data_src = data_src


# def wrapped_subband_memmap(signal, out_array, start_idx, end_idx):
#   sr = 30000
#   n = 38
#   low_lim = 50
#   high_lim = 15000
#   sample_factor = 1
#   downsample = 6000
#   nonlinearity = 'db'
#   downsample = None
#   nonlinearity = None

#   # out_array[start_idx:end_idx] = cgram.human_cochleagram(signal[start_idx:end_idx], sr, n=n, low_lim=low_lim, hi_lim=high_lim,
#   out_array[start_idx:end_idx] = cgram.human_cochleagram(signal, sr, n=n, low_lim=low_lim, hi_lim=high_lim,
#     sample_factor=sample_factor, pad_factor=None, downsample=downsample, nonlinearity=nonlinearity,
#     fft_mode='np', ret_mode='subband', strict=True).astype(np.float32)
#   out_array.flush()


# def wrapped_subband(signal):
#   # signal = wav.read(signal)[1][0:30000, 0]
#   # signal = wav.read(signal)[1].swapaxes(1, 0)[:, :30000]
#   print(signal.shape)

#   sr = 30000
#   n = 38
#   low_lim = 50
#   high_lim = 15000
#   sample_factor = 1
#   downsample = 6000
#   nonlinearity = 'db'
#   downsample = None
#   nonlinearity = None

#   subband = cgram.human_cochleagram(signal, sr, n=n, low_lim=low_lim, hi_lim=high_lim,
#     sample_factor=sample_factor, pad_factor=None, downsample=downsample, nonlinearity=nonlinearity,
#     fft_mode='auto', ret_mode='subband', strict=True)
#   return subband.astype(np.float32)
#   # return None


# def run_francl_test(in_dir, n_per_batch=8, num_workers=8):
#   num_workers = 8
#   # fntp_all = np.vstack([wav.read(os.path.join(in_dir,f))[1][:30000].swapaxes(1,0) for f in os.listdir(in_dir) if f.endswith('.wav')])
#   fntp_all = [wav.read(os.path.join(in_dir,f))[1][:30000].swapaxes(1,0) for f in os.listdir(in_dir) if f.endswith('.wav')]
#   # fntp = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.endswith('.wav')]
#   with Parallel(n_jobs=num_workers) as parallel:
#     # cgram_shape = utils.compute_cochleagram_shape(x_dataset.shape[1], sr, n, sample_factor, downsample)
#     # cgram_shape = (batch_size, *cgram_shape)

#     cgram_shape = (n_per_batch * 2, 40, 30000)
#     print(cgram_shape)
#     folder = tempfile.mkdtemp()
#     out = os.path.join(folder, 'cgram_out')
#     cgram_memmap = np.memmap(out, dtype=np.float32, shape=cgram_shape, mode='w+')

#     idx = 0
#     while True:
#       idx += 1
#       list_start_ind = idx * n_per_batch
#       list_end_ind = (idx + 1) * n_per_batch
#       fntp = np.vstack(fntp_all[list_start_ind:list_end_ind])

#       batch_size = fntp.shape[0]
#       start_ind = list_start_ind * 2
#       end_ind = list_end_ind * 2
#       subbatch_size = ceil(batch_size / num_workers)
#       subbatch_idx_list = [(x*subbatch_size, (x+1)*subbatch_size) for x in range(num_workers)]
#       print('memmap cgram out shape ', cgram_shape, '[%s, %s]' % (start_ind, end_ind), subbatch_idx_list, fntp.shape)
#       # pdb.set_trace()
#       start_time = time.time()
#       parallel(delayed(wrapped_subband_memmap)(fntp[start:end], cgram_memmap, start, end) for start, end in subbatch_idx_list)
#       results = cgram_memmap
#       # results = np.vstack(parallel(delayed(wrapped_subband)(fntp[start:end]) for start, end in subbatch_idx_list))
#       # wrapped_subband(fntp)
#       print(results.shape)
#       print(time.time() - start_time)
#       yield results


######## raygon ########
# def wrapped_cochleagram_memmap(signal, out_array, start_idx, end_idx):
#   sr = 48000
#   n = 38
#   low_lim = 50
#   high_lim = 20000
#   sample_factor = 4
#   downsample = 6000
#   nonlinearity = 'db'
#   # downsample = None
#   # nonlinearity = None
#   print('wrapped memmap cgram [%s %s]' % (start_idx, end_idx))

#   out_array[start_idx:end_idx] = cgram.human_cochleagram(signal, sr, n=n, low_lim=low_lim, hi_lim=high_lim,
#     sample_factor=sample_factor, pad_factor=None, downsample=downsample, nonlinearity=nonlinearity,
#     fft_mode='auto', ret_mode='envs', strict=True).astype(np.float32)


def wrapped_cochleagram(signal, extra_data=[]):
  sr = 48000
  n = 38
  low_lim = 50
  high_lim = 20000
  sample_factor = 4
  downsample = 6000
  nonlinearity = 'db'
  # downsample = None
  # nonlinearity = None

  # print('computing coch %s' % signal.shape) ## WOAHH BREAKS SHIT
  # print('computing coch %s' % str(signal.shape)) ## Works fine

  # ipdb.set_trace()
  print('wrapped_cochleagram in misc.py')
  coch = cgram.human_cochleagram(signal, sr, n=n, low_lim=low_lim, hi_lim=high_lim,
    sample_factor=sample_factor, pad_factor=None, downsample=downsample, nonlinearity=nonlinearity,
    fft_mode='auto', ret_mode='envs', strict=True, no_hp_lp_filts=True)

  return (coch.astype(np.float32), *extra_data)


def wrapped_cochleagram_main(signal, extra_data=[]):
  sr = 48000
  n = 38
  low_lim = 50
  high_lim = 20000
  sample_factor = 2
  downsample = 6000
  nonlinearity = 'db'
  # downsample = None
  # nonlinearity = None

  coch = cgram.human_cochleagram(signal, sr, n=n, low_lim=low_lim, hi_lim=high_lim,
    sample_factor=sample_factor, pad_factor=None, downsample=downsample, nonlinearity=nonlinearity,
    fft_mode='auto', ret_mode='envs', strict=True, no_hp_lp_filts=True)
  # print('computing coch %s' % str(coch.shape))
  out = coch.astype(np.float32)
  out = np.expand_dims(out, axis=-1)
  # print('OUT SHAPE ', out.shape, flush=True)
  labels = keras.utils.to_categorical(extra_data[0], 104)
  # print(labels.shape)
  return (out, labels)


# def queued_generator_subbatched_helper(output_queue, data_src, callback_fx, num_epochs, batch_size, num_workers, extra_data_src_list=[]):
#   subbatch_size = ceil(batch_size / num_workers)
#   num_per_epoch = len(data_src) // batch_size

#   cgram_shape = utils.compute_cochleagram_shape(data_src.shape[1], 48000, 38, 4, 6000)
#   cgram_shape = (batch_size, *cgram_shape)
#   print('memmap cgram out shape ', cgram_shape)
#   # folder = tempfile.mkdtemp()
#   # out = os.path.join(folder, 'cgram_out')
#   # batch_data_memmap = np.memmap(out, dtype=np.float32, shape=cgram_shape, mode='r+')

#   def output_callback(worker_data):
#     # print('CALLBACK ---->>>', worker_data)
#     print(output_queue)
#     output_queue.put(worker_data)

#   pool = Pool(processes=num_workers)
#   do_loop = True
#   while do_loop:
#     for epoch in range(num_epochs):
#       for idx in range(num_per_epoch):
#         print('>>>> E: %s B: %s/%s' % (epoch + 1, idx + 1, num_per_epoch))

#         # compute the slice of the dataset
#         start_idx = idx * batch_size
#         end_idx = (idx + 1) * batch_size
#         batch_data = data_src[start_idx:end_idx]

#         # break the slice into sub-slices for multiprocessing
#         subbatch_idx_list = [(x * subbatch_size, (x + 1) * subbatch_size) for x in range(num_workers)]
#         print('[%s, %s]' % (start_idx, end_idx), subbatch_idx_list)
#         # print(batch_data.shape)

#         # generate cochleagrams on the sub-slices
#         # deferred = [pool.apply_async(callback_fx, (batch_data[start:end], batch_data_memmap),) for start, end in subbatch_idx_list]
#         deferred = [pool.apply(wrapped_cochleagram, (batch_data[start:end],)) for start, end in subbatch_idx_list]
#         # deferred = [pool.apply_async(wrapped_cochleagram, (batch_data[start:end],), callback=output_callback) for start, end in subbatch_idx_list]
#         # deferred = [pool.apply_async(wrapped_cochleagram_memmap, (batch_data[start:end], batch_data_memmap, start, end)) for start, end in subbatch_idx_list]
#         # deferred = [pool.apply(wrapped_cochleagram_memmap, (batch_data[start:end], batch_data_memmap, start, end),) for start, end in subbatch_idx_list]
#         test = ([d.get() for d in deferred])
#         # print(test)


#         # slice any extra data
#         batch_extra = [x[start_idx:end_idx] for x in extra_data_src_list]
#         # print(batch_extra)

#         # reassemble the subslices
#         # **NOTE: due to the nature of apply_async, the sub-slices might
#         # appear in a different order than they are in the slice, though the
#         # entire slice data is the same (data but not order is preserved within a slice/batch)
#         # batch_data = np.vstack([d.get() for d in deferred])
#         # print(batch_data.shape)
#         # print(np.unique(np.array(batch_data_memmap)))

#         # output_queue.put((batch_data, *batch_extra))
#         # output_queue.put((batch_data_memmap.copy(), *batch_extra))
#     do_loop = False
#   pool.terminate()
#   # pool.close()
#   pool.join()

#   output_queue.close()
#   # output_queue.join_thread()


# def queued_generator_subbatched(data_src, callback_fx, num_epochs, batch_size, num_workers, extra_data_src_list=[], queue_prewait_seconds=0):
#   # sanity check data_src shapes
#   if len(extra_data_src_list) > 0:
#     if not all([len(d) == len(data_src) for d in extra_data_src_list]):
#       raise ValueError('All data sources must have the same length')

#   num_inputs = len(data_src)
#   batches_per_epoch = ceil(num_inputs / batch_size)
#   total_batches = batches_per_epoch * num_epochs

#   print('Queued Generator --> %s: num_workers: %s batch_size: %s prewait_seconds: %s' % (callback_fx.__name__, num_workers, batch_size, queue_prewait_seconds))
#   out_queue = Queue()
#   p = Process(target=queued_generator_subbatched_helper, args=(out_queue, data_src, callback_fx, num_epochs, batch_size, num_workers, extra_data_src_list))
#   try:
#     p.start()
#     time.sleep(queue_prewait_seconds)
#     for i in range (total_batches):
#       out = out_queue.get()
#       # print(len(out))
#       print(out.shape)
#       # print(out)
#       if out:
#         yield out
#     p.join()
#     out_queue.close()
#     out_queue.join_thread()
#   except Exception as e:
#     raise e
#   # finally:
#   #   print('Shutting down queue')
#   #   p.join()
#   #   out_queue.close()
#   #   out_queue.join_thread()


def queued_generator_helper(output_queue, data_src, callback_fx, num_epochs, batch_size, num_workers, extra_data_src_list=[], oncomplete_callback=None):
  num_per_epoch = ceil(len(data_src) / batch_size)

  # cgram_shape = utils.compute_cochleagram_shape(data_src.shape[1], 48000, 38, 4, 6000)
  # cgram_shape = (batch_size, *cgram_shape)
  # print('memmap cgram out shape ', cgram_shape)

  if oncomplete_callback is None:
    def oncomplete_callback(worker_data):
      output_queue.put(worker_data)
      print(output_queue.qsize())

  pool = Pool(processes=num_workers)
  for epoch in range(num_epochs):
    batch_idx_list = [(b * batch_size, (b + 1) * batch_size) for b in range(num_per_epoch)]
    print('\tgeneration epoch --> %s/%s' % (epoch + 1, num_epochs))
    assert num_per_epoch == len(batch_idx_list), 'computed and emperical number of batches aren different.'

    # NOTE: serializing here could get expensive with extra_data_src_list
    # deferred = [pool.apply_async(wrapped_cochleagram, (data_src[start:end], [x[start:end] for x in extra_data_src_list]), callback=oncomplete_callback) for start, end in batch_idx_list]
    deferred = [pool.apply_async(callback_fx, (data_src[start:end], [x[start:end] for x in extra_data_src_list]), callback=oncomplete_callback) for start, end in batch_idx_list]
    # deferred = [pool.apply_async(wrapped_cochleagram, (data_src[start:end],), callback=oncomplete_callback) for start, end in batch_idx_list]

  # we don't want to close the queue here, since workers are still adding to it,
  # but we are done creating tasksf for workers so close the process pool
  # pool.terminate()  # this will kill the pool too fast
  pool.close()
  pool.join()  #TODO: consider using a single thread pool and joining in main thread


def queued_generator(data_src, callback_fx, num_epochs, batch_size, num_workers, extra_data_src_list=[], queue_prewait_seconds=0):
  # sanity check data_src shapes
  if len(extra_data_src_list) > 0:
    if not all([len(d) == len(data_src) for d in extra_data_src_list]):
      raise ValueError('All data sources must have the same length')

  num_inputs = len(data_src)
  batches_per_epoch = ceil(num_inputs / batch_size)
  total_batches = batches_per_epoch * num_epochs

  print('queued_generator --> %s: num_workers: %s batch_size: %s prewait_seconds: %s' % (callback_fx.__name__, num_workers, batch_size, queue_prewait_seconds))
  out_queue = Queue()
  p = Process(target=queued_generator_helper, args=(out_queue, data_src, callback_fx, num_epochs, batch_size, num_workers, extra_data_src_list))
  try:
    p.start()
    time.sleep(queue_prewait_seconds)
    for i in range (total_batches):
      print('\t\tfetching data %s/%s' % (i, total_batches))
      out = out_queue.get()
      if out is not None:
        yield out
    # p.join()  # pool is created/destroyed for each epoch of data
    out_queue.close()
    out_queue.join_thread()
  except StopIteration:
    print("Internal generator is done")
    p.terminate()
    out_queue.close()
    out_queue.join_thread()
  except Exception as e:
    p.terminate()
    out_queue.close()
    out_queue.join_thread()
    out_queue.cancel_join_thread()  # i'm not sure what this does, but we just want to exit quickly
    raise e


def cochleagram_generator(data_src, num_epochs, batch_size, num_workers, extra_data_src_list=[], queue_prewait_seconds=0):
  if num_workers == 1:
    def single_thread_gx():
      num_inputs = len(data_src)
      batches_per_epoch = ceil(num_inputs / batch_size)
      total_batches = batches_per_epoch * num_epochs

      print('single process cgram generator --> %s: num_workers: %s batch_size: %s prewait_seconds: %s' % (wrapped_cochleagram.__name__, num_workers, batch_size, queue_prewait_seconds))
      for i in range (total_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        print('\t\tfetching data %s/%s [%s, %s]' % (i, total_batches, start, end))
        yield wrapped_cochleagram(data_src[start:end], [x[start:end] for x in extra_data_src_list])
    cgram_gx = single_thread_gx()
  else:
    # cgram_gx = queued_generator_subbatched(data_src, wrapped_cochleagram, num_epochs, batch_size, num_workers=num_workers,
    #                             extra_data_src_list=extra_data_src_list, queue_prewait_seconds=queue_prewait_seconds)
    cgram_gx = queued_generator(data_src, wrapped_cochleagram_main, num_epochs, batch_size, num_workers=num_workers,
                              extra_data_src_list=extra_data_src_list, queue_prewait_seconds=queue_prewait_seconds)
  # ipdb.set_trace()
  return cgram_gx


def main():
  num_workers = 8
  num_epochs = 3
  batch_size = 100
  queue_prewait_seconds = 10
  data_fn = '../../deepFerret/src/sanity/dfSynthStimSmall.h5'
  with h5py.File(data_fn, 'r') as h5rf:
    print('starting read')
    start = time.time()
    labels_data = h5rf['labels'].value
    signal_data = h5rf['signal'].value
    print('end read: %s' % (time.time() - start))
    extra_data_src_list = (labels_data,)
    cgram_gx = cochleagram_generator(signal_data, num_epochs, batch_size, num_workers, extra_data_src_list, queue_prewait_seconds)

    big_start_time = time.time()
    batch_ctr = 0
    try:
      while True:
      # for i, temp_batch in enumerate(cgram_gx):
        batch_ctr += 1
        start_time = time.time()
        temp_batch = next(cgram_gx)
        temp_time = time.time() - start_time

        summary_str = 'len: %s t: %0.2fs %s' % (len(temp_batch), temp_time, str(temp_batch[0].shape))
        print('dequeued batch %s > %s' % (batch_ctr, summary_str))
    except StopIteration:
      print('exhausted generator')
    total_time = time.time() - big_start_time
    print('Total run time: %0.2f %0.2f s/100 cgrams' %  (total_time, total_time / (batch_ctr * batch_size) * 100))


def francl_main():
  n_per_batch = 40
  gx = run_francl_test('/Users/raygon/Desktop/mdLab/projects/public/cochleagram/sandbox/francl_testStimOnTheFlyCgramGeneration', n_per_batch=n_per_batch, num_workers=8)
  start_time = time.time()
  [next(gx) for x in gx]
  total_time = time.time() -  start_time
  print('total time: %s > %s s/cgram' % (total_time, total_time / (num_epochs * n_per_batch)))


if __name__ == '__main__':
  main()
