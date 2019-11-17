# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for AWD ENAS search process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf

from enas_lm.src import child
from enas_lm.src import controller
from enas_lm.src import utils
from enas_lm.src.vocab import Vocab
import enas_lm.src.constants as constants
from enas_lm.src.data_loader import DataLoader
import copy


flags = tf.app.flags
gfile = tf.gfile
FLAGS = flags.FLAGS

#flags.DEFINE_boolean('reset_output_dir', False, '/Volumes/External HDD/dataset/reset_output')
#flags.DEFINE_string('output_dir', '/Volumes/External HDD/dataset/output', '')
#flags.DEFINE_string('data_path', '/Volumes/External HDD/dataset/tacred/data/json', '')
#flags.DEFINE_string('vocab_file', '/Volumes/External HDD/dataset/tacred/data/vocab/vocab.pkl', '')
flags.DEFINE_boolean('reset_output_dir', False, '/usr0/home/gis/research/enas_re/tmp/datasets/tacred/reset_output')
flags.DEFINE_string('output_dir', '/usr0/home/gis/research/enas_re/tmp/datasets/tacred/output', '')
flags.DEFINE_string('data_path', '/usr0/home/gis/research/enas_re/tmp/datasets/tacred/data/json', '')
flags.DEFINE_string('vocab_file', '/usr0/home/gis/research/enas_re/tmp/datasets/tacred/vocab/vocab.pkl', '')
flags.DEFINE_string('token_emb_file', '/usr0/home/gis/research/enas_re/tmp/datasets/tacred/vocab/embedding.npy', '')
flags.DEFINE_integer('log_every', 200, '')
flags.DEFINE_integer('batch_size', 50, '')


def set_default_args(params):
    params.add_hparam('controller_hidden_size', 64)
    params.add_hparam('controller_num_layers', FLAGS.controller_num_layers)
    params.add_hparam('controller_num_functions', 4)  # tanh, relu, sigmoid, iden

    params.add_hparam('controller_baseline_dec', FLAGS.controller_baseline_dec)
    params.add_hparam('controller_entropy_weight',
                      FLAGS.controller_entropy_weight)
    params.add_hparam('controller_temperature', FLAGS.controller_temperature)
    params.add_hparam('controller_tanh_constant', FLAGS.controller_tanh_constant)
    params.add_hparam('controller_learning_rate', FLAGS.controller_learning_rate)
    params.add_hparam('controller_num_aggregate', 10)
    params.add_hparam('controller_num_train_steps', 25)

    params.add_hparam('alpha', 0.0)  # activation L2 reg
    params.add_hparam('beta', 1.)  # activation slowness reg
    params.add_hparam('best_valid_ppl_threshold', 5)

    # params.add_hparam('batch_size', FLAGS.child_batch_size)
    # params.add_hparam('bptt_steps', FLAGS.child_bptt_steps)

    # for dropouts: dropping rate, NOT keeping rate
    params.add_hparam('drop_e', 0.10)  # word
    params.add_hparam('drop_i', 0.20)  # embeddings
    params.add_hparam('drop_x', 0.75)  # input to RNN cells
    params.add_hparam('drop_l', 0.25)  # between layers
    params.add_hparam('drop_o', 0.75)  # output
    params.add_hparam('drop_w', 0.00)  # weight

    params.add_hparam('grad_bound', 0.1)
    params.add_hparam('hidden_size', 200)
    params.add_hparam('init_range', 0.04)
    params.add_hparam('learning_rate', 20.)
    params.add_hparam('num_train_epochs', 600)
    # params.add_hparam('vocab_size', 10000)

    params.add_hparam('weight_decay', 8e-7)

    return params


def get_ops(params, x_train, x_valid):
  """Build [train, valid, test] graphs."""


  ct = controller.Controller(params=params)
  lm = child.LM(params, ct)
  ct.build_trainer(lm)
  ops = {
      'train_op': lm.train_op,
      'learning_rate': lm.learning_rate,
      'grad_norm': lm.grad_norm,
      'train_loss': lm.train_loss,
      'l2_reg_loss': lm.l2_reg_loss,
      'global_step': tf.train.get_or_create_global_step(),
      'reset_batch_states': lm.batch_init_states['reset'],
      'eval_valid': lm.eval_valid,

      # 'reset_start_idx': lm.reset_start_idx,
      # 'should_reset': lm.should_reset,

      'controller_train_op': ct.train_op,
      'controller_grad_norm': ct.train_op,
      'controller_sample_arc': ct.sample_arc,
      'controller_entropy': ct.sample_entropy,
      'controller_reward': ct.reward,
      'controller_baseline': ct.baseline,
      'controller_optimizer': ct.optimizer,
      'controller_train_fn': ct.train,
  }
  # print('-' * 80)
  # print('HParams:\n{0}'.format(params.to_json(indent=2, sort_keys=True)))

  return ops, lm, ct


def train(params):
  """Entry train function."""
  # with gfile.GFile(params.data_path, 'rb') as finp:
  #   x_train, x_valid, _, _, _ = pickle.load(finp)
  #   print('-' * 80)
  #   print('train_size: {0}'.format(np.size(x_train)))
  #   print('valid_size: {0}'.format(np.size(x_valid)))
  x_train = None
  x_valid = None
  params = set_default_args(params)

  vocab = Vocab(FLAGS.vocab_file, load=True)
  token_embs = np.load(FLAGS.token_emb_file)
  params.add_hparam('vocab_size', vocab.size)
  params.add_hparam('num_classes', len(constants.LABEL_TO_ID))
  print('-' * 80)
  print('HParams:\n{0}'.format(params.to_json(indent=2, sort_keys=True)))
  # Skip the embeddings printout
  params.add_hparam('token_embeddings', token_embs)

  g = tf.Graph()
  with g.as_default():
    dataloader = DataLoader(dataset_name='tacred', vocab=vocab, do_lower=params.do_lower)
    train_dataset = dataloader.train_dataset(directory=FLAGS.data_path,
                                           batch_size=FLAGS.batch_size,
                                           dropout_prob=0.)
    valid_dataset = dataloader.eval_dataset(directory=FLAGS.data_path,
                                          batch_size=FLAGS.batch_size,
                                          dataset_type='dev')

    train_iterator = train_dataset.make_one_shot_iterator()
    valid_iterator = valid_dataset.make_initializable_iterator()
    ops, lm, ct = get_ops(params, x_train, x_valid)
    run_ops = [
        ops['train_loss'],
        ops['l2_reg_loss'],
        ops['grad_norm'],
        ops['learning_rate'],
        # ops['should_reset'],
        ops['train_op'],
    ]


    saver = tf.train.Saver(max_to_keep=5)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
        params.output_dir, save_steps=params.num_train_batches, saver=saver)
    hooks = [checkpoint_saver_hook]
    hooks.append(ops['controller_optimizer'].make_session_run_hook(True))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.train.SingularMonitoredSession(config=config, hooks=hooks,
                                             checkpoint_dir=params.output_dir)

    train_iterator_handle = sess.run(train_iterator.string_handle())
    valid_iterator_handle = sess.run(valid_iterator.string_handle())

    accum_loss = 0
    accum_step = 0
    epoch = 0
    best_valid_ppl = []
    start_time = time.time()
    train_step = 0
    while True:
      # try:
      loss, l2_reg, gn, lr, _ = sess.run(run_ops, feed_dict={lm.input_iterator_handle: train_iterator_handle})

      accum_loss += loss
      accum_step += 1
      step = sess.run(ops['global_step'])
      if step % params.log_every == 0:
        train_ppl = np.exp(accum_loss / accum_step)
        mins_so_far = (time.time() - start_time) / 60.
        log_string = 'epoch={0:<5d}'.format(epoch)
        log_string += ' step={0:<7d}'.format(step)
        log_string += ' ppl={0:<9.2f}'.format(train_ppl)
        log_string += ' lr={0:<7.2f}'.format(lr)
        log_string += ' |w|={0:<6.2f}'.format(l2_reg)
        log_string += ' |g|={0:<6.2f}'.format(gn)
        log_string += ' mins={0:<.2f}'.format(mins_so_far)
        print(log_string)

      if train_step > 0 and train_step % params.switch_interval == 0:
        ops['controller_train_fn'](sess, ops['reset_batch_states'],
                                 input_iterator=valid_iterator,
                                 iterator_handle=valid_iterator_handle)
        epoch += 1
        accum_loss = 0
        accum_step = 0
        train_step += 1
        valid_ppl = ops['eval_valid'](sess)
        sess.run([ops['reset_batch_states']])
        best_valid_ppl.append(valid_ppl)

      if step >= params.num_train_steps:
        break
      #except tf.errors.InvalidArgumentError:
       # last_checkpoint = tf.train.latest_checkpoint(params.output_dir)
        #print('rolling back to previous checkpoint {0}'.format(last_checkpoint))
        #saver.restore(sess, last_checkpoint)

    sess.close()


def main(unused_args):
  np.set_printoptions(precision=3, suppress=True, threshold=int(1e9),
                      linewidth=80)

  print('-' * 80)
  if not gfile.IsDirectory(FLAGS.output_dir):
    print('Path {} does not exist. Creating'.format(FLAGS.output_dir))
    gfile.MakeDirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print('Path {} exists. Reseting'.format(FLAGS.output_dir))
    gfile.DeleteRecursively(FLAGS.output_dir)
    gfile.MakeDirs(FLAGS.output_dir)

  print('-' * 80)
  log_file = os.path.join(FLAGS.output_dir, 'stdout')
  print('Logging to {}'.format(log_file))
  sys.stdout = utils.Logger(log_file)

  params = tf.contrib.training.HParams(
      data_path=FLAGS.data_path,
      log_every=FLAGS.log_every,
      output_dir=FLAGS.output_dir,
      pos_dim=30,
      ner_dim=30,
      position_dim=30,
      vocab_dim=300,
      num_pos=len(constants.POS_TO_ID),
      num_ner=len(constants.NER_TO_ID),
      max_len=100,
      do_lower=True,
      num_train_batches=1000,
      num_train_steps=50000,
      switch_interval=1000,
      base_bptt=35,
      bptt_steps=35,
      batch_size=FLAGS.batch_size
  )

  train(params)

if __name__ == '__main__':
  tf.app.run()
