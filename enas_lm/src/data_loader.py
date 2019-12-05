from __future__ import absolute_import, division, print_function

import abc
import glob
import json
import logging
import os
import tarfile
import six
import requests
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import enas_lm.src.constants as constants
import pickle

# TODO: Find way to create batches of variable lengths based on max sequence length per batch
# TODO: Right now every sample is padded to the max sequence size which is most likely not very efficient
class DataLoader(object):
    def __init__(self, dataset_name, vocab, do_lower=False):
        self.dataset_name = dataset_name
        self.do_lower = do_lower
        self.vocab = vocab

    def maybe_create_tf_record_files(self,
                                     directory,
                                     max_records_per_file=10000000):
        print('Creating TF record files for the {} dataset'.format(self.dataset_name))

        files = [constants.TRAIN_JSON, constants.DEV_JSON, constants.TEST_JSON]

        tf_record_filenames = {}
        for idx, json_file in enumerate(files):
            file_idx = 0
            count = 0
            total = 0

            data_basename = json_file.split('.')[0]
            tfrecords_filename = os.path.join(directory, '{0}-{1}.tfrecords'.format(data_basename, file_idx))
            if not os.path.exists(tfrecords_filename):
                tf_records_writer = tf.python_io.TFRecordWriter(tfrecords_filename)
                tf_record_filenames[data_basename] = [tfrecords_filename]

                with open(os.path.join(directory, json_file), 'r') as handle:
                    json_data = json.load(handle)
                processed_data = self.preprocess(json_data)

                json_filename = os.path.join(directory, data_basename + '_processed.pkl')
                self.write_to_pkl(processed_data, json_filename)

                for sample in processed_data:
                    sample_record = self.encode_sample_as_tf_record(sample)
                    tf_records_writer.write(sample_record.SerializeToString())

                    count += 1
                    total += 1

                    if count >= max_records_per_file:
                        tf_records_writer.close()
                        count = 0
                        file_idx += 1
                        tfrecords_filename = os.path.join(directory,
                                                          '{0}-{1}.tfrecords'.format(data_basename, file_idx))
                        tf_record_filenames[data_basename].append(tfrecords_filename)
                        tf_records_writer = tf.python_io.TFRecordWriter(tfrecords_filename)

                tf_records_writer.close()
                print('Total TF records in {}: {}'.format(data_basename, total))
            else:
                print('TF records already exist for {}. We are not recreating them'.format(data_basename))
                tf_record_filenames[data_basename] = glob.glob(os.path.join(directory,
                                                                            '{0}-{1}.tfrecords'.format(data_basename,
                                                                                                       '*')))

        def tf_record_parser(sample):
            features = {
                'tokens': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                'pos': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                'ner': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                'deprel': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                'subj_positions': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                'obj_positions': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                'relation': tf.FixedLenFeature([], tf.int64),
                'masks': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
            }
            return tf.parse_single_example(sample, features=features)

        return tf_record_parser, tf_record_filenames

    def write_to_pkl(self, json_data, filepath):
        with open(filepath, 'wb') as handle:
            pickle.dump(json_data, handle)

    def encode_sample_as_tf_record(self, sample):
        tokens = sample['tokens']
        pos = sample['pos']
        ner = sample['ner']
        deprel = sample['deprel']
        subj_positions = sample['subj_positions']
        obj_positions = sample['obj_positions']
        relation = sample['relation']
        masks = sample['masks']

        def _int64(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=values))

        features = tf.train.Features(feature={
            'tokens': _int64(tokens),
            'pos': _int64(pos),
            'ner': _int64(ner),
            'deprel': _int64(deprel),
            'subj_positions': _int64(subj_positions),
            'obj_positions': _int64(obj_positions),
            'relation': _int64([relation]),
            'masks': _int64(masks)
        })

        return tf.train.Example(features=features)

    def preprocess(self, data):
        processed_data = []
        max_tokens = 0
        for sample in data:
            # print(sample.keys())
            tokens = sample['token']
            if self.do_lower:
                tokens = [token.lower() for token in tokens]
            subj_start, subj_end = sample['subj_start'], sample['subj_end']
            obj_start, obj_end = sample['obj_start'], sample['obj_end']
            # Mask out actual subject/object phrases for more generality
            tokens[subj_start:subj_end+1] = ['SUBJ-'+sample['subj_type']] * (subj_end - subj_start + 1)
            tokens[obj_start:obj_end+1] = ['OBJ-'+sample['obj_type']] * (obj_end - obj_start + 1)

            tokens = self.map_to_ids(tokens, mapping=self.vocab.word2id)
            pos = self.map_to_ids(sample['stanford_pos'], constants.POS_TO_ID)
            ner = self.map_to_ids(sample['stanford_ner'], constants.NER_TO_ID)
            deprel = self.map_to_ids(sample['stanford_deprel'], constants.DEPREL_TO_ID)
            num_tokens = len(tokens)
            subj_positions = self.get_positions(subj_start, subj_end, num_tokens)
            obj_positions = self.get_positions(obj_start, obj_end, num_tokens)
            relation = constants.LABEL_TO_ID[sample['relation']]
            processed_sample = {
                'tokens': tokens,
                'pos': pos,
                'ner': ner,
                'deprel': deprel,
                'subj_positions': subj_positions,
                'obj_positions': obj_positions,
                'relation': relation
            }
            processed_data.append(processed_sample)
            if num_tokens > max_tokens:
                max_tokens = num_tokens
        # pad everything to uniform length
        for sample in processed_data:
            sample['tokens'] = self.pad_ids(sample['tokens'], max_tokens)
            sample['pos'] = self.pad_ids(sample['pos'], max_tokens)
            sample['ner'] = self.pad_ids(sample['ner'], max_tokens)
            sample['deprel'] = self.pad_ids(sample['deprel'], max_tokens)
            sample['subj_positions'] = self.pad_ids(sample['subj_positions'], max_tokens)
            sample['obj_positions'] = self.pad_ids(sample['obj_positions'], max_tokens)

            token_mask = np.equal(sample['tokens'], constants.PAD_ID)
            sample['masks'] = token_mask

        return processed_data

    def pad_ids(self, tokens_list, max_len):
        tokens_list += [constants.PAD_ID] * (max_len - len(tokens_list))
        return tokens_list

    def map_to_ids(self, tokens, mapping):
        ids = [mapping[token] if token in mapping else constants.UNK_ID for token in tokens]
        return ids

    def get_positions(self, start_idx, end_idx, length):
        left_positions = list(range(-start_idx, 0))
        overlapping_postions = [0] * (end_idx - start_idx + 1)
        right_positions = list(range(1, length - end_idx))
        return left_positions + overlapping_postions + right_positions

    def _generate_dropout_mask(self, shape, drop_prob, epsilon=1e-5):
        keep_prob = 1. - drop_prob
        mask = tf.random_uniform(shape, dtype=tf.float32)
        # Ensures keep_prob can range between [0, 1]. i.e.
        # keep_prob=0 doesn't result in divide by zero nans
        dropout_mask = tf.floor(mask + keep_prob)
                                # / tf.cond(tf.less_equal(keep_prob, epsilon), lambda: 1., lambda: keep_prob))
        return dropout_mask

    def train_dataset(self,
                      directory,
                      batch_size,
                      dataset_type='train',
                      cache=False,
                      dropout_prob=.5,
                      num_parallel_readers=32,
                      prefetch_buffer_size=10):
        parser, record_mappings = self.maybe_create_tf_record_files(directory=directory)
        train_files = record_mappings[dataset_type]

        def add_mask(sample):
            sample = parser(sample)
            # Create word dropout mask. Make sure it only dropouts actual words
            dropout_mask = self._generate_dropout_mask(shape=tf.shape(sample['tokens']), drop_prob=dropout_prob)
            dropout_mask = tf.cast(tf.where(tf.cast(sample['masks'], dtype=tf.bool),
                                            tf.zeros(tf.shape(sample['masks'])),
                                            dropout_mask),
                                   dtype=tf.bool)
            tokens = tf.where(dropout_mask,
                              tf.cast(tf.ones((tf.shape(dropout_mask))) * constants.UNK_ID, dtype=tf.int64),
                              sample['tokens'])
            return {
                'tokens': tokens,
                # 'tokens_masked': tokens,
                'pos': sample['pos'],
                'ner': sample['ner'],
                'deprel': sample['deprel'],
                'subj_positions': sample['subj_positions'],
                'obj_positions': sample['obj_positions'],
                'labels': sample['relation'],
                'masks': sample['masks']
            }
        data = tf.data.Dataset.from_tensor_slices(train_files).interleave(tf.data.TFRecordDataset,
                                                                          cycle_length=num_parallel_readers,
                                                                          block_length=batch_size).map(add_mask)
        if cache:
            data.cache()
        data = data.repeat().shuffle(buffer_size=1000).batch(batch_size).prefetch(prefetch_buffer_size)
        return data

    def eval_dataset(self,
                     directory,
                     batch_size,
                     dataset_type='dev',
                     prefetch_buffer_size=10):
        parser, record_mappings = self.maybe_create_tf_record_files(directory=directory)
        eval_files = record_mappings[dataset_type]

        def add_mask(sample):
            sample = parser(sample)
            print(sample.keys())
            return {
                'tokens': sample['tokens'],
                'pos': sample['pos'],
                'ner': sample['ner'],
                'deprel': sample['deprel'],
                'subj_positions': sample['subj_positions'],
                'obj_positions': sample['obj_positions'],
                'labels': sample['relation'],
                'masks': sample['masks'],
                # 'tokens_masked': sample['tokens']
            }

        data = tf.data.Dataset.from_tensor_slices(eval_files).\
            flat_map(tf.data.TFRecordDataset).\
            map(add_mask).\
            batch(batch_size).prefetch(prefetch_buffer_size)
        return data


if __name__ == '__main__':
    from enas_lm.src.vocab import Vocab
    import numpy as np
    directory = '/Volumes/External HDD/dataset/tacred/data/json'
    vocab_file = '/Volumes/External HDD/dataset/tacred/data/vocab/vocab.pkl'
    word_embs = '/Volumes/External HDD/dataset/tacred/data/vocab/embedding.npy'
    vocab = Vocab(vocab_file, load=True)
    word_embs = np.load(word_embs)

    dataloader = DataLoader(dataset_name='tacred', vocab=vocab, do_lower=True)
    train_dataset = dataloader.train_dataset(directory=directory,
                                             batch_size=10)
    eval_dataset = dataloader.eval_dataset(directory=directory,
                                           batch_size=10)

    train_iterator = train_dataset.make_one_shot_iterator()
    eval_iterator = eval_dataset.make_initializable_iterator()

    sess = tf.Session()
    train_iterator_handle = sess.run(train_iterator.string_handle())
    eval_iterator_handle = sess.run(eval_iterator.string_handle())

    input_iterator_handle = tf.placeholder(
        tf.string, shape=[], name='input_iterator_handle')
    input_iterator = tf.data.Iterator.from_string_handle(
        input_iterator_handle,
        output_types={
            'token_ids': tf.int64,
            'labels': tf.int64,
            'masks': tf.int64,
            'pos_ids': tf.int64,
            'ner_ids': tf.int64,
            'subj_positions': tf.int64,
            'obj_positions': tf.int64,
            'deprel': tf.int64,
            # 'tokens_masked': tf.int64
        },
        output_shapes={
            'token_ids': [None, None],
            'labels': [None],
            'masks': [None, None],
            'pos_ids': [None, None],
            'ner_ids': [None, None],
            'subj_positions': [None, None],
            'obj_positions': [None, None],
            'deprel': [None, None],
            # 'tokens_masked': [None, None]
        }
    )
    batch_input = input_iterator.get_next()

    for i in range(2):
        components = sess.run(batch_input, feed_dict={input_iterator_handle: train_iterator_handle})
        print("Training | token ids: {}, labels: {}, masks: {}, pos_ids: {}, ner_ids: {}, subj_positions: {}, obj_positions: {}".format(
            components['token_ids'].shape,
            components['labels'].shape,
            components['masks'].shape,
            components['pos_ids'].shape,
            components['ner_ids'].shape,
            components['subj_positions'].shape,
            components['obj_positions'].shape
        ))
        print('tokens with dropout example: {}'.format((components['token_ids'][2])), sep=', ')
        print('labels: {}'.format(components['labels']), sep=', ')
        # print('token with dropout example: {}'.format((components['tokens_masked'][2])), sep=', ')

    sess.run(eval_iterator.initializer)
    for i in range(2):
        components = sess.run(batch_input, feed_dict={input_iterator_handle: eval_iterator_handle})
        print("Eval | token ids: {}, labels: {}, masks: {}, pos_ids: {}, ner_ids: {}, subj_positions: {}, obj_positions: {}".format(
            components['token_ids'].shape,
            components['labels'].shape,
            components['masks'].shape,
            components['pos_ids'].shape,
            components['ner_ids'].shape,
            components['subj_positions'].shape,
            components['obj_positions'].shape
        ))






