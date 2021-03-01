

# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from absl import app
from absl import flags
from absl import logging
from official.nlp import optimization 


TFHUB_HANDLE_ENCODER = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
TFHUB_HANDLE_PREPROCESS = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
LOCAL_TB_FOLDER = '/tmp/logs'

FLAGS = flags.FLAGS
flags.DEFINE_integer('steps_per_epoch', 10, 'Steps per training epoch')
flags.DEFINE_integer('eval_steps', 5, 'Evaluation steps')
flags.DEFINE_integer('epochs', 2, 'Nubmer of epochs')
flags.DEFINE_integer('per_replica_batch_size', 32, 'Per replica batch size')
flags.DEFINE_string('training_data_path', 'gs://jk-demos-bucket/tfrecords/train', 'Training data GCS path')
flags.DEFINE_string('validation_data_path', 'gs://jk-demos-bucket/tfrecords/valid', 'Validation data GCS path')
flags.DEFINE_string('testing_data_path', 'gs://jk-demos-bucket/data/imdb/test', 'Testing data GCS path')

flags.DEFINE_string('job_dir', 'gs://jk-demos-bucket/jobs', 'A base GCS path for jobs')
flags.DEFINE_enum('strategy', 'mirrored', ['mirrored', 'multiworker'], 'Distribution strategy')

flags.DEFINE_bool('is_chief', True, 'Set on a chief worker in multi-worker setup')


def create_unbatched_dataset(tfrecords_folder):
    """Creates an unbatched dataset in the format required by the 
       sentiment analysis model from the folder with TFrecords files."""
    
    feature_description = {
        'text_fragment': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    def _parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        return parsed_example['text_fragment'], parsed_example['label']
  
    file_paths = [f'{tfrecords_folder}/{file_path}' for file_path in tf.io.gfile.listdir(tfrecords_folder)]
    dataset = tf.data.TFRecordDataset(file_paths)
    dataset = dataset.map(_parse_function)
    
    return dataset


def configure_dataset_for_performance(ds):
    """
    Optimizes the performance of a dataset.
    """
    
    ds = ds.cache()
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def create_input_pipelines(train_dir, valid_dir, test_dir, batch_size):
    """Creates input pipelines from Imdb dataset."""
    
    train_ds = create_unbatched_dataset(train_dir)
    train_ds = train_ds.batch(batch_size)
    train_ds = configure_dataset_for_performance(train_ds)
    
    valid_ds = create_unbatched_dataset(valid_dir)
    valid_ds = valid_ds.batch(batch_size)
    valid_ds = configure_dataset_for_performance(valid_ds)
    
    test_ds = create_unbatched_dataset(test_dir)
    test_ds = test_ds.batch(batch_size)
    test_ds = configure_dataset_for_performance(test_ds)

    return train_ds, valid_ds, test_ds


def build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder):
    """Builds a simple binary classification model with BERT trunk."""
    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    
    return tf.keras.Model(text_input, net)


def copy_tensorboard_logs(local_path: str, gcs_path: str):
    """Copies Tensorboard logs from a local dir to a GCS location.
    
    After training, batch copy Tensorboard logs locally to a GCS location. This can result
    in faster pipeline runtimes over streaming logs per batch to GCS that can get bottlenecked
    when streaming large volumes.
    
    Args:
      local_path: local filesystem directory uri.
      gcs_path: cloud filesystem directory uri.
    Returns:
      None.
    """
    pattern = '{}/*/events.out.tfevents.*'.format(local_path)
    local_files = tf.io.gfile.glob(pattern)
    gcs_log_files = [local_file.replace(local_path, gcs_path) for local_file in local_files]
    for local_file, gcs_file in zip(local_files, gcs_log_files):
        tf.io.gfile.copy(local_file, gcs_file)


def main(argv):
    del argv
    
    logging.info('Setting up training.')
    logging.info('   epochs: {}'.format(FLAGS.epochs))
    logging.info('   steps_per_epoch: {}'.format(FLAGS.steps_per_epoch))
    logging.info('   eval_steps: {}'.format(FLAGS.eval_steps))
    logging.info('   strategy: {}'.format(FLAGS.strategy))
    
    if FLAGS.strategy == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    global_batch_size = (strategy.num_replicas_in_sync *
                         FLAGS.per_replica_batch_size)
    
    
    num_train_steps = FLAGS.steps_per_epoch * FLAGS.epochs
    num_warmup_steps = int(0.1*num_train_steps)
    init_lr = 3e-5
    seed = 42
    
    with strategy.scope():
        model = build_classifier_model(TFHUB_HANDLE_PREPROCESS, TFHUB_HANDLE_ENCODER)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = tf.metrics.BinaryAccuracy()
        optimizer = optimization.create_optimizer(
            init_lr=init_lr,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            optimizer_type='adamw')

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)
        
        train_ds, valid_ds, test_ds = create_input_pipelines(
            FLAGS.training_data_path,
            FLAGS.validation_data_path,
            FLAGS.testing_data_path,
            global_batch_size)
        

        
    # Configure BackupAndRestore callback
    backup_dir = '{}/backupandrestore'.format(FLAGS.job_dir)
    callbacks = [tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=backup_dir)]
    
    # Configure TensorBoard callback on Chief
    if FLAGS.is_chief:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=LOCAL_TB_FOLDER, update_freq='batch'))
    
    logging.info('Starting training ...')
    
    history = model.fit(x=train_ds,
                        validation_data=valid_ds,
                        steps_per_epoch=FLAGS.steps_per_epoch,
                        validation_steps=FLAGS.eval_steps,
                        epochs=FLAGS.epochs,
                        callbacks=callbacks)

    saved_model_dir = '{}/saved_model'.format(FLAGS.job_dir)
    logging.info('Training completed. Saving the trained model to: {}'.format(saved_model_dir))
    model.save(saved_model_dir)
    
    # Copy tensorboard logs to GCS
    if FLAGS.is_chief:
        tb_logs = '{}/tb_logs'.format(FLAGS.job_dir)
        logging.info('Copying TensorBoard logs to: {}'.format(tb_logs))
        copy_tensorboard_logs(LOCAL_TB_FOLDER, tb_logs)
        
    
if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
