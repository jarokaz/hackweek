

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

FLAGS = flags.FLAGS
flags.DEFINE_integer('steps_per_epoch', 10, 'Steps per training epoch')
flags.DEFINE_integer('eval_steps', 5, 'Evaluation steps')
flags.DEFINE_integer('epochs', 2, 'Nubmer of epochs')
flags.DEFINE_integer('per_replica_batch_size', 32, 'Per replica batch size')
flags.DEFINE_string('training_data_path', 'gs://jk-demos-bucket/data/imdb/train', 'Training data GCS path')
flags.DEFINE_string('testing_data_path', 'gs://jk-demos-bucket/data/imdb/test', 'Testing data GCS path')
flags.DEFINE_float('validation_split', 0.2, 'Validation data split')
flags.DEFINE_string('job_dir', 'gs://jk-demos-bucket/jobs', 'A base GCS path for jobs')
flags.DEFINE_enum('strategy', 'mirrored', ['mirrored', 'multiworker'], 'Distribution strategy')

flags.DEFINE_bool('compile_only', False, 'Compile the pipeline but do not submit a run')
flags.DEFINE_bool('use_cloud_pipelines', False, 'Use AI Platform Pipelines')
flags.DEFINE_bool('use_cloud_executors', False, 'Use AI Platform and Dataflow for executors')


def create_input_pipelines(train_dir, test_dir, val_split, seed, batch_size):
    """Creates input pipelines from Imdb dataset."""
    
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=val_split,
        subset='training',
        seed=seed)

    class_names = raw_train_ds.class_names
    train_ds = raw_train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=val_split,
        subset='validation',
        seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size)

    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


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
        
        train_ds, val_ds, test_ds, class_names = (
            create_input_pipelines(FLAGS.training_data_path, 
                                   FLAGS.testing_data_path, 
                                   FLAGS.validation_split, 
                                   seed, 
                                   global_batch_size)
        )
        
    
    logging.info('Starting training ...')
    
    history = model.fit(x=train_ds,
                        validation_data=val_ds,
                        steps_per_epoch=FLAGS.steps_per_epoch,
                        validation_steps=FLAGS.eval_steps,
                        epochs=FLAGS.epochs)

    saved_model_dir = FLAGS.job_dir
    logging.info('Training completed. Saving the trained model to: {}'.format(saved_model_dir))
    # model.save
    
if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
