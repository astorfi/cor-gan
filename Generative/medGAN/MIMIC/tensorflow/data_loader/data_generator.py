import numpy as np
import os
import tensorflow as tf


class DataGenerator:
    def __init__(self, config):
        self.config = config

        # load data here
        self.input = np.load(os.path.expanduser(config.data_file),allow_pickle=True)
        self.sampleSize = self.input.shape[0]
        self.featureSize = self.input.shape[1]

        # Split train-test
        indices = np.random.permutation(self.sampleSize)
        training_idx, test_idx = indices[:int(0.8*self.sampleSize)], indices[int(0.8*self.sampleSize):]
        self.train, self.test = self.input[training_idx, :], self.input[test_idx, :]
        self.trainSize = self.train.shape[0]
        self.testSize = self.test.shape[0]


        ### API ###
        TRAIN_BUF = self.train.shape[0]
        BATCH_SIZE = config.batch_size
        TEST_BUF = self.testSize
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(self.test).shuffle(TEST_BUF).batch(BATCH_SIZE)

    def next_batch_train(self, batch_size):
        idx = np.random.choice(self.trainSize, batch_size)
        yield self.train[idx], None

    def next_batch_test(self, batch_size):
        idx = np.random.choice(self.testSize, batch_size)
        yield self.test[idx], None
