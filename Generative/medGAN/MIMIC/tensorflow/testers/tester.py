from base.base_test import BaseTest
from tqdm import tqdm
import numpy as np
import tensorflow as tf


class Tester(BaseTest):
    def __init__(self, sess, model, data, config,logger):
        super(Tester, self).__init__(sess, model, data, config,logger)

    def test_gererated(self):
        """
        """

        # Defining required parameters.
        loop = tqdm(range(self.config.num_iter_per_test))
        losses_ae = []
        accuracy_real_list = []
        accuracy_fake_list = []

        # Loop through the epoch
        for _ in loop:
            accuracy_real, accuracy_fake, aeloss = self.test_step_discriminator()
            losses_ae.append(aeloss)
            accuracy_real_list.append(accuracy_real)
            accuracy_fake_list.append(accuracy_fake)

        # Calculating the loss and accuracy parameters per epoch by averaging batch-wise values.
        loss_ae = np.mean(losses_ae)
        accuracy_real = np.mean(accuracy_real_list)
        accuracy_fake = np.mean(accuracy_fake_list)

        print("accuracy_fake=",accuracy_fake)
        print("accuracy_real=",accuracy_real)

        # Writing summaries
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict_test = {
            'accuracy_real': accuracy_real,
            'accuracy_fake': accuracy_fake,
            'loss_autoencoder_gan': loss_ae,
        }
        # self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.logger.summarize(cur_it, summarizer="test", summaries_dict=summaries_dict_test)

    def test_step_discriminator(self):

        # Create data and noise batches
        batch_x, _ = next(self.data.next_batch_train(self.config.batch_size_test))
        batch_noise = self.generate_random_noise(feature_size=self.config.generator_noise_size)

        # The generated output by the generator
        batch_g = self.sess.run(self.model.decoded_g,
                                feed_dict={self.model.x_g: batch_noise, self.model.is_training_generator: False})

        # #### APPROACH 1 #####
        # # Concatenate both vectors
        # idx = np.random.permutation(2 * batch_x.shape[0])
        # batch_dis = np.concatenate((batch_x, batch_g))[idx]
        # batch_dis_label = np.concatenate((batch_x_label, batch_g_label))[idx]
        #
        #
        # # Feed "batch_noise" to generator
        # feed_dict = {self.model.x_dis: batch_dis, self.model.y_dis: batch_dis_label, self.model.is_training_discriminator: True}
        # _, loss_d = self.sess.run([self.model.train_step_discriminator, self.model.disloss],
        #                         feed_dict=feed_dict)
        # ####################

        #### APPROACH 2 #####
        # Feed "batch_noise" to generator
        batch_g = np.round(batch_g) # Create zero and one
        feed_dict = {self.model.x_fake: batch_g, self.model.x_real: batch_x, self.model.x_ae: batch_x,
                     self.model.is_training_discriminator: False, self.model.is_training_autoencoder: False}
        accuracy_real, accuracy_fake, aeloss = self.sess.run([self.model.correct_prediction_real, self.model.correct_prediction_fake, self.model.aeloss],
                                  feed_dict=feed_dict)
        ########################

        return accuracy_real, accuracy_fake, aeloss

    def generate_random_noise(self, feature_size):
        return np.random.choice(np.arange(2), size=(self.config.batch_size, feature_size),
                         p=[0.8, 0.2])




