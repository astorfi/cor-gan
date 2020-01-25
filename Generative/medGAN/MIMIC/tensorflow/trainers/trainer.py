from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import random


class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(Trainer, self).__init__(sess, model, data, config,logger)

    def train_epoch_autoencoder(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []

        # Loop through the epoch
        for _ in loop:
            loss= self.train_step_autoencoder(on_real_data=self.config.train_autoencoder_on_real_data)
            losses.append(loss)
            # accs.append(acc)
        loss = np.mean(losses)
        # acc = np.mean(accs)

        # Test per epoch
        lossTest = self.test_step_autoencoder(on_real_data=self.config.train_autoencoder_on_real_data)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss_autoencoder': loss,
            # 'acc': acc,
        }
        summaries_dict_test = {
            'loss_autoencoder': lossTest,
            # 'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.logger.summarize(cur_it, summarizer="test", summaries_dict=summaries_dict_test)
        self.model.save(self.sess)

    def train_epoch(self):
        """
        Training the GAN model.
            * We train the Discriminator & the Generator.
            * We pre-trained the Autoencoder. There is no need to train it again.
        """

        # Defining required parameters.
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses_d = []
        losses_g = []
        losses_ae = []
        accuracy_real_list = []
        accuracy_fake_list = []
        accuracy_total_list = []

        # Loop through the epoch
        for _ in loop:
            loss_g = self.train_step_generator()
            loss_d = self.train_step_discriminator()
            # loss_ae_train = self.train_step_autoencoder()
            total_accuracy, accuracy_real, accuracy_fake, aeloss = self.test_step_discriminator()
            losses_d.append(loss_d)
            losses_g.append(loss_g)
            losses_ae.append(aeloss)
            accuracy_real_list.append(accuracy_real)
            accuracy_fake_list.append(accuracy_fake)
            accuracy_total_list.append(total_accuracy)

        # Calculating the loss and accuracy parameters per epoch by averaging batch-wise values.
        loss_d = np.mean(losses_d)
        loss_g = np.mean(losses_g)
        loss_ae = np.mean(losses_ae)
        accuracy_real = np.mean(accuracy_real_list)
        accuracy_fake = np.mean(accuracy_fake_list)
        accuracy_total = np.mean(accuracy_total_list)

        print("accuracy_fake=",accuracy_fake)
        print("accuracy_real=",accuracy_real)
        print("accuracy_total=",accuracy_total)

        # Writing summaries
        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss_discriminator': loss_d,
            'loss_generator': loss_g,
            # 'acc': acc,
        }
        summaries_dict_test = {
            'accuracy_real': accuracy_real,
            'accuracy_fake': accuracy_fake,
            'accuracy_total': accuracy_total,
            'loss_autoencoder_gan': loss_ae,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.logger.summarize(cur_it, summarizer="test", summaries_dict=summaries_dict_test)
        self.model.save(self.sess)


    ###################
    ### Autoencoder ###
    ###################
    def train_step_autoencoder(self, on_real_data=True):
        batch_ae_real,_ = next(self.data.next_batch_train(self.config.batch_size))
        batch_ae_fake = self.generate_random_noise(feature_size=batch_ae_real.shape[1])
        randomStatus = random.randint(0, 1)
        if on_real_data:
            batch_ae = batch_ae_real
        else:
            if randomStatus:
                batch_ae = batch_ae_fake
            else:
                batch_ae = batch_ae_real
        feed_dict = {self.model.x_ae: batch_ae, self.model.is_training_autoencoder: True}
        _, loss = self.sess.run([self.model.train_step_autoencoder, self.model.aeloss],
                                     feed_dict=feed_dict)
        return loss

    def test_step_autoencoder(self, on_real_data=True):
        batch_ae_real,_ = next(self.data.next_batch_train(self.config.batch_size))
        batch_ae_fake = self.generate_random_noise(feature_size=batch_ae_real.shape[1])
        randomStatus = random.randint(0, 1)
        if on_real_data:
            batch_ae = batch_ae_real
        else:
            if randomStatus:
                batch_ae = batch_ae_fake
            else:
                batch_ae = batch_ae_real
        feed_dict = {self.model.x_ae: batch_ae, self.model.is_training_autoencoder: False}
        loss = self.sess.run(self.model.aeloss,
                                     feed_dict=feed_dict)
        return loss

    #################
    ### Generator ###
    #################

    def train_step_generator(self):
        """
        The GAN generator.
            * Input: Random noize
            * Output: A continuous synthesized data
        :return: The generator loss
        """
        # batch_noise = np.random.uniform(0, 1, size=(self.config.batch_size, self.config.generator_noise_size))
        # generate a sparse random vector similar to the real data
        batch_x, _ = next(self.data.next_batch_train(self.config.batch_size))
        batch_noise = self.generate_random_noise(feature_size=self.config.generator_noise_size)
        feed_dict = {self.model.x_g: batch_noise, self.model.x_real: batch_x,self.model.is_training_generator: True}
        _, loss_g= self.sess.run([self.model.train_step_generator, self.model.genloss],
                                     feed_dict=feed_dict)
        return loss_g

    #####################
    ### Discriminator ###
    #####################
    def train_step_discriminator(self):
        """
        Training the discriminator.
        STEPS:
            1. We create a batch of real data: batch_x
            2. We create a batch of random noise: batch_noise
            3. We pass the "batch_noise" to Generator to synthesize a batch of continuous fake data.
            4. The continuous fake data is fed to the pretrained Autoencoder[the decoder part] to
               decode and generate the associated discrete value: batch_g
            5. The discriminator take "batch_g" and "batch_x" and try to distinguish between them.

        :return:
        """

        # Create data and noise batches
        batch_x, _ = next(self.data.next_batch_train(self.config.batch_size))
        batch_x_label = np.ones((batch_x.shape[0], 1))
        batch_noise = self.generate_random_noise(feature_size=self.config.generator_noise_size)

        # The generated output by the generator and decoded to discrete data by autoencoder
        batch_g_label = np.zeros((batch_x.shape[0], 1))
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
        feed_dict = {self.model.x_fake: batch_g, self.model.x_real: batch_x,
                     self.model.is_training_discriminator: True}
        _, loss_d = self.sess.run([self.model.train_step_discriminator, self.model.disloss],
                                  feed_dict=feed_dict)
        ########################

        return loss_d


    def test_step_discriminator(self):

        # Create data and noise batches
        batch_x, _ = next(self.data.next_batch_test(self.config.batch_size))
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
        batch_g = np.around(np.clip(batch_g, 0, 1)) # Create zero and one
        feed_dict = {self.model.x_fake: batch_g, self.model.x_real: batch_x, self.model.x_ae: batch_x,
                     self.model.is_training_discriminator: False, self.model.is_training_autoencoder: False}
        total_accuracy, accuracy_real, accuracy_fake, aeloss = self.sess.run([self.model.total_accuracy, self.model.correct_prediction_real, self.model.correct_prediction_fake, self.model.aeloss],
                                  feed_dict=feed_dict)
        ########################

        return total_accuracy, accuracy_real, accuracy_fake, aeloss

    def generate_random_noise(self, feature_size):
        return np.random.choice(np.arange(2), size=(self.config.batch_size, feature_size),
                         p=[0.5, 0.5])




