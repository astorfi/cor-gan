from base.base_model import BaseModel
import tensorflow as tf
import numpy as np


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.init_saver()
        self.conv = False

        # Input place holder for discriminator
        self.x_real = tf.placeholder(tf.float32, shape=[None] + [self.config.state_size])
        # self.x_real = tf.expand_dims(tf.expand_dims(self.x_real, axis=2), axis=3)

        self.x_fake = tf.placeholder(tf.float32, shape=[None] + [self.config.state_size])
        # self.x_fake = tf.expand_dims(tf.expand_dims(self.x_fake, axis=2), axis=3)

        # input place holder
        self.x_g = tf.placeholder(tf.float32, shape=[None] + [self.config.generator_noise_size])
        # self.x_g = tf.expand_dims(tf.expand_dims(self.x_g, axis=2), axis=3)

        # Input place holders for autoencoder.
        self.x_ae = tf.placeholder(tf.float32, shape=[None] + [self.config.state_size])
        # self.x_ae = tf.expand_dims(tf.expand_dims(self.x_ae, axis=2), axis=3)

        ####################
        ### Autoencoder ####
        ####################

        self.autoencoderEncodeNet = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.config.state_size, 1, 1)),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(7,1), strides=(2,1), data_format="channels_last", activation=tf.nn.tanh),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(7,1), strides=(2,1), data_format="channels_last", activation=tf.nn.tanh)
            ]
        )

        self.autoencoderDecodeNet = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(7, 1), data_format="channels_last", strides=(2, 1), activation=tf.nn.tanh),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(7, 1), data_format="channels_last", strides=(2, 1), activation=None),
            ]
        )

        self.generatorNet = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(4, 1), data_format="channels_last", strides=(1, 1), activation=tf.nn.relu),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(4, 1), data_format="channels_last", strides=(2, 1), activation=tf.nn.tanh),
            ]
        )

        self.discriminatorNet = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.config.state_size, 1, 1)),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(7,1), strides=(2,1), data_format="channels_last", activation=tf.nn.tanh),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(7,1), strides=(2,1), data_format="channels_last", activation=tf.nn.tanh),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(264, 1), strides=(1, 1), data_format="channels_last", activation=None)
            ]
        )


        # building the autoencoder model
        self.build_autoencoder()
        self.build_generator()
        self.build_discriminator()

    ###############################
    #### Discriminator Methods ####
    ###############################

    def expand_dim(self,var):
        return tf.expand_dims(tf.expand_dims(var, axis=2), axis=3)


    def discriminatorForward(self, x_dis):
        """
        This function operates as the forward pass of the discriminator
        :param x: The input of the discriminator
        :return: The discriminator output
        """
        # Generator forward pass
        logits = self.discriminatorNet(x_dis)
        logits = tf.squeeze(logits, axis=[2, 3])

        return logits

    def build_discriminator(self):
        self.is_training_discriminator = tf.placeholder(tf.bool)


        # self.y_dis = tf.placeholder(tf.float32, shape=[None, 1])

        # # Generator forward pass
        x_real = self.expand_dim(self.x_real)
        x_fake = self.expand_dim(self.x_fake)
        #####################

        self.output_disc_input_real = self.discriminatorForward(x_real)
        self.output_disc_input_fake = self.discriminatorForward(x_fake)
        #####################

        # inside the function "tf.nn.sigmoid_cross_entropy_with_logits", there is a f(logits) in which f is sigmoid.
        # Tensorflow wording: For brevity, let x = logits, z = labels. The logistic loss is
        #   z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        # = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        # = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        # = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        # = (1 - z) * x + log(1 + exp(-x))
        # = x - x * z + log(1 + exp(-x))
        # [1,1,...,1] with real samples
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.output_disc_input_real),
                                                    logits=self.output_disc_input_real)

        # [0,0,...,0] with generated samples since they are fake
        generated_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.zeros_like(self.output_disc_input_fake),
            logits=self.output_disc_input_fake)

        # Total generator loss
        self.disloss = real_loss + generated_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # By default, the optimizer will use all of the variables in tf.trainable_variables().
            # But this is NOT the desired action as here we only want to train discriminator.
            # So we define var_list=self.discriminatorNet.trainable_variables.
            optimizer_discriminator = tf.train.AdamOptimizer(self.config.learning_rate)
            # self.train_step_discriminator = optimizer.minimize(self.disloss, global_step=self.global_step_tensor,
            #                                                    var_list=self.discriminatorNet.trainable_variables)

            gradients_of_discriminator = optimizer_discriminator.compute_gradients(self.disloss,
                                                                                   self.discriminatorNet.trainable_variables)
            self.train_step_discriminator = optimizer_discriminator.apply_gradients(gradients_of_discriminator,
                                                                                    global_step=self.global_step_tensor)

        # calculate the accuract of each class
        self.correct_prediction_fake = tf.reduce_mean(
            tf.cast(tf.less(tf.nn.sigmoid(self.output_disc_input_fake), 0.5), tf.float32))
        self.correct_prediction_real = tf.reduce_mean(
            tf.cast(tf.greater(tf.nn.sigmoid(self.output_disc_input_real), 0.5), tf.float32))


    ###########################
    #### Generator Methods ####
    ###########################

    def generatorEncode(self, x_g):
        """
        This function operates as the forward pass of the generator
        :param x: The input noise
        :return: The generated continuous value
        """
        generated = self.generatorNet(x_g)
        return generated

    def generatorForward(self, x_g):
        """
        This function operates as the forward pass of the generator
        :param x: The input noise
        :return: The generated continuous value
        """
        generated = self.generatorEncode(x_g)

        return generated

    def build_generator(self):

        self.is_training_generator = tf.placeholder(tf.bool)

        # Generator forward pass
        x_g = self.expand_dim(self.x_g)
        generated = self.generatorForward(x_g)

        # Decoding continuously generated values to their equivalent discrete values using the autoencoder.
        self.decoded_g, recounstructed_g = self.autoencoderDecode(generated)
        x_real = self.expand_dim(self.x_real)

        # Shape match
        assert x_real.get_shape()[1:] == self.decoded_g.get_shape()[1:]

        # Labels generated by discriminator
        output_disc_input_fake_for_generator_loss = self.discriminatorForward(self.decoded_g)
        output_disc_input_real_for_generator_loss = self.discriminatorForward(x_real)

        # Gen loss
        if self.config.feature_matching:
            self.genloss = tf.losses.mean_squared_error(
                labels=output_disc_input_real_for_generator_loss,
                predictions=output_disc_input_fake_for_generator_loss,
                reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        # When optimizing G, we want the quantity D(X′) to be maximized (successfully fooling D).
        # The value function for G is: log(D(G(z)))
        else:
            self.genloss = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.ones_like(output_disc_input_fake_for_generator_loss),
                logits=output_disc_input_fake_for_generator_loss)

        generator_optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # ATTENTION: Chose the desired variables only
            # By default, the optimizer will use all of the variables in tf.trainable_variables().
            # But this is NOT the desired action as here we only want to train generator.
            # So we define var_list=self.generatorNet.trainable_variables.

            # There are two approaches for gradient computation: (1) optimizer.compute_gradients and (2) tf.gradients.
            ### Approach 1 ###
            gradients_of_generator = generator_optimizer.compute_gradients(self.genloss,
                                                                           var_list=self.autoencoderDecodeNet.trainable_variables + self.generatorNet.trainable_variables)
            self.train_step_generator = generator_optimizer.apply_gradients(gradients_of_generator,
                                                                            global_step=self.global_step_tensor)

    #############################
    #### Autoencoder Methods ####
    #############################

    def autoencoderEncode(self, x):
        encoded = self.autoencoderEncodeNet(x)
        return encoded

    def autoencoderDecode(self, x):

        # Autoencoder output
        recounstructed = self.autoencoderDecodeNet(x)

        # We use sigmoid function for decoding binary variables [decoding: transforming the values to the continuous [0,1] range].
        decoded = tf.nn.sigmoid(recounstructed)

        return decoded, recounstructed

    def autoencoderForward(self, x):
        # x = tf.expand_dims(tf.expand_dims(x, axis=2), axis=3)
        encoded = self.autoencoderEncodeNet(x)
        decoded, recounstructed = self.autoencoderDecode(encoded)

        # Remove auxiliary dimension
        recounstructed = tf.squeeze(recounstructed, axis=[2, 3])
        decoded = tf.squeeze(decoded, axis=[2, 3])

        return decoded, recounstructed

    # def compute_loss_ae(self, x):
    #     """
    #     :param x: Input to the autoencoder
    #     :return: Autoencoder loss
    #
    #     * Attention:
    #          recounstructed: output of the autoencoder WITHOUT sigmoid
    #          decoded: output of the autoencoder WITH sigmoid
    #          The reason that we use "recounstructed" instead of "decoded" in tf.nn.sigmoid_cross_entropy_with_logits,
    #          is the definition of the tf.nn.sigmoid_cross_entropy_with_logits funtion.
    #     """
    #     decoded, recounstructed = self.autoencoderForward(x)
    #     labels = self.x
    #     logits = recounstructed
    #     aeloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #         _sentinel=None,
    #         labels=labels,
    #         logits=logits,
    #         name=None
    #     ))
    #     return aeloss

    def build_autoencoder(self):

        self.is_training_autoencoder = tf.placeholder(tf.bool)

        x_ae = self.expand_dim(self.x_ae)
        decoded, recounstructed = self.autoencoderForward(x_ae)

        with tf.name_scope("loss_autoencoder"):
            labels = self.x_ae
            logits = recounstructed

            # Shape match
            assert labels.get_shape()[1:] == recounstructed.get_shape()[1:]

            self.aeloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                _sentinel=None,
                labels=labels,
                logits=logits,
                name=None
            ))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step_autoencoder = tf.train.AdamOptimizer(self.config.learning_rate_autoencoder).minimize(
                    self.aeloss,
                    global_step=self.global_step_tensor)

    #########################
    #### General Methods ####
    #########################

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
