from base.base_model import BaseModel
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class MD(layers.Layer):
    """
    Minibatch Discrimination Layer
    """
    def __init__(self, units=128, num_kernels=5, kernel_dim=3, minibatch_discrimination=True):
        super(MD, self).__init__()
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
        self.units = units
        self.minibatch_discrimination = minibatch_discrimination


    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[1].value, self.num_kernels, self.kernel_dim),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='MD')

    def call(self, inputs):

        if self.minibatch_discrimination:

            # For M matrix of size (?,num_kernels,kernel_dim)
            M = tf.tensordot(inputs, self.w, axes=[[1], [0]], name=None)

            # L1-norm of differences, row-wise(axis=[2])
            M_expand = tf.expand_dims(M, 3) - \
                       tf.expand_dims(tf.transpose(M, [1, 2, 0]), 0)
            c_b = tf.exp(-tf.reduce_sum(tf.abs(M_expand), axis=[2]))
            auxiliary_features = tf.reduce_sum(c_b, 2)
            return tf.concat([inputs, auxiliary_features], 1)
        else:
            return inputs


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.init_saver()

        # input place holder for generator
        self.x_g = tf.placeholder(tf.float32, shape=[None] + [self.config.generator_noise_size])

        # Input place holder for discriminator
        self.x_real = tf.placeholder(tf.float32, shape=[None] + [self.config.state_size])
        self.x_fake = tf.placeholder(tf.float32, shape=[None] + [self.config.state_size])

        # Input place holders for autoencoder.
        self.x_ae = tf.placeholder(tf.float32, shape=[None] + [self.config.state_size])

        ####################
        ### Autoencoder ####
        ####################

        self.autoencoderEncodeNet = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.config.state_size,)),
                tf.keras.layers.Dense(128, activation=tf.nn.tanh),
            ]
        )

        self.autoencoderDecodeNet = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation=tf.nn.tanh),
                tf.keras.layers.Dense(1071),
            ]
        )

        ##################
        ### Generator ####
        ##################
        self.generatorNet = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dense(128, activation=None),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dense(128, activation=tf.nn.tanh),
            ]
        )

        #######################
        #### Discriminator ####
        #######################
        self.discriminatorNet = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.config.state_size,), name='disInputLayer'),
                tf.keras.layers.Dense(256, activation=None),
                tf.keras.layers.LeakyReLU(0.2),
                MD(units=256, num_kernels=5, kernel_dim=3,
                   minibatch_discrimination=self.config.minibatch_discrimination),
                tf.keras.layers.Dense(128, activation=None),
                tf.keras.layers.LeakyReLU(0.2,name='intermediate'),
            ]
        )

        self.discriminatorNetLaseDense = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1, activation=None),
            ]
        )


        # building the model
        self.build_autoencoder()
        self.build_generator()
        self.build_discriminator()

    # def get_discriminator_intermediate(self,x_input):
    #     from tensorflow.keras import backend as K
    #
    #     # with a Sequential model
    #     get_layer_output = K.function([self.discriminatorNet.input],
    #                                       [self.discriminatorNet.get_layer('intermediate').output])
    #     layer_output = get_layer_output([x_input])[0]
    #
    #     return layer_output

    ###############################
    #### Discriminator Methods ####
    ###############################

    # def minibatch(self, input, num_kernels=5, kernel_dim=3):
    #     x = linear(input, num_kernels * kernel_dim)
    #     activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    #     diffs = tf.expand_dims(activation, 3) - \
    #             tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    #     abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    #     minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    #     return tf.concat(1, [input, minibatch_features])

    def discriminatorForward(self, x_dis):
        """
        This function operates as the forward pass of the discriminator
        :param x: The input of the discriminator
        :return: The discriminator output
        """
        # Generator forward pass
        interfeatures = self.discriminatorNet(x_dis)
        logits = self.discriminatorNetLaseDense(interfeatures)

        return logits, interfeatures

    def build_discriminator(self):
        self.is_training_discriminator = tf.placeholder(tf.bool)

        # Discriminator forward pass
        #####################
        self.output_disc_input_real, interfeatures = self.discriminatorForward(self.x_real)
        self.output_disc_input_fake, interfeatures = self.discriminatorForward(self.x_fake)
        #####################

        # inside the function "tf.nn.sigmoid_cross_entropy_with_logits", there is a f(logits) in which f is sigmoid.
        # Tensorflow wording: For brevity, let x = logits, z = labels. The logistic loss is
        #   z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        # = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        # = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        # = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        # = (1 - z) * x + log(1 + exp(-x))
        # = x - x * z + log(1 + exp(-x))

        # with tf.variable_scope('Discriminator') as scope:
        #######################################
        ### Approach 1 for loss calculation ###
        #######################################

        # [1,1,...,1] with real samples
        real_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(self.output_disc_input_real),
                                                    logits=self.output_disc_input_real))

        # [0,0,...,0] with generated samples since they are fake
        generated_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.zeros_like(self.output_disc_input_fake),
            logits=self.output_disc_input_fake))

        # Total generator loss
        self.disloss = real_loss + generated_loss

        # #######################################
        # ### Approach 2 for loss calculation ###
        # #######################################
        # # Calculate loss by combining both fake and real logits
        # vec_real = tf.concat([self.output_disc_input_real, tf.ones_like(self.output_disc_input_real)], axis=1)
        # vec_fake = tf.concat([self.output_disc_input_fake, tf.zeros_like(self.output_disc_input_fake)], axis=1)
        # vec = tf.concat([vec_real, vec_fake], axis=0)
        #
        # ##### Random selection ###
        # # There is no gradiaent computation for tf.random.shuffle(vec, seed=None, name=None). So we should either use
        # # tf.gather or tf.dynamic_partition. The later case provides more memory saving.
        # vec = tf.gather(vec, tf.random.shuffle(tf.range(tf.shape(vec)[0])))
        # # vec = tf.dynamic_partition(vec, partitions=tf.random.shuffle(tf.range(tf.shape(vec)[0])), num_partitions=1)[0]
        # labels= vec[:, 1:]
        # logits= vec[:, 0:1]
        # self.disloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     _sentinel=None,
        #     labels=labels,
        #     logits=logits,
        #     name=None
        # ))

        #################################
        #################################

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # By default, the optimizer will use all of the variables in tf.trainable_variables().
            # But this is NOT the desired action as here we only want to train discriminator.
            # So we define var_list=self.discriminatorNet.trainable_variables.
            optimizer_discriminator = tf.train.AdamOptimizer(self.config.learning_rate)
            # self.train_step_discriminator = optimizer.minimize(self.disloss, global_step=self.global_step_tensor,
            #                                                    var_list=self.discriminatorNet.trainable_variables)

            gradients_of_discriminator = optimizer_discriminator.compute_gradients(self.disloss,
                                                                                   self.discriminatorNet.trainable_variables+self.discriminatorNetLaseDense.trainable_variables)
            self.train_step_discriminator = optimizer_discriminator.apply_gradients(gradients_of_discriminator,
                                                                                    global_step=self.global_step_tensor)

        # calculate the accuract of each class
        self.correct_prediction_fake = tf.reduce_mean(
            tf.cast(tf.less(tf.nn.sigmoid(self.output_disc_input_fake), 0.5), tf.float32))
        self.correct_prediction_real = tf.reduce_mean(
            tf.cast(tf.greater(tf.nn.sigmoid(self.output_disc_input_real), 0.5), tf.float32))
        self.total_accuracy = tf.math.divide(tf.math.add(self.correct_prediction_fake, self.correct_prediction_real), 2)

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

        generated = self.generatorForward(self.x_g)

        # Decoding continuously generated values to their equivalent discrete values using the autoencoder.
        self.decoded_g, recounstructed_g = self.autoencoderDecode(generated)

        # Labels generated by discriminator
        output_disc_input_fake_for_generator_loss, interfeatures  = self.discriminatorForward(self.decoded_g)
        output_disc_input_real_for_generator_loss, interfeatures = self.discriminatorForward(self.x_real)

        # Gen loss
        if self.config.feature_matching:

            logits, intermediate_output_input_real = self.discriminatorForward(self.x_real)
            logits, intermediate_output_input_generator_output = self.discriminatorForward(self.decoded_g)

            self.genloss = tf.losses.mean_squared_error(
                labels=intermediate_output_input_real,
                predictions=intermediate_output_input_generator_output,
                reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE)
        # When optimizing G, we want the quantity D(X′) to be maximized (successfully fooling D).
        # The value function for G is: log(D(G(z)))
        else:
            self.genloss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.ones_like(output_disc_input_fake_for_generator_loss),
                logits=output_disc_input_fake_for_generator_loss))

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
                                                                           var_list= self.autoencoderDecodeNet.trainable_variables + self.generatorNet.trainable_variables)
            self.train_step_generator = generator_optimizer.apply_gradients(gradients_of_generator,
                                                                            global_step=self.global_step_tensor)

            # ### Approach 2 ###
            # var_list = self.generatorNet.trainable_variables
            # gradients_of_generator = tf.gradients(self.genloss, var_list=var_list)
            # self.train_step_generator = generator_optimizer.apply_gradients(zip(gradients_of_generator, var_list),global_step=self.global_step_tensor)

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
        encoded = self.autoencoderEncodeNet(x)
        decoded, recounstructed = self.autoencoderDecode(encoded)

        return decoded, recounstructed

    def compute_loss_ae(self, x):
        """
        :param x: Input to the autoencoder
        :return: Autoencoder loss

        * Attention:
             recounstructed: output of the autoencoder WITHOUT sigmoid
             decoded: output of the autoencoder WITH sigmoid
             The reason that we use "recounstructed" instead of "decoded" in tf.nn.sigmoid_cross_entropy_with_logits,
             is the definition of the tf.nn.sigmoid_cross_entropy_with_logits funtion.
        """
        decoded, recounstructed = self.autoencoderForward(x)
        labels = self.x
        logits = recounstructed
        aeloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            _sentinel=None,
            labels=labels,
            logits=logits,
            name=None
        ))
        return aeloss

    def build_autoencoder(self):
        self.is_training_autoencoder = tf.placeholder(tf.bool)

        decoded, recounstructed = self.autoencoderForward(self.x_ae)

        # Shape match
        assert self.x_ae.get_shape()[1:] == recounstructed.get_shape()[1:]

        with tf.name_scope("loss_autoencoder"):
            labels = self.x_ae
            logits = recounstructed
            self.aeloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                _sentinel=None,
                labels=labels,
                logits=logits,
                name=None
            ))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                learning_rate_autoencoder = tf.compat.v1.train.exponential_decay(self.config.learning_rate_autoencoder,
                                                                     global_step=self.global_step_tensor,
                                                                     decay_steps=50000, decay_rate=0.96, staircase=True)

                self.train_step_autoencoder = tf.train.AdamOptimizer(learning_rate_autoencoder).minimize(
                    self.aeloss,
                    global_step=self.global_step_tensor)

    #########################
    #### General Methods ####
    #########################

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
