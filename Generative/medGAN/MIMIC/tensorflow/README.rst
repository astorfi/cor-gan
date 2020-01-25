Generating Synthetic Discrete Data Using Generative Adversarial Networks
=================================================================================

**Deep learning** models have demonstrated high-quality performance in several areas,
such as image classification and speech processing. However, training a deep learning
model using **privacy-restricted** data brings unique challenges and making research difficult
working with such data. To overcome these challenges, researchers propose to generate and
utilize **realistic synthetic data** that can be used instead of real data. Existing methods
for artificial data generation are limited by being bound to *particular uses cases*.
Furthermore, their *generalizability* to real-world problems is questionable. There is a
need to establish valid synthetic data that overcomes privacy restrictions and functions
as a real-world analog for deep learning.

.. raw:: html

   <div align="center">

.. raw:: html

   <img align="center" hight="400" width="800" src="https://github.com/astorfi/tempGAN/blob/master/_img/syntheticdatastructure.png">

.. raw:: html

   </div>

In this project, we use of **Generative Adversarial Networks** to generate **realistic discrete data**.
GANs are having trouble handling and making **discrete data**. In fact, *GANs are designed to produce
continuous variables*. Despite the impressive performance of GANs regarding continuous data, it remains
challenging to create discrete data with GANs, which restricts its applicability in domains such as **Natural Language Processing**.
In this project, we use **autoencoders** as the mapping models between discrete-continuous data domains.

Table Of Contents
=================

-  `In a Nutshell`_
-  `In Details`_

   -  `Project architecture`_
   -  `Folder structure`_
   -  `Main Components`_

      -  `Models`_
      -  `Trainer`_
      -  `Data Loader`_
      -  `Logger`_
      -  `Configuration`_
      -  `Main`_

-  `Future Work`_
-  `Contributing`_
-  `Acknowledgments`_

.. _In a Nutshell: #in-a-nutshell
.. _In Details: #in-details
.. _Project architecture: #project-architecture
.. _Folder structure: #folder-structure
.. _Main Components: #main-components
.. _Models: #models
.. _Trainer: #trainer
.. _Data Loader: #data-loader
.. _Logger: #logger
.. _Configuration: #configuration
.. _Main: #main
.. _Future Work: #future-work
.. _Contributing: #contributing
.. _Acknowledgments: #acknowledgments


In a Nutshell
=============

In a nutshell here's how to use this code, so **for model utilization** assume
you want to implement ``Model`` so you should do the following:

-  In models folder there is a class named ``Model`` that inherits the
   ``BaseModel`` class from ``base/base_model.py``. This class build the
   architecture of *overall model* with below *high-level structure*:

.. code:: python


    class Model(BaseModel):
      def __init__(self, config):
         super(Model, self).__init__(config)
         self.init_saver()

         self.model_autoencoder = tf.keras.Sequential(model_autoencoder_architecture)
         self.model_generator = tf.keras.Sequential(model_generator_architecture)
         self.model_discriminator = tf.keras.Sequential(model_discriminator_architecture)

         # building the model
         self.build_autoencoder()
         self.build_generator()
         self.build_discriminator()

-  We defined some functions called ``build_model`` to form different elements of the model such as ``autoencoder`` and ``generator``. In ``init_saver`` we defined a tensorflow saver, then will
   call them in the ``initalizer``.

.. code:: python

        def build_autoencoder(self):
           # here we build the tensorflow graph of autoencoder model (or any model) and also define the loss.
           pass

        def init_saver(self):
           # here you initalize the tensorflow saver that will be used in saving the checkpoints.
           self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

-  In trainers folder, in file ``trainer.py``, there is a ``trainer`` class that inherit from "base_train" class

.. code:: python

      class Trainer(BaseTrain):
          def __init__(self, sess, model, data, config,logger):
              super(Trainer, self).__init__(sess, model, data, config,logger)

-  We wrote different functions such as "train_epoch" where we
   write the logic of the training process.

.. code:: python


       def train_epoch(self):
           """
          implement the process of training for an epoch:
          -number of iterations in defined in the config file
          - call the train step
          -add any summaries [optional]
           """
           pass


-  In main file, we create the session and instances of the following
   objects "Model", "Logger", "Data_Generator", "Trainer", and config

.. code:: python

       sess = tf.Session()
       # create instance of the model you want
       model = Model(config)
       # create your data generator
       data = DataGenerator(config)
       # create tensorboard logger
       logger = Logger(sess, config)

-  Pass the all these objects to the trainer object, and start your
   training by calling "trainer.train()"

.. code:: python

       trainer = Trainer(sess, model, data, config, logger)

       # here you train your model after pretraining the autoencoder
       trainer.pretrainAutoencoder()
       trainer.train()

In Details
==========

Project architecture
--------------------

.. raw:: html

   <div align="center">

.. raw:: html

   <img align="center" hight="400" width="800" src="https://github.com/astorfi/tempGAN/blob/master/_img/baselinegan.png">

.. raw:: html

   </div>

Folder structure
----------------

::

   ├──  base
   │   ├─ base_model.py   - this file contains the abstract class of the model.
   │   └── base_train.py   - this file contains the abstract class of the trainer.
   │
   │
   ├── model               - this folder contains any model of the project.
   │   ├─ modelMLP.py
   │   └─ modelConv1D.py
   │      ............
   │
   │
   ├── trainer             - this folder contains trainers of the project.
   │   └── trainer.py
   │
   ├──  mains              - here's the main(s) of the project.
   │    └── example.py     - here's an example of main that is responsible for the whole pipeline.

   │
   ├──  data _loader
   │    └── data_generator.py  - here's the data_generator that is responsible for all data handling.
   │
   └── utils
        ├── logger.py
        └── any_other_utils_you_need

** Run ``main.py`` in the root!**


Main Components
---------------

Models
~~~~~~

--------------

-  .. rubric:: **Base model**
      :name: base-model

   Base model is an abstract class that must be Inherited by any model
   you create, the idea behind this is that there's much shared stuff
   between all models. The base model contains:

   -  **Save** -This function to save a checkpoint to the desk.
   -  **Load** -This function to load a checkpoint from the desk.
   -  **Cur_epoch, Global_step counters** -These variables to keep track
      of the current epoch and global step.
   -  **Init_Saver** An abstract function to initialize the saver used
      for saving and loading the checkpoint, **Note**: override this
      function in the model you want to implement.
   -  **Build_model** Here's an abstract function to define the model,
      **Note**: override this function in the model you want to
      implement.

-  .. rubric:: **Synthetic Data Generator Model**
      :name: the-model

   The implementation of the model contains the following:

   -  The model class which inherits the base_model class
   -  An autoencoder defined by ``def build_discriminator()`` responsible for discrete-continuous data mapping.
   -  The ``GAN`` architecture which contains a ``generator`` and ``discriminator``.
   -  We call the "build_model" and "init_saver" in the initializer.

**NOTE**: For the discriminator, in addition to the ``self.discriminatorNet``, the gradient MUST flow through
the ``self.autoencoderDecodeNet``. Simply, during the training of discriminato, the weights of ``self.autoencoderDecodeNet``
MUST be updated. This is due to the fact that, the transformation between continuous->discrete values MUST be re-trained during
the general training process.

Trainer
~~~~~~~~

--------------

-  .. rubric:: **Base Trainer**
      :name: base-trainer

   Base trainer is an abstract class that just wrap the training
   process.

-  .. rubric:: **Trainer**
      :name: main-trainer

   What is implemented in the auto encoder trainer.

   1. The general trainer class which inherits the ``base_trainer`` class.
   2. ``train_epoch_autoencoder()`` trainer used for pretraining the autoencoder.

        * **NOTE:** We can pretrain autoencoder with ``real`` or ``random`` data. This is inspired
          by the fact that, autoencoder will operate on top of the generator output.
          It makes sense that we train autoencoder using both  ``real`` AND ``random`` data.

   3. ``train_epoch()`` to train the whole architecture after pretraining the autoencoder.

        * **NOTE:** Special GAN training techniques such as ``Minibatch Discrimination`` and ``Feature Matching`` have
          been implemented.

   4. ``generate_random_noise()`` function aims to syntesize random noise as the ``GAN's generato input``.


Data Loader
~~~~~~~~~~~

This class is responsible for all data handling and processing and
provide an easy interface that can be used by the trainer.

Logger
~~~~~~

This class is responsible for the tensorboard summary, in your trainer
create a dictionary of all tensorflow variables you want to summarize
then pass this dictionary to logger.summarize().


Configuration
~~~~~~~~~~~~~

The ``JSON`` format is used as configuration method, so we wrote all
configs we want in ``"configs/config.json"``.

Main
~~~~

Here's where you combine all previous part.

1. Parse the config file.
2. Create a tensorflow session.
3. Create an instance of "Model", "Data_Generator" and "Logger" and
   parse the config to all of them.
4. Create an instance of "Trainer" and pass all previous objects to it.
5. Now you can train your model by calling "Trainer.train()"

Future Work
===========

-  Add Conv1D model.

Contributing
============

Any kind of enhancement or contribution is welcomed.

Acknowledgments
===============

.. Thanks for my colleague `Mo'men Abdelrazek`_ for contributing in this
.. work. and thanks for `Mohamed Zahran`_ for the review. **Thanks for Jtoy
.. for including the repo in**\ `Awesome Tensorflow`_\ **.**
..
.. .. _Mo'men Abdelrazek: https://github.com/moemen95
.. .. _Mohamed Zahran: https://github.com/moh3th1
.. .. _Awesome Tensorflow: https://github.com/jtoy/awesome-tensorflow
