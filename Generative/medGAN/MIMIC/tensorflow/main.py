import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.modelConv1D import Model
from trainers.trainer import Trainer
from testers.tester import Tester
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # create tensorflow session
    sess = tf.Session()

    # create your data generator
    data = DataGenerator(config)

    # create an instance of the model you want
    model = Model(config)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, data, config, logger)
    tester = Tester(sess, model, data, config, logger)

    if config.fineTuning:
        #load model if exists
        model.load(sess)
        trainer.train()

    else:

        if config.training:
            # here you train your model
            trainer.pretrainAutoencoder()
            trainer.train()
        else:
            model.load(sess)
            tester.test()



if __name__ == '__main__':
    main()
