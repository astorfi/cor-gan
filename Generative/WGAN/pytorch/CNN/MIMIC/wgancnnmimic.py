import argparse
import os
import numpy as np
import math
import time
import random

import matplotlib.pyplot as plt

import os
from subprocess import call

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

# import matplotlib
# matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()

# experimentName is the current file name without extension
experimentName = os.path.splitext(os.path.basename(__file__))[0]

parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data/PhisioNet/MIMIC/processed/out_binary.matrix'),
                    help="Dataset file")

parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--n_epochs_pretrain", type=int, default=500,
                    help="number of epochs of pretraining the autoencoder")
parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument('--n_iter_D', type=int, default=5, help='number of D iters per each G iter')

# Check the details
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")

parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between samples")
parser.add_argument("--epoch_time_show", type=bool, default=True, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=100, help="number of epops per model save")
parser.add_argument("--minibatch_averaging", type=bool, default=False, help="Minibatch averaging")

parser.add_argument("--training", type=bool, default=False, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/model/'+experimentName),
                    help="Training status")
opt = parser.parse_args()
print(opt)

# Create experiments DIR
if not os.path.exists(opt.expPATH):
    os.system('mkdir {0}'.format(opt.expPATH))

# Random seed for pytorch
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
cudnn.benchmark = True

# Check cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device BUT it is not in use...")

# Activate CUDA
device = torch.device("cuda:0" if opt.cuda else "cpu")

##########################
### Dataset Processing ###
##########################

data = np.load(os.path.expanduser(opt.DATASETPATH), allow_pickle=True)

sampleSize = data.shape[0]
featureSize = data.shape[1]

# Split train-test
indices = np.random.permutation(sampleSize)
training_idx, test_idx = indices[:int(0.8 * sampleSize)], indices[int(0.8 * sampleSize):]
trainData = data[training_idx, :]
testData = data[test_idx, :]

# Trasnform Object array to float
trainData = trainData.astype(np.float32)
testData = testData.astype(np.float32)

# ave synthetic data
np.save(os.path.join(opt.expPATH, "dataTrain.npy"), trainData, allow_pickle=False)
np.save(os.path.join(opt.expPATH, "dataTest.npy"), testData, allow_pickle=False)

class Dataset:
    def __init__(self, data, transform=None):

        # Transform
        self.transform = transform

        # load data here
        self.data = data
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)

        if self.transform:
           pass

        return torch.from_numpy(sample)


# Train data loader
dataset_train_object = Dataset(data=trainData, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
dataloader_train = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                              shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

# Test data loader
dataset_test_object = Dataset(data=testData, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
dataloader_test = DataLoader(dataset_test_object, batch_size=opt.batch_size,
                             shuffle=False, num_workers=1, drop_last=True, sampler=samplerRandom)

# Generate random samples for test
random_samples = next(iter(dataloader_test))
feature_size = random_samples.size()[1]

####################
### Architecture ###
####################
class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        n_channels_base = 4

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=5, stride=2, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=5, stride=2, padding=0,
                      dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=2 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=4 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(16 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16 * n_channels_base, out_channels=32 * n_channels_base, kernel_size=8, stride=1,
                      padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5, stride=1, padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=4, padding=0,
                               dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=7, stride=4,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4 * n_channels_base, out_channels=2 * n_channels_base, kernel_size=7, stride=3,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2 * n_channels_base, out_channels=n_channels_base, kernel_size=7, stride=2,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=n_channels_base, out_channels=1, kernel_size=3, stride=2,
                               padding=0, dilation=1,
                               groups=1, bias=True, padding_mode='zeros'),
            nn.Sigmoid(),
        )
        # self.decoder = nn.Sequential(nn.Linear(128, dataset_train_object.featureSize)
        #                              , nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x.view(-1, 1, x.shape[1]))
        x = self.decoder(x)
        return torch.squeeze(x)

    def decode(self, x):
        x = self.decoder(x)
        return torch.squeeze(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.genDim = 128
        self.linear1 = nn.Linear(opt.latent_dim, self.genDim)
        self.bn1 = nn.BatchNorm1d(self.genDim, eps=0.001, momentum=0.01)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(opt.latent_dim, self.genDim)
        self.bn2 = nn.BatchNorm1d(self.genDim, eps=0.001, momentum=0.01)
        self.activation2 = nn.Tanh()

    def forward(self, x):
        # Layer 1
        residual = x
        temp = self.activation1(self.bn1(self.linear1(x)))
        out1 = temp + residual

        # Layer 2
        residual = out1
        temp = self.activation2(self.bn2(self.linear2(out1)))
        out2 = temp + residual
        return out2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Discriminator's parameters
        self.disDim = 256

        # The minibatch averaging setup
        ma_coef = 1
        if opt.minibatch_averaging:
            ma_coef = ma_coef * 2

        self.model = nn.Sequential(
            nn.Linear(ma_coef * dataset_train_object.featureSize, self.disDim),
            nn.ReLU(True),
            nn.Linear(self.disDim, int(self.disDim)),
            nn.ReLU(True),
            nn.Linear(self.disDim, int(self.disDim)),
            nn.ReLU(True),
            nn.Linear(int(self.disDim), 1)
        )

    def forward(self, x):

        if opt.minibatch_averaging:
            ### minibatch averaging ###
            x_mean = torch.mean(x, 0).repeat(x.shape[0], 1)  # Average over the batch
            x = torch.cat((x, x_mean), 1)  # Concatenation

        # Feeding the model
        output = self.model(x)
        return output


###############
### Lossess ###
###############

def generator_loss(y_fake, y_true):
    """
    Gen loss
    Can be replaced with generator_loss = torch.nn.BCELoss(). Think why?
    """
    epsilon = 1e-12
    return -0.5 * torch.mean(torch.log(y_fake + epsilon))


def autoencoder_loss(x_output, y_target):
    """
    autoencoder_loss
    This implementation is equivalent to the following:
    torch.nn.BCELoss(reduction='sum') / batch_size
    As our matrix is too sparse, first we will take a sum over the features and then do the mean over the batch.
    WARNING: This is NOT equivalent to torch.nn.BCELoss(reduction='mean') as the later on, mean over both features and batches.
    """
    epsilon = 1e-12
    term = y_target * torch.log(x_output + epsilon) + (1. - y_target) * torch.log(1. - x_output + epsilon)
    loss = torch.mean(-torch.sum(term, 1), 0)
    return loss


def discriminator_loss(outputs):
    """
    autoencoder_loss
    Cab be replaced with discriminator_loss = torch.nn.BCELoss(). Think why?
    """
    loss = torch.mean((1 - labels) * outputs) - torch.mean(labels * outputs)
    return loss


#################
### Functions ###
#################

def discriminator_accuracy(predicted, y_true):
    """
    The discriminator accuracy on samples
    :param predicted: The predicted labels
    :param y_true: The gorund truth labels
    :return: Accuracy
    """
    total = y_true.size(0)
    correct = (torch.abs(predicted - y_true) <= 0.5).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy


def sample_transform(sample):
    """
    Transform samples to their nearest integer
    :param sample: Rounded vector.
    :return:
    """
    sample[sample >= 0.5] = 1
    sample[sample < 0.5] = 0
    return sample


def weights_init(m):
    """
    Custom weight initialization.
    :param m: Input argument to extract layer type
    :return: Initialized architecture
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


#############
### Model ###
#############

# Initialize generator and discriminator
generatorModel = Generator()
discriminatorModel = Discriminator()
autoencoderModel = Autoencoder()
autoencoderDecoder = autoencoderModel.decoder

# Define cuda Tensors
Tensor = torch.FloatTensor
one = torch.FloatTensor([1])
mone = one * -1


if torch.cuda.device_count() > 1 and opt.multiplegpu:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  generatorModel = nn.DataParallel(generatorModel, list(range(opt.num_gpu)))
  discriminatorModel = nn.DataParallel(discriminatorModel, list(range(opt.num_gpu)))
  autoencoderModel = nn.DataParallel(autoencoderModel, list(range(opt.num_gpu)))
  autoencoderDecoder = nn.DataParallel(autoencoderDecoder, list(range(opt.num_gpu)))

if opt.cuda:
    """
    model.cuda() will change the model inplace while input.cuda() 
    will not change input inplace and you need to do input = input.cuda()
    ref: https://discuss.pytorch.org/t/when-the-parameters-are-set-on-cuda-the-backpropagation-doesnt-work/35318
    """
    generatorModel.cuda()
    discriminatorModel.cuda()
    autoencoderModel.cuda()
    autoencoderDecoder.cuda()
    one, mone = one.cuda(), mone.cuda()
    Tensor = torch.cuda.FloatTensor

# Weight initialization
generatorModel.apply(weights_init)
discriminatorModel.apply(weights_init)
autoencoderModel.apply(weights_init)

# Optimizers
g_params = [{'params': generatorModel.parameters()},
            {'params': autoencoderDecoder.parameters(), 'lr': 1e-4}]
# g_params = list(generatorModel.parameters()) + list(autoencoderModel.decoder.parameters())
optimizer_G = torch.optim.Adam(g_params, lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)
optimizer_A = torch.optim.Adam(autoencoderModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)

################
### TRAINING ###
################
if opt.training:

    if opt.resume:
        #####################################
        #### Load model and optimizer #######
        #####################################

        # Loading the checkpoint
        checkpoint = torch.load(os.path.join(opt.PATH, "model_epoch_1000.pth"))

        # Load models
        generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
        discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])
        autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
        autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

        # Load optimizers
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])

        # Load losses
        g_loss = checkpoint['g_loss']
        d_loss = checkpoint['d_loss']
        a_loss = checkpoint['a_loss']

        # Load epoch number
        epoch = checkpoint['epoch']

        generatorModel.eval()
        discriminatorModel.eval()
        autoencoderModel.eval()
        autoencoderDecoder.eval()

    for epoch_pre in range(opt.n_epochs_pretrain):
        for i, samples in enumerate(dataloader_train):

            # Configure input
            real_samples = Variable(samples.type(Tensor))

            # Generate a batch of images
            recons_samples = autoencoderModel(real_samples)

            # Loss measures generator's ability to fool the discriminator
            a_loss = autoencoder_loss(recons_samples, real_samples)

            # # Reset gradients (if you uncomment it, it would be a mess. Why?!!!!!!!!!!!!!!!)
            optimizer_A.zero_grad()

            a_loss.backward()
            optimizer_A.step()

            batches_done = epoch_pre * len(dataloader_train) + i
            if batches_done % opt.sample_interval == 0:
                print(
                    "[Epoch %d/%d of pretraining] [Batch %d/%d] [A loss: %.3f]"
                    % (epoch_pre + 1, opt.n_epochs_pretrain, i, len(dataloader_train), a_loss.item())
                    , flush=True)

    gen_iterations = 0
    for epoch in range(opt.n_epochs):
        epoch_start = time.time()
        for i, samples in enumerate(dataloader_train):

            # Adversarial ground truths
            valid = Variable(Tensor(samples.shape[0]).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(samples.shape[0]).fill_(0.0), requires_grad=False)

            # Configure input
            real_samples = Variable(samples.type(Tensor))

            # Sample noise as generator input
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            # -----------------
            #  Train Generator
            # -----------------

            # We’re supposed to clear the gradients each iteration before calling loss.backward() and optimizer.step().
            #
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch
            # accumulates the gradients on subsequent backward passes. This is convenient while training RNNs. So,
            # the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
            #
            # Because of this, when you start your training loop, ideally you should zero out the gradients so
            # that you do the parameter update correctly. Else the gradient would point in some other direction
            # than the intended direction towards the minimum (or maximum, in case of maximization objectives).

            # Since the backward() function accumulates gradients, and you don’t want to mix up gradients between
            # minibatches, you have to zero them out at the start of a new minibatch. This is exactly like how a general
            # (additive) accumulator variable is initialized to 0 in code.

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = False

            # Zero grads
            optimizer_G.zero_grad()

            # Generate a batch of images
            fake_samples = generatorModel(z)

            # uncomment if there is no autoencoder
            fake_samples = torch.squeeze(autoencoderDecoder(fake_samples.unsqueeze(dim=2)))

            # Loss measures generator's ability to fool the discriminator
            errG = torch.mean(discriminatorModel(fake_samples).view(-1))
            errG.backward(one)

            # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4
            optimizer_G.step()
            gen_iterations += 1

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = True

            # train the discriminator n_iter_D times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                n_iter_D = 100
            else:
                n_iter_D = opt.n_iter_D
            j = 0
            while j < n_iter_D:
                j += 1

                # clamp parameters to a cube
                for p in discriminatorModel.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                # reset gradients of discriminator
                optimizer_D.zero_grad()

                errD_real = torch.mean(discriminatorModel(real_samples).view(-1))
                errD_real.backward(one)

                # Measure discriminator's ability to classify real from generated samples
                # The detach() method constructs a new view on a tensor which is declared
                # not to need gradients, i.e., it is to be excluded from further tracking of
                # operations, and therefore the subgraph involving this view is not recorded.
                # Refer to http://www.bnikolic.co.uk/blog/pytorch-detach.html.

                errD_fake = torch.mean(discriminatorModel(fake_samples.detach()).view(-1))
                errD_fake.backward(mone)
                errD = errD_real - errD_fake

                # Optimizer step
                optimizer_D.step()

        with torch.no_grad():

            # Variables
            real_samples_test = next(iter(dataloader_test))
            real_samples_test = Variable(real_samples_test.type(Tensor))
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            # Generator
            fake_samples_test_temp = generatorModel(z)
            fake_samples_test = torch.squeeze(autoencoderDecoder(fake_samples_test_temp.unsqueeze(dim=2)))

            # Discriminator
            # F.sigmoid() is needed as the discriminator outputs are logits without any sigmoid.
            out_real_test = discriminatorModel(real_samples_test).view(-1)
            accuracy_real_test = discriminator_accuracy(F.sigmoid(out_real_test), valid)

            out_fake_test = discriminatorModel(fake_samples_test.detach()).view(-1)
            accuracy_fake_test = discriminator_accuracy(F.sigmoid(out_fake_test), fake)

            # Test autoencoder
            reconst_samples_test = autoencoderModel(real_samples_test)
            a_loss_test = autoencoder_loss(reconst_samples_test, real_samples_test)

        print('TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_D: %.3f Loss_G: %.3f Loss_D_real: %.3f Loss_D_fake %.3f'
              % (epoch + 1, opt.n_epochs, i, len(dataloader_train),
                 errD.item(), errG.item(), errD_real.item(), errD_fake.item()), flush=True)

        print(
            "TEST: [Epoch %d/%d] [Batch %d/%d] [A loss: %.2f] [real accuracy: %.2f] [fake accuracy: %.2f]"
            % (epoch + 1, opt.n_epochs, i, len(dataloader_train),
               a_loss_test.item(), accuracy_real_test,
               accuracy_fake_test)
            , flush=True)

        # End of epoch
        epoch_end = time.time()
        if opt.epoch_time_show:
            print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)

        if (epoch + 1) % opt.epoch_save_model_freq == 0:
            # Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html
            torch.save({
                'epoch': epoch + 1,
                'Generator_state_dict': generatorModel.state_dict(),
                'Discriminator_state_dict': discriminatorModel.state_dict(),
                'Autoencoder_state_dict': autoencoderModel.state_dict(),
                'Autoencoder_Decoder_state_dict': autoencoderDecoder.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'optimizer_A_state_dict': optimizer_A.state_dict(),
            }, os.path.join(opt.expPATH, "model_epoch_%d.pth" % (epoch + 1)))

            # keep only the most recent 10 saved models
            # ls -d -1tr /home/sina/experiments/pytorch/model/* | head -n -10 | xargs -d '\n' rm -f
            call("ls -d -1tr " + opt.expPATH + "/*" + " | head -n -10 | xargs -d '\n' rm -f", shell=True)

if opt.finetuning:

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.PATH, "model_epoch_100.pth"))

    # Setup model
    generatorModel = Generator()
    discriminatorModel = Discriminator()

    if cuda:
        generatorModel.cuda()
        discriminatorModel.cuda()
        discriminator_loss.cuda()

    # Setup optimizers
    optimizer_G = torch.optim.Adam(generatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
    discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])

    # Load optimizers
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    # Load losses
    g_loss = checkpoint['g_loss']
    d_loss = checkpoint['d_loss']

    # Load epoch number
    epoch = checkpoint['epoch']

    generatorModel.eval()
    discriminatorModel.eval()

if opt.generate:

    # Check cuda
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device BUT it is not in use...")

    # Activate CUDA
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    #####################################
    #### Load model and optimizer #######
    #####################################

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.expPATH, "model_epoch_300.pth"))

    # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
    autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
    autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

    # insert weights [required]
    generatorModel.eval()
    autoencoderModel.eval()
    autoencoderDecoder.eval()

    #######################################################
    #### Load real data and generate synthetic data #######
    #######################################################

    # Load real data
    real_samples = dataset_train_object.return_data()
    num_fake_samples = 10000

    # Generate a batch of samples
    gen_samples = np.zeros_like(real_samples, dtype=type(real_samples))
    n_batches = int(num_fake_samples / opt.batch_size)
    for i in range(n_batches):
        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
        z = torch.randn(opt.batch_size, opt.latent_dim, device=device)
        gen_samples_tensor = generatorModel(z)
        gen_samples_decoded = torch.squeeze(autoencoderDecoder(gen_samples_tensor.unsqueeze(dim=2)))
        gen_samples[i * opt.batch_size:(i + 1) * opt.batch_size, :] = gen_samples_decoded.cpu().data.numpy()
        # Check to see if there is any nan
        assert (gen_samples[i, :] != gen_samples[i, :]).any() == False

    gen_samples = np.delete(gen_samples, np.s_[(i + 1) * opt.batch_size:], 0)
    gen_samples[gen_samples >= 0.5] = 1.0
    gen_samples[gen_samples < 0.5] = 0.0

    # Trasnform Object array to float
    gen_samples = gen_samples.astype(np.float32)

    # ave synthetic data
    np.save(os.path.join(opt.expPATH, "synthetic.npy"), gen_samples, allow_pickle=False)

if opt.evaluate:
    # Load synthetic data
    gen_samples = np.load(os.path.join(opt.expPATH, "synthetic.npy"), allow_pickle=False)

    # Load real data
    real_samples = dataset_train_object.return_data()[0:gen_samples.shape[0], :]

    # Dimenstion wise probability
    prob_real = np.mean(real_samples, axis=0)
    prob_syn = np.mean(gen_samples, axis=0)

    p1 = plt.scatter(prob_real, prob_syn, c="b", alpha=0.5, label="WGAN")
    x_max = max(np.max(prob_real), np.max(prob_syn))
    x = np.linspace(0, x_max + 0.1, 1000)
    p2 = plt.plot(x, x, linestyle='-', color='k', label="Ideal")  # solid
    plt.tick_params(labelsize=12)
    plt.legend(loc=2, prop={'size': 15})
    # plt.title('Scatter plot p')
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.show()