from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--epoch_save_model_freq', type=int, default=10, metavar='N',
                    help='how many epochs per saving the model')

parser.add_argument('--compress_dim', type=int, default=128, metavar='N',
                    help='compressed dimension')

parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data/PhisioNet/MIMIC/processed/out_binary.matrix'),
                    help="Dataset file")
parser.add_argument("--MODELPATH", type=str,
                    default=os.path.expanduser('~/experiments/pytorch/model/VAE'),
                    help="Model Path")

parser.add_argument('--train', type=bool, default=False, metavar='T',
                    help='training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


# Create the experiments path
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create the model PATH
ensure_dir(args.MODELPATH)

########## Dataset Processing ###########

class Dataset:
    def __init__(self, data_file, train=None, transform=None):

        # Transform
        self.transform = transform
        self.train = train

        # load data here
        self.input = np.load(os.path.expanduser(data_file), allow_pickle=True)
        self.sampleSize = self.input.shape[0]
        self.featureSize = self.input.shape[1]

        # Split train-test
        indices = np.random.permutation(self.sampleSize)
        training_idx, test_idx = indices[:int(0.9 * self.sampleSize)], indices[int(0.9 * self.sampleSize):]
        if self.train == True:
            self.data = self.input[training_idx, :]
        else:
            self.data = self.input[test_idx, :]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)

        # if self.transform:
        #     sample = (sample - 0.5) / 0.5

        return torch.from_numpy(sample)


#### Train data loader ####
dataset_train_object = Dataset(data_file=args.DATASETPATH, train=True, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
train_loader = DataLoader(dataset_train_object, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, drop_last=True, sampler=samplerRandom)

### Test data loader ####

dataset_test_object = Dataset(data_file=args.DATASETPATH, train=False, transform=False)
samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
test_loader = DataLoader(dataset_test_object, batch_size=args.batch_size,
                             shuffle=False, num_workers=1, drop_last=True, sampler=samplerRandom)

random_samples = next(iter(test_loader))
feature_size = random_samples.size()[1]

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(1071, 256)
        self.fc21 = nn.Linear(256, args.compress_dim)
        self.fc22 = nn.Linear(256, args.compress_dim)
        self.fc3 = nn.Linear(args.compress_dim, 256)
        self.fc4 = nn.Linear(256, 1071)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1071))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1071), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    if epoch % args.epoch_save_model_freq == 0:
        # Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save({
            'epoch': epoch,
            'Generator_state_dict': model.state_dict(),
            'optimizer_G_state_dict': optimizer.state_dict(),
            'loss': loss.item() / len(data),
        }, os.path.join(args.MODELPATH, "model_epoch_%d.pth" % epoch))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def generate():
    #####################################
    #### Load model and optimizer #######
    #####################################

    # random seed
    np.random.seed(1234)

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(args.MODELPATH, "model_epoch_1000.pth"))

    # Setup model
    model = VAE().to(device)

    # in order to not to have a mismatch, if the training is done using multiple GPU,
    # same should be done while testing.
    # if cuda and opt.multiplegpu:
    #     ngpu = 2
    #     generatorModel = nn.DataParallel(model, list(range(ngpu)))

    # Load models
    model.load_state_dict(checkpoint['Generator_state_dict'])

    # insert weights [required]
    model.eval()

    #######################################################
    #### Load real data and generate synthetic data #######
    #######################################################

    # Load real data
    real_samples = dataset_train_object.return_data()
    num_fake_samples = 10000

    # Generate a batch of samples
    gen_samples = np.zeros_like(real_samples, dtype=type(real_samples))
    n_batches = int(num_fake_samples / args.batch_size)
    for i in range(n_batches):
        random_input = torch.randn(args.batch_size, args.compress_dim).to(device)
        sample = model.decode(random_input).cpu().data.numpy()
        gen_samples[i * args.batch_size:(i + 1) * args.batch_size, :] = sample
        # Check to see if there is any nan
        assert (gen_samples[i, :] != gen_samples[i, :]).any() == False

    gen_samples = np.delete(gen_samples, np.s_[(i + 1) * args.batch_size:], 0)
    print(gen_samples.shape[0])
    gen_samples[gen_samples >= 0.5] = 1.0
    gen_samples[gen_samples < 0.5] = 0.0

    # ave synthetic data
    np.save(os.path.join(args.MODELPATH, "synthetic.npy"), gen_samples)

def evaluate():

    # Load synthetic data
    gen_samples = np.load(os.path.join(args.MODELPATH, "synthetic.npy"), allow_pickle=True)

    # Load real data
    real_samples = dataset_train_object.return_data()[0:gen_samples.shape[0], :]

    # Dimenstion wise probability
    prob_real = np.mean(real_samples, axis=0)
    prob_syn = np.mean(gen_samples, axis=0)

    colors = (0, 0, 0)
    plt.scatter(prob_real, prob_syn, c=colors, alpha=0.5)
    x_max = max(np.max(prob_real), np.max(prob_syn))
    x = np.linspace(0, x_max + 0.1, 1000)
    plt.plot(x, x, linestyle='-', color='k')  # solid
    plt.title('Scatter plot p')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    if args.train:

        for epoch in range(1, args.epochs + 1):


            train(epoch)
            test(epoch)
            with torch.no_grad():
                sample_size = 64
                sample = torch.randn(sample_size, args.compress_dim).to(device)
                sample = model.decode(sample).cpu()
                sample[sample >= 0.5] = 1.0
                sample[sample < 0.5] = 0.0
                unique, counts = np.unique(sample, return_counts=True)
                print(dict(zip(unique, counts)))
    else:
        generate()
        evaluate()

