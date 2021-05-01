import numpy as np
from Data_util.NILMDataset import NILMDataset
from torch.utils.data import DataLoader
import sys
from torch.nn import functional as F
from my_model.W_CNN_GAN import *
from my_model.model_loader import model_loader
from scipy.stats import entropy
from torchvision.models.inception import inception_v3

epoch_num = 150
N = 512
window_size = 599
z_dimension = 32
data = NILMDataset(r'D:\Research\NILM\dataset\uk_dale', 'mw', 'uk', window_size, houses=[1], mode=0)
path = r'C:\Users\69037\Desktop\NILM\git\NILM-GAN\save\uk-house1-macrowave'
txt_path = path + '\\' + 'IS_validation.txt'

def inception_score(seqs, batch_size=64, splits=10):
    N = len(seqs)
    dataloader = DataLoader(seqs, batch_size=batch_size)

    # Load the inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model.eval()
    up = nn.Upsample(size = (599, 100), mode='bilinear', align_corners=False).cuda()

    def get_pred(x):
        x = up(x)
        x = inception_model(x)
        out = F.softmax(x, dim=1).data.cpu().numpy()
        del x
        return out

    # Get prediction output using inception_v3 model
    preds = np.zeros((N, 1000))
    for i, batch in enumerate(dataloader, 0):
        batch = batch.cuda()
        batch_size_i = batch.size()[0]
        with torch.no_grad():
            preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch)

    # compute the mean KL Divergence
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)


for i in range(1, epoch_num):
    G_path = path + '\\generator\\' + str(i) +'.pkl'
    loader = model_loader(model_type='W_CNN_GAN',g_load_path=G_path,d_load_path=None)
    G = loader.generator().cuda()

    # get scale
    minn, maxx = data.sample_min_max()
    g_max = maxx
    g_min = minn
    seqs = None
    for j in range(N):
        # resample min max
        if j % 5 == 0:
            minn, maxx = data.sample_min_max()
            if minn < g_min:
                g_min = minn
            if maxx > g_max:
                g_max = maxx
        z = torch.rand(1, z_dimension, window_size).cuda()
        z = z * (maxx - minn) + minn
        z_v = torch.autograd.Variable(z).cuda()
        fake_load = G(z_v, minn, maxx)
        new = fake_load.reshape(1, 1, 599, 1).data.cpu().numpy().repeat(3, 1)
        if seqs is None:
            seqs = new
        else:
            seqs = np.concatenate((seqs,new), axis=0)
    seqs = (seqs-g_min)/(g_max-g_min)
    f = open(txt_path, 'a')
    is_score =inception_score(seqs)

    print('Epoch [{}/{}], IS: {:.6f},'.format(
        i + 1, epoch_num, is_score), file=f,
        flush=True)
    print('Epoch [{}/{}], IS: {:.6f},'.format(
        i + 1, epoch_num, is_score), file=sys.stdout)
