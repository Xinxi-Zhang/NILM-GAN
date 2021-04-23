import numpy as np
from torch.utils.data import DataLoader
from Data_util.NILMDataset import NILMDataset
import sys
from my_model.W_CNN_GAN import *
from my_model.model_loader import model_loader

epoch_num = 89
batch_size = 512
window_size = 599
z_dimension = 32
critic = utils()
path = r'C:\Users\69037\Desktop\NILM\git\NILM-GAN\save\W-CNN-GAN-00-uk-house1-wm(same_pace)'
txt_path = path + '\\' + 'validation.txt'
validate_data = NILMDataset(r'D:\Research\NILM\dataset\uk_dale', 'wm', 'uk', window_size, houses=[1], mode=1)
validata_dataloader = DataLoader(validate_data, batch_size=batch_size, shuffle=True)
loader = model_loader(model_type='W_CNN_GAN',
                          d_load_path=r'C:\Users\69037\Desktop\NILM\git\NILM-GAN\save\W-CNN-GAN-00-uk-house1-wm(SGD_mv)\discriminator\182.pkl',
                          g_load_path=r'C:\Users\69037\Desktop\NILM\git\NILM-GAN\save\W-CNN-GAN-00-uk-house1-wm(SGD_mv)\generator\182.pkl')

for i in range(1, epoch_num):
    # get load path
    d_load_path = path + '\\discriminator\\' + str(i) + '.pkl'
    g_load_path = path + '\\generator\\' + str(i) + '.pkl'
    loader = model_loader(model_type='W_CNN_GAN',
                          d_load_path=d_load_path,
                          g_load_path=g_load_path)

    # load
    D = loader.discriminator()
    G = loader.generator()
    if torch.cuda.is_available():
        D = D.cuda()
        G = G.cuda()
    real_scores_l = []
    fake_scores_l = []
    D_cost_l = []
    Wasserstein_D_l = []
    for j, (real_load, real_label) in enumerate(validata_dataloader):

        real_load = real_load.reshape(real_load.shape[0], window_size).cuda()
        real_load_v = torch.autograd.Variable(real_load)
        real_scores = D(real_load_v)
        real_scores = real_scores.mean()
        real_scores_l.append(float(real_scores.data))

        z = torch.rand(real_load.shape[0], z_dimension, window_size).cuda()
        min, max = validate_data.sample_min_max()
        z = z * (max - min) + min
        z_v = torch.autograd.Variable(z)

        fake_load = G(z_v, min, max)
        fake_load = torch.autograd.Variable(fake_load.data)

        inputV = fake_load

        fake_scores = D(inputV)
        fake_scores = fake_scores.mean()
        fake_scores_l.append(float(fake_scores.data))

        # train with gradient penalty
        penalty = critic.calc_gradient_penalty(D, real_load_v.data, fake_load.data, real_load.shape[0])
        D_cost = fake_scores - real_scores + penalty
        D_cost_l.append(float(D_cost.data))
        Wasserstein_D = real_scores - fake_scores
        Wasserstein_D_l.append(float(Wasserstein_D))

    f = open(txt_path, 'a')

    print('Epoch [{}/{}], d_loss: {:.6f}, '
          'D real: {:.6f}, D fake: {:.6f}, Wass_cost: {:.6f},'.format(
        i + 1, epoch_num, np.mean(D_cost_l),
        np.mean(real_scores_l), np.mean(fake_scores_l), np.mean(Wasserstein_D_l)), file=f,
        flush=True)

    print('Epoch [{}/{}], d_loss: {:.6f}, '
          'D real: {:.6f}, D fake: {:.6f}, Wass_cost: {:.6f},'.format(
        i + 1, epoch_num, np.mean(D_cost_l),
        np.mean(real_scores_l), np.mean(fake_scores_l), np.mean(Wasserstein_D_l)), file=sys.stdout)

    f.close()


