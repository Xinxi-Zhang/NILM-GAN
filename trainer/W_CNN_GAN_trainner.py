import numpy as np
from torch.utils.data import DataLoader
from Data_util.NILMDataset import NILMDataset
import sys
from my_model.W_CNN_GAN import *
from my_model.model_saver import saver
import torch.autograd as autograd
from my_model.model_loader import model_loader

if __name__ == '__main__':
    # initial some basic parameters
    batch_size = 512
    num_epoch = 5000
    window_size = 599
    z_dimension = 32
    d_lr = 1e-4
    g_lr = 1e-4
    start_epoch = 40
    Lambda = 10
    critic_steps = 1
    mission = 'mv'
    saver = saver(taskname=mission, flag=False)

    # to load the data using the NILMDataset
    data = NILMDataset(r'D:\Research\NILM\dataset\uk_dale', 'wm', 'uk', window_size, houses=[1], mode=0)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    loader = model_loader(model_type='W_CNN_GAN',
                          d_load_path=r'C:\Users\69037\Desktop\NILM\git\NILM-GAN\save\mv\discriminator\40.pkl',
                          g_load_path=r'C:\Users\69037\Desktop\NILM\git\NILM-GAN\save\mv\generator\40.pkl')

    # define the discriminator and generator
    D = loader.discriminator()
    G = loader.generator()
    critic = utils()
    if torch.cuda.is_available():
        D = D.cuda()
        G = G.cuda()

    # Binary cross entropy loss and optimizer
    d_optimizer = torch.optim.SGD(D.parameters(), lr=d_lr)
    g_optimizer = torch.optim.SGD(G.parameters(), lr=g_lr)

    one = torch.tensor(1, dtype = torch.float)
    mone = one * -1
    one = one.cuda()
    mone = mone.cuda()

    for epoch in range(start_epoch, num_epoch):

        real_scores_l = []
        fake_scores_l = []
        G_cost_l = []
        D_cost_l = []
        W_l = []

        for i, (real_load, real_label) in enumerate(dataloader):
            for p in D.parameters():
                p.requires_grad = True
            for p in G.parameters():
                p.requires_grad = False

            if i % critic_steps == 0:
                flag = True
            else:
                flag = False

            real_load = real_load.reshape(real_load.shape[0], window_size).cuda()
            real_load_v = autograd.Variable(real_load)

            # train with real
            D.zero_grad()
            real_scores = D(real_load_v)
            real_scores = real_scores.mean()

            # train with fake
            z = torch.rand(real_load.shape[0], z_dimension, window_size).cuda()
            min, max = data.sample_min_max()
            z = z * (max - min) + min
            z_v = autograd.Variable(z)

            fake_load = G(z_v, min, max)
            fake_load = autograd.Variable(fake_load.data)

            inputV = fake_load

            fake_scores = D(inputV)
            fake_scores = fake_scores.mean()

            # train with gradient penalty
            penalty = critic.calc_gradient_penalty(D, real_load_v.data, fake_load.data, real_load.shape[0])

            D_cost = fake_scores - real_scores + penalty
            D_cost.backward()
            Wasserstein_D = real_scores - fake_scores
            d_optimizer.step()

            fake_scores_l.append(float(fake_scores.data))
            real_scores_l.append(float(real_scores.data))
            D_cost_l.append(float(D_cost.data))
            W_l.append(float(Wasserstein_D.data))

            if flag == True:
                for p in D.parameters():
                    p.requires_grad = False  # to avoid computation
                for p in G.parameters():
                    p.requires_grad = True
                G.zero_grad()
                z = torch.rand(real_load.shape[0], z_dimension, window_size).cuda()
                z = z.cuda()
                min, max = data.sample_min_max()
                z = z * (max - min) + min
                z_v = autograd.Variable(z)
                fake_load = G(z_v, min, max)
                fake_scores = D(fake_load)
                fake_scores = -fake_scores.mean()
                fake_scores.backward()
                G_cost = -fake_scores
                g_optimizer.step()
                G_cost_l.append(float(G_cost.data))

        f = open(saver.get_txt_path(), 'a')

        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
              'D real: {:.6f}, D fake: {:.6f}, Wass_cost: {:.6f}, D Step: {:.1f}'.format(
            epoch + 1, num_epoch, np.mean(D_cost_l), np.mean(G_cost_l),
            np.mean(real_scores_l), np.mean(fake_scores_l), np.mean(W_l), critic_steps), file=f,
            flush=True)

        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
              'D real: {:.6f}, D fake: {:.6f}, Wass_cost: {:.6f}, D Step: {:.1f}'.format(
            epoch + 1, num_epoch, D_cost.data, G_cost.data,
            real_scores.data.mean(), fake_scores.data.mean(), Wasserstein_D.data.mean(), critic_steps), file=sys.stdout)
        f.close()

        if (epoch + 1) % 1 == 0:
            for k in range(10):
                z = torch.rand(1, z_dimension, window_size).cuda()
                min, max = data.sample_min_max()
                z = z * (max - min) + min
                z_v = autograd.Variable(z)
                fake_load = G(z_v, min, max)
                l = fake_load.cpu().detach().numpy()
                l = l.reshape(window_size)
                l = l.tolist()
                saver.save_img(img=l, epoch=epoch, index=k)

        if (epoch + 1) % 1 == 0:
            saver.save_model(g=G, d=D, epoch=epoch)
