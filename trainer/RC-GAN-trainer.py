import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Data_util.NILMDataset import NILMDataset
from my_model.RC_GAN import discriminator, generator
from my_model.model_saver import saver
if __name__ == '__main__':
    # initial some basic parameters
    batch_size = 512
    num_epoch = 5000
    window_size = 599
    z_dimension = 100
    lr = 0.0003
    mission = 'RC-GAN-01-uk-house1-wm'
    saver = saver(taskname = mission, flag = True)

    # the steps for training epoch for discriminator and generator
    d_steps = 10
    g_steps = 1

    # to load the data using the NILMDataset
    data = NILMDataset(r'D:\Research\NILM\dataset\uk_dale', 'wm', 'uk', window_size, houses=[1], mode= 0)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # define the discriminator and generator
    D = discriminator()
    G = generator()
    if torch.cuda.is_available():
        D = D.cuda()
        G = G.cuda()

    # Binary cross entropy loss and optimizer
    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr)

    for epoch in range(num_epoch):

        for i, (real_load, real_label) in enumerate(dataloader):
            flag = False
            #print("i = " + str(i) + ': discriminator training ')
            # =================train discriminator
            real_load = real_load.reshape(real_load.shape[0], window_size).cuda()
            real_label = real_label.cuda()
            fake_label = torch.zeros(real_load.shape[0], 1).cuda()

            # compute loss of real_img
            real_out = D(real_load)
            d_loss_real = criterion(real_out, real_label)
            real_scores = real_out  # closer to 1 means better

            # compute loss of fake_img
            z = torch.randn(real_load.shape[0], z_dimension,1).cuda()
            fake_load = G(z)
            fake_out = D(fake_load)
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out  # closer to 0 means better

            # bp and optimize
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            d_loss.backward()
            d_optimizer.step()

            if ((i+1)%d_steps == 0):
                if real_scores.data.mean() > 0.95:
                    flag = True

            while(flag):

                # print("i = " + str(i) + ': generator training ')
                # ===============train generator
                # compute loss of fake_img
                z = torch.randn(batch_size, z_dimension, 1).cuda()
                fake_load = G(z)
                output = D(fake_load)
                real_label = torch.ones(batch_size, 1).cuda()
                g_loss = criterion(output, real_label)

                # bp and optimize
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                if output.data.mean() > 0.9:
                    # print(j, output.data.mean())
                    break


            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                'D real: {:.6f}, D fake: {:.6f}'.format(
                epoch, num_epoch, d_loss.data, g_loss.data,
                real_scores.data.mean(), fake_scores.data.mean()))

        if (epoch + 1) % 20 == 0:
            for k in range(10):
                z = torch.randn(1, z_dimension, 1).cuda()
                fake_load = G(z)
                l = fake_load.cpu().detach().numpy()
                l = l.reshape(window_size)
                l = l.tolist()
                saver.save_img(img=l, epoch=epoch, index=k)

        if (epoch + 1) % 50 == 0:
            saver.save_model(g=G, d=D, epoch=epoch)











