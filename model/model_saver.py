import os
import torch
from Data_util.paint import plot_window_size
class saver:
    def __init__(self,taskname,flag):
        self.flag = flag
        path = r'C:\Users\69037\Desktop\NILM\git\NILM-GAN\save'
        self.main_path = path + '\\' + taskname
        self.d_path = self.main_path + '\\' + 'discriminator'
        self.g_path = self.main_path + '\\' + 'generator'
        self.img_path = self.main_path + '\\' + 'output'
        self.make_path()

    def make_path(self):
        if self.flag == True:
            os.makedirs(self.main_path)
            os.makedirs(self.d_path)
            os.makedirs(self.g_path)
            os.makedirs(self.img_path)

    def save_model(self,d,g,epoch):
        tmp = self.g_path + '\\' + str(epoch + 1) + '.pkl'
        torch.save(g.state_dict(), tmp)
        tmp = self.d_path + '\\' + str(epoch + 1) + '.pkl'
        torch.save(d.state_dict(), tmp)

    def save_img(self,img,epoch):
        tmp = self.img_path + '\\' + str(epoch + 1) + '.png'
        plot_window_size(img, save_path=tmp)
