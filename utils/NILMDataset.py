from torch.utils.data import  Dataset
import numpy as np
import pandas as pd
import torch
import random
uk_houses = [1, 5]
uk_channels = {
    'wm' : [5, 24],
    'dw' : [6, 22],
    'kt' : [10, 18],
    'fr' : [12, 19],
    'mw' : [13, 23]
}
redd_houses = [2, 3, 4, 5, 6]
redd_channels = {
    'wm' : [7, 13, 7, 8, 4],
    'dw' : [10, 9, 15, 20, 9],
    'kt' : [-1, -1, -1, -1, -1],
    'fr' : [9, 7, -1, 18, 8],
    'mw' : [6, 16, -1, 3, -1]
}
refit_houses = [2, 3, 5, 9]
refit_channels = {
    'wm' : [2, 13, 7, 8],
    'dw' : [3, 9, 15, 20],
    'kt' : [-1, -1, -1, -1],
    'fr' : [9, 7, -1, 18],
    'mw' : [6, 16, -1, 3]
}
class NILMDataset(Dataset):

    #数据集初始化，mode = 0 为训练集， 1 为验证集， 2为测试集
    def __init__(self,path,app,dataset,window_size,houses,mode = 0):
        self.all_houses_aggregate_data = []
        self.valid_sample_indices = []
        self.path = path

        if dataset == 'uk':
            self.houses = uk_houses
            self.channels = uk_channels[app]

        if dataset == 'redd':
            self.houses = redd_houses
            self.channels = redd_channels[app]
            
        if dataset == 'refit':
            self.houses = refit_houses
            self.channels = refit_channels[app]
            
        self.window_size = window_size

        for i in range(0,len(self.houses)):
            if self.houses[i] not in houses:
                continue
            if self.channels[i] == -1:
                continue
            tmp = self.makepath(i,-1)
            aggregate_data = np.array(pd.read_csv(tmp,header = None))
            tmp = self.makepath(i,1)
            appliance_data = np.array(pd.read_csv(tmp, header = None))

            # 获取电器数据和总数据的公共时间戳部分
            if appliance_data[0][0] < aggregate_data[0][0]:
                appliance_data = np.delete(appliance_data, np.where(appliance_data[:, 0] < aggregate_data[0][0])[0], 0)
            else:
                aggregate_data = np.delete(aggregate_data, np.where(aggregate_data[:, 0] < appliance_data[0][0])[0], 0)
            if appliance_data[-1][0] > aggregate_data[-1][0]:
                appliance_data = np.delete(appliance_data, np.where(appliance_data[:, 0] > aggregate_data[-1][0])[0], 0)
            else:
                aggregate_data = np.delete(aggregate_data, np.where(aggregate_data[:, 0] > appliance_data[-1][0])[0], 0)

            # 删除时间戳，保留功率数值
            aggregate_data = aggregate_data[:, 1]
            appliance_data = appliance_data[:, 1]
            self.all_houses_aggregate_data.append(aggregate_data)


            '先获得可用的有效样本'
            # 获取所有样本的索引
            all_sample_indices = np.array([i for i in range(0, self.all_houses_aggregate_data[-1].shape[0] - window_size + 1)])
            # 去掉总数据中有0的样本索引，获得该家庭的有效样本索引
            zero_sample_indices = np.where(self.all_houses_aggregate_data[-1] == 0)[0]
            zero_sample_indices = np.unique(np.concatenate((zero_sample_indices, zero_sample_indices - window_size + 1), axis=0))
            valid_sample_indices = all_sample_indices[~np.in1d(all_sample_indices, zero_sample_indices)]
            # 去掉标签中电器数据比总数据大的样本索引
            abnormal_sample_indices = np.where(aggregate_data < appliance_data)[0] - int((window_size - 1) / 2)
            valid_sample_indices = valid_sample_indices[~np.in1d(valid_sample_indices, abnormal_sample_indices)]

            # 载入该家庭的正样本索引
            tmp = self.makepath(i,0)
            positive_sample_indices = np.array(pd.read_csv(tmp, header=None), dtype=int).reshape(-1)
            # 获得有效的正样本索引
            valid_positive_sample_indices = valid_sample_indices[np.in1d(valid_sample_indices, positive_sample_indices)]

            if mode == 0:
                valid_positive_sample_indices = valid_positive_sample_indices[0 : int(0.9*valid_positive_sample_indices.shape[0])]
            elif mode == 1:
                valid_positive_sample_indices = valid_positive_sample_indices[int(0.9 * valid_positive_sample_indices.shape[0]) : int(valid_positive_sample_indices.shape[0])]
            self.valid_sample_indices.append(valid_positive_sample_indices)

        del appliance_data
        del aggregate_data

        # 获取每个家庭数据的维度，用于将所有家庭的有效样本索引储存在一起
        all_houses_size = [house.shape[0] for house in self.all_houses_aggregate_data]
        all_houses_size = [np.sum(all_houses_size[: i + 1]) for i in range(0, len(all_houses_size))]
        for i in range(1, len(self.valid_sample_indices)):
            self.valid_sample_indices[0] = np.concatenate((self.valid_sample_indices[0],
                                                           self.valid_sample_indices[i] + all_houses_size[i - 1]),
                                                          axis=0)
        self.valid_sample_indices = self.valid_sample_indices[0]

        # 合并所有家庭的总表数据
        for i in range(1, len(self.all_houses_aggregate_data)):
            self.all_houses_aggregate_data[0] = np.concatenate((self.all_houses_aggregate_data[0],
                                                                self.all_houses_aggregate_data[i]), axis=0)
        self.all_houses_aggregate_data = self.all_houses_aggregate_data[0]

        # 获取数据长度（有效样本数量）
        self.len = self.valid_sample_indices.shape[0]

        # 获取minmax数据
        self.min_max_list = self.get_min_max()

    def makepath(self,num,mode):
        house = self.houses[num]
        channel = self.channels[num]
        #mode = -1 means aggregate data path
        if mode == -1:
            path = self.path+'\\house_'+str(house)+'\\mains.csv'
            return path

        #mode = 1 means appliance data path
        if mode == 1:
            path = self.path+'\\house_'+str(house)+'\\channel_'+str(channel)+'\\channel_'+str(channel)+'.csv'
            return path

        # mode = 0 means positive indices data path
        if mode == 0:
            path = self.path + '\\house_' + str(house) + '\\channel_' + str(channel) + '\\positive_sample_indices.csv'
            return path

    def __len__(self):
        return self.len

        # 获取数据集中的元素
    def __getitem__(self, index):
        selected_index = self.valid_sample_indices[index]
        # 获取训练数据
        selected_sample_data = np.expand_dims(
            self.all_houses_aggregate_data[selected_index: selected_index + self.window_size], axis=1).transpose(1, 0)
        selected_sample_data = torch.from_numpy(selected_sample_data).type(torch.FloatTensor)
        # 获取训练标签
        selected_sample_labels = torch.from_numpy(
            np.array([1])).type(torch.FloatTensor)
        return selected_sample_data, selected_sample_labels

    def get_min_max(self):
        list = []
        index = 0
        while(index < self.len):
            tmp, b = self.__getitem__(index)
            tmp = min(tmp)
            minn = min(tmp)
            maxx = max(tmp)
            list.append([minn.item(), maxx.item()])
            index += int(self.window_size/2)
        return list

    def sample_min_max(self):
        index = random.randint(0, len(self.min_max_list)-1)
        tmp = self.min_max_list[index]
        return tmp[0], tmp[1]



if __name__ == '__main__':
    dataset = NILMDataset(r'D:\科研项目\NILM\dataset\uk_dale', 'wm', 'uk', 599)
