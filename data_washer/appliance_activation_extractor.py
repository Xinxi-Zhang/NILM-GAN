import torch
import torchvision
import nilmtk.electric as elec
import nilmtk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

'''hyperparameters'''
dataset = '8_downsampled_uk_dale'
# 编号顺序washing machine, dish washer, kettle, fridge, microwave
# house = 1
# channels = [5, 6, 10, 12, 13]
# house = 2
# channels = [12, 13, 8, 14, 15]
house = 5
channels = [24, 22, 18, 19, 23]

on_power_threshold = {
    channels[0]: 20,
    channels[1]: 10,
    channels[2]: 2000,
    channels[3]: 50,
    channels[4]: 200
}
min_on_duration = {
    channels[0]: 1800,
    channels[1]: 1800,
    channels[2]: 12,
    channels[3]: 60,
    channels[4]: 12
}
min_off_duration = {
    channels[0]: 160,
    channels[1]: 1800,
    channels[2]: 0,
    channels[3]: 12,
    channels[4]: 30
}

'''初始化载入数据'''
# 载入电器名称字典
appliance_name = pd.read_csv('uk_dale/house_%d/appliance_name.csv' % (house), header=None).set_index(0)[1]
print('house%d' % house)

for channel in channels:
    # 电器功率数据
    appliance_data = pd.read_csv('%s/house_%d/channel_%d.csv' % (dataset, house, channel), header=None)
    # pd.datetime()记录时间戳是从1970-01-01 00:00:00开始的，
    # 而time.mktime()是从1970-01-01 08:00:00开始的，
    # 所以按照uk-dale论文中的说法，还是pd.datetime()是准的
    appliance_data[0] = pd.to_datetime(appliance_data[0],unit='s')
    appliance_data = appliance_data.set_index(0)[1]
    print('%s——start:%s,end:%s' % (appliance_name[channel], appliance_data.index[0], appliance_data.index[-1]))

    # 获取电器activations
    activations = elec.get_activations(appliance_data, on_power_threshold=on_power_threshold[channel], min_on_duration=min_on_duration[channel], min_off_duration=min_off_duration[channel])

    '''储存每个电器每个activation的起止时间戳'''
    origins_and_terminations = []
    for activation in activations:
        origin = activation.index[0] - pd.datetime(1970,1,1)
        origin = origin.days * 86400 + origin.seconds

        termination = activation.index[-1] - pd.datetime(1970,1,1)
        termination = termination.days * 86400 + termination.seconds

        origins_and_terminations.append([origin, termination])

    pd.DataFrame(origins_and_terminations).to_csv('%s/house_%d/channel%d_activations.csv' % (dataset, house, channel), header=None, index=None)
    print("%s_done" % appliance_name[channel])





    '''打印每个activation的图，包含电器数据和1s降采样后的总功率数据'''
    # 家庭总功率数据
    # aggragate_data_downsampled = pd.read_csv('clean_data/house_%d/downsampled_active_mains.csv' % house, header=None).set_index(0)[1]

    # for activation in activations:
    #     spike_number = 0
    #     # 将activation中的日期格式变为对应的时间戳
    #     temp = activation.index - pd.datetime(1970,1,1)
    #     activation.index = temp.days * 86400 + temp.seconds
    #
    #     plt.figure()
    #     plt.plot(np.arange(0, activation.shape[0]), activation.values)
    #
    #     # 判断该activation是否在1s数据中，首先获得1s起始数据的时间戳
    #     # if activation.index[0] >= aggragate_data_downsampled.index[0]:
    #     #     corresponding_aggregate_data_downsampled = aggragate_data_downsampled[activation.index[0]: activation.index[-1]]
    #     #     # plt.plot(np.arange(0, activation.shape[0]), corresponding_aggregate_data_downsampled.values,
    #     #     #          label='aggregate_data_downsampled')
    #     #     spike_number = activation[activation >= corresponding_aggregate_data_downsampled].shape[0] / activation.shape[0]
    #
    #     # 图片名称制作：开始时间，结束时间，持续时间，有无spike(电器数据大于总功率数据的点)
    #     start_time = pd.to_datetime(activation.index[0], unit='s')
    #     end_time = pd.to_datetime(activation.index[-1], unit='s')
    #     start_time_str = "%d年%d月%d日%d时%d分%d秒" % \
    #                  (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    #     end_time_str = "%d年%d月%d日%d时%d分%d秒" % \
    #                  (end_time.year, end_time.month, end_time.day, end_time.hour, end_time.minute, end_time.second)
    #     duration = end_time - start_time
    #     duration = "%d天%d时%d分%d秒" % \
    #                (duration.components.days, duration.components.hours, duration.components.minutes, duration.components.seconds)
    #     plt.ylabel('Watt')
    #     plt.xlabel('Minutes')
    #     # plt.xticks(np.arange(0, activation.values.shape[0]) / 10)
    #     # 保存图片
    #     plt.savefig('figure/house_%d/appliance_activations/%s/starttime=%s_endtime=%s_duration=%s_spike=%f.png' %
    #                 (house, appliance_name[channel], start_time_str, end_time_str, duration, spike_number))
    #     plt.close()
