'''清洗家庭单个电器的6s功率数据'''
import time
import pandas as pd
import numpy as np
import math
'''超参数tachi'''
house = 4
channel_num = 6


# 清洗每个channel的数据
for channel in range(1, channel_num + 1):
# for channel in [5, 6, 10, 12, 13]:
    # 获取原始数据
    '''在洗数据之前判断一下数据是否存在重复逆序以及负值，确认无误后在开始洗'''
    data = pd.read_csv('raw_data/house_%d/channel_%d.dat' % (house, channel), header=None, delimiter=" ")
    '''在洗数据之后判断一下数据是否存在重复逆序以及负值，确认无误后在开始使用'''
    # data = pd.read_csv('clean_data/house_%d/channel_%d.csv' % (house, channel), header=None)
    # print('channel%d:duplicated=%s_non-ordered=%s_negative=%s' % (channel, True in data[0].duplicated().values, True in (data[0].diff(1).values[1:] < 0), True in (data[1].values < 0)))
    # continue
    data = np.array(data)

    '''取得该channel的始末时间戳'''
    start_time_stamp = time.localtime(int(data[0][0] / 6) * 6) # 干净数据的起始时间，即原数据起始时间的向下取整
    # otherStyleTime1 = time.strftime("%Y--%m--%d %H:%M:%S", start_time_stamp)
    start_time_stamp = int(time.mktime(start_time_stamp))
    end_time_stamp = time.localtime(math.ceil(data[-1][0] / 6) * 6) # 干净数据的终止时间，即原数据终止时间的向上取整
    # otherStyleTime2 = time.strftime("%Y--%m--%d %H:%M:%S", end_time_stamp)
    end_time_stamp = int(time.mktime(end_time_stamp))

    # print('channel%d——starttime:%s, endtime:%s' % (channel, otherStyleTime1, otherStyleTime2))

    # 取原数据中读数不为0的数据
    data = data[data[:, 1] != 0]
    # 原始数据中可直接用于插值的数据（整6）
    directly_interpolate_data = data[(data[:, 0] - start_time_stamp) % 6 == 0]

    '''根据时间戳范围制造待插值数组'''
    # 待插值的数组
    data_to_be_interpolated = np.concatenate((np.arange(start_time_stamp, end_time_stamp + 6, 6).reshape(-1, 1),
                                              np.zeros((int((end_time_stamp - start_time_stamp) / 6) + 1, 1))), axis=1)
    # 直接将整6的可直接插值的原始数据赋给待插值数组，如果不存在整6的时间点则跳过
    if directly_interpolate_data.shape[0] != 0:
        data_to_be_interpolated[np.in1d(data_to_be_interpolated[:, 0], directly_interpolate_data[:, 0]), 1] = directly_interpolate_data[:, 1]

    '''取每个待插值时间点的上下界'''
    bound_set = []
    # 找到所有待插值时间点在原始数据中的位置
    insert_position = np.searchsorted(data[:, 0], data_to_be_interpolated[:, 0])
    for i in range(insert_position.shape[0]):
        # 跳过有值的点
        if data_to_be_interpolated[i][1] == 0:
            print("channel_%d:%d/%d" % (channel, i, data_to_be_interpolated.shape[0]))
            # 若上下界有一方不存在，则直接处理，否则将上下界放入上下界数组中等之后统一处理
            # 上界不存在
            if insert_position[i] == data.shape[0]:
                lower_bound = data[-1]

                if data_to_be_interpolated[i][0] - lower_bound[0] < 6:  # 小于六秒钟，读数记为下界读数即可
                    data_to_be_interpolated[i][1] = lower_bound[1]

            # 下界不存在
            elif insert_position[i] == 0:
                upper_bound = data[0]

                if upper_bound[0] - data_to_be_interpolated[i][0] < 6:  # 小于六秒钟，读数记为上界读数即可
                    data_to_be_interpolated[i][1] = upper_bound[1]

            # 上下界均存在
            else:
                bound_set.append([data_to_be_interpolated[i][0], data[insert_position[i]][0], data[insert_position[i] - 1][0]])


    '''根据每个待插值的时间点的上下界之差来进行插值操作'''
    # 判断每个未处理的时间点的上下界数组是否为空
    bound_set = np.array(bound_set)
    if bound_set.shape[0] != 0:
        data_to_be_interpolated = pd.DataFrame(data_to_be_interpolated).set_index(0)[1]
        data = pd.DataFrame(data).set_index(0)[1]

        # 小于120s大于12s的时间点直接去下界的功率即可
        follow_previous_power = bound_set[((bound_set[:, 1] - bound_set[:, 2]) < 120) & ((bound_set[:, 1] - bound_set[:, 2]) >= 12)]
        data_to_be_interpolated[follow_previous_power[:, 0]] = data[follow_previous_power[:, 2]].values

        # 小于12s的时间点需要根据上下界进行插值处理
        need_to_be_interpolated = bound_set[((bound_set[:, 1] - bound_set[:, 2]) < 12)]
        k = (data[need_to_be_interpolated[:, 1]].values - data[need_to_be_interpolated[:, 2]].values) \
            / (need_to_be_interpolated[:, 1] - need_to_be_interpolated[:, 2])
        b = (need_to_be_interpolated[:, 1] * data[need_to_be_interpolated[:, 2]].values -
             need_to_be_interpolated[:, 2] * data[need_to_be_interpolated[:, 1]].values) /\
            (need_to_be_interpolated[:, 1] - need_to_be_interpolated[:, 2])
        data_to_be_interpolated[need_to_be_interpolated[:, 0]] = k * need_to_be_interpolated[:, 0] + b

    pd.DataFrame(data_to_be_interpolated).to_csv('clean_data/house_%d/channel_%d.csv' % (house, channel), header=None)
