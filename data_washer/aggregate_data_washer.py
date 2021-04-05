import time
import pandas as pd
import numpy as np
import math

'''超参数tati'''
house = 5
chunk_size = 10000000

'''取得总表1s数据'''
mains_data = pd.read_csv('raw_data/house_%d/mains.dat' % house, header=None, delimiter=" ", chunksize=chunk_size, iterator=True)
# mains_data = pd.read_csv('clean_data/house_%d/reactive_mains.csv' % house, header=None, chunksize=chunk_size, iterator=True)

chunk_number = 1
'''循环遍历所有的总表数据'''
for data in mains_data:
    print('开始处理第%d个chunk，chunk_size=%d' % (chunk_number, chunk_size))
    # print('chunk%d:duplicated=%s_non-ordered=%s_negative=%s' % (chunk_number, True in data[0].duplicated().values, True in (data[0].diff(1).values[1:] < 0), True in (data[1].values < 0)))
    chunk_number += 1
    # continue

    # # 用于定位错误
    # if chunk_number != 11:
    #     continue

    '''原始数据去重（保留第一个），重复分为两种情况：①单纯地相邻两个条目的时间戳相等②数据中偶尔会出现一小部分时间戳与很久之前的条目的时间戳相等的数据，这些数据可以当作是错误数据'''
    data = data[~data[0].duplicated()]
    # 取有效功率
    data = np.array(data[[0, 1]])

    '''取得该channel的始末时间戳'''
    start_time_stamp = time.localtime(int(data[0][0]))  # 干净数据的起始时间，即原数据起始时间的向下取整
    start_time_stamp = int(time.mktime(start_time_stamp))
    end_time_stamp = time.localtime(int(data[-1][0]))  # 干净数据的终止时间，即原数据终止时间的向下取整
    end_time_stamp = int(time.mktime(end_time_stamp))

    # 取原数据中读数不为0的数据
    data = data[data[:, 1] != 0]
    # 原始数据中可直接用于插值的数据（整1）
    directly_interpolate_data = data[(data[:, 0] - start_time_stamp) % 1 == 0]

    '''根据时间戳范围制造待插值数组'''
    # 待插值的数组
    data_to_be_interpolated = np.concatenate((np.arange(start_time_stamp, end_time_stamp + 1).reshape(-1, 1),
                                              np.zeros((int(end_time_stamp - start_time_stamp) + 1, 1))), axis=1)
    # 直接将整1的可直接插值的原始数据赋给待插值数组，如果不存在整1的时间点则跳过
    if directly_interpolate_data.shape[0] != 0:
        data_to_be_interpolated[
            np.in1d(data_to_be_interpolated[:, 0], directly_interpolate_data[:, 0]), 1] = directly_interpolate_data[:, 1]

    '''取每个待插值时间点的上下界'''
    bound_set = []
    # 找到所有待插值时间点在原始数据中的位置
    insert_position = np.searchsorted(data[:, 0], data_to_be_interpolated[:, 0])
    for i in range(insert_position.shape[0]):
        # 跳过有值的点
        if data_to_be_interpolated[i][1] == 0:
            # 若上下界有一方不存在，则直接处理，否则将上下界放入上下界数组中等之后统一处理
            # 上界不存在
            if insert_position[i] == data.shape[0]:
                lower_bound = data[-1]

                if data_to_be_interpolated[i][0] - lower_bound[0] < 1:  # 小于六秒钟，读数记为下界读数即可
                    data_to_be_interpolated[i][1] = lower_bound[1]

            # 下界不存在
            elif insert_position[i] == 0:
                upper_bound = data[0]

                if upper_bound[0] - data_to_be_interpolated[i][0] < 1:  # 小于六秒钟，读数记为上界读数即可
                    data_to_be_interpolated[i][1] = upper_bound[1]

            # 上下界均存在
            else:
                bound_set.append(
                    [data_to_be_interpolated[i][0], data[insert_position[i]][0], data[insert_position[i] - 1][0]])

    '''根据每个待插值的时间点的上下界之差来进行插值操作'''
    # 判断每个未处理的时间点的上下界数组是否为空
    bound_set = np.array(bound_set)
    if bound_set.shape[0] != 0:
        data_to_be_interpolated = pd.DataFrame(data_to_be_interpolated).set_index(0)[1]
        data = pd.DataFrame(data).set_index(0)[1]

        # 小于120s大于12s的时间点直接去下界的功率即可
        follow_previous_power = bound_set[
            ((bound_set[:, 1] - bound_set[:, 2]) < 120) & ((bound_set[:, 1] - bound_set[:, 2]) >= 2)]
        data_to_be_interpolated[follow_previous_power[:, 0]] = data[follow_previous_power[:, 2]].values

        # 小于12s的时间点需要根据上下界进行插值处理
        need_to_be_interpolated = bound_set[((bound_set[:, 1] - bound_set[:, 2]) < 2)]
        k = (data[need_to_be_interpolated[:, 1]].values - data[need_to_be_interpolated[:, 2]].values) / \
            (need_to_be_interpolated[:, 1] - need_to_be_interpolated[:, 2])
        b = (need_to_be_interpolated[:, 1] * data[need_to_be_interpolated[:, 2]].values -
             need_to_be_interpolated[:, 2] * data[need_to_be_interpolated[:, 1]].values) / \
            (need_to_be_interpolated[:, 1] - need_to_be_interpolated[:, 2])
        data_to_be_interpolated[need_to_be_interpolated[:, 0]] = k * need_to_be_interpolated[:, 0] + b

    pd.DataFrame(data_to_be_interpolated).to_csv('clean_data/house_%d/active_mains.csv' % house, mode='a', header=None)