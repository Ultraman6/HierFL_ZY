import copy
import torch
from torch import nn
# 权重平均聚合
# def average_weights(w, s_num):
#     #copy the first client's weights
#     total_sample_num = sum(s_num)
#     # print(s_num)
#     temp_sample_num = s_num[0]
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():  #the nn layer loop
#         for i in range(1, len(w)):   #the client loop
#             # w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
#             # result type Float can't be cast to the desired output type Long
#             w_avg[k] = w_avg[k] + torch.mul(w[i][k], s_num[i] / temp_sample_num)
#         w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
#     return w_avg

def average_weights(w, s_num):
    total_sample_num = sum(s_num)
    scale_factors = [num / total_sample_num for num in s_num]

    # 初始化平均权重
    w_avg = {k: torch.zeros_like(w[0][k]) for k in w[0].keys()}

    # 累加每个客户端的权重
    for i, w_client in enumerate(w):
        for k in w_client.keys():
            w_avg[k] += w_client[k] * scale_factors[i]

    return w_avg
