# TODO 计算loss的相关函数
import torch
import torch.nn as nn


def get_gram_matrix(f_map):  #
    n, c, h, w = f_map.shape
    f_map = f_map.reshape(n, c, h * w)
    gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
    return gram_matrix


def content_loss(base, combination):
    return nn.MSELoss()(base, combination)


def style_loss(style, combination, batch_size, image_size=(256, 256)):
    S = get_gram_matrix(style)
    S = S.expand(
        batch_size,
        S.shape[1],
        S.shape[2]
    )  # 将S的维度扩充到C
    C = get_gram_matrix(combination)
    channel = 3
    size = image_size[0] * image_size[1]
    return nn.MSELoss()(S, C) # / (4.0 * (channel**2) * (size**2))














