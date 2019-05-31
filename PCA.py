# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


class PCA(nn.Module):
    def __init__(self):
        super(PCA, self).__init__()

    def forward(self, descriptors):
        # size(descriptors) = num,channel,W,H
        # 根据论文中的公式K是h x w x N
        num, channel, W, H = descriptors.size()
        k = num * W * H
        mean = descriptors.sum(dim=3).sum(dim=2).sum(dim=0) / k
        self.mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        descriptors = descriptors - self.mean

        # 根据公式将其维度变为 channel,(num_image x W x H)
        descriptors_reshape = descriptors.permute(1, 0, 2, 3) \
            .contiguous().view(channel, -1)

        cov = torch.matmul(descriptors_reshape, descriptors_reshape.t()) / k
        eigenvalue, eigenvector = torch.eig(cov, eigenvectors=True)

        self.max_vector = eigenvector[:, 0]

        # project : (num_image , W , H)
        project = torch.matmul(self.max_vector.unsqueeze(0), descriptors_reshape).view(num, W, H)

        # size(project) : (num,W,H)
        return project

    def predict(self, input):
        num, channel, W, H = input.size()
        # self.mean : (1,512,1,1)
        descriptors = input - self.mean
        descriptors_reshape = descriptors.permute(1, 0, 2, 3) \
            .contiguous().view(channel, -1)
        project = torch.matmul(self.max_vector.unsqueeze(0), descriptors_reshape).view(num, W, H)
        return project

if __name__ == '__main__':
    img = torch.randn(5, 512, 14, 14)
    pca = PCA()
    pca(img)
