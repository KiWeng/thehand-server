import numpy as np


class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        """
        F: 状态转移矩阵
        H: 观测矩阵
        Q: 状态噪声协方差矩阵
        R: 观测噪声协方差矩阵
        P: 初始状态协方差矩阵
        x: 初始状态向量
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        # 预测步骤
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # 更新步骤
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(len(self.x)) - np.dot(K, self.H), self.P)
