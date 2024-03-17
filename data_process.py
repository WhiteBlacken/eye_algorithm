# Data处理
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch

class TextEyeTrackingDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 假设csv中的列名分别为"text_sequences", "eye_tracking_sequences", "labels"
        text_seq = np.array(eval(self.data_frame.iloc[idx]['text_sequences']))
        eye_tracking_seq = np.array(eval(self.data_frame.iloc[idx]['eye_tracking_sequences']))
        labels = np.array(eval(self.data_frame.iloc[idx]['labels']))

        # 仅使用单词索引作为文本特征的示例，实际应用中应使用词嵌入
        word_indices = text_seq[:, 0]  # 假设单词索引在第0列
        word_indices = [int(i) for i in word_indices]
        word_features = text_seq[:, 2:].astype(np.float32)  # 假设坐标和大小特征从第2列开始

        # 标准化眼动追踪数据（这里仅为示例，实际应用中应根据数据分布进行标准化）
        eye_tracking_features = (eye_tracking_seq - eye_tracking_seq.mean(axis=0)) / eye_tracking_seq.std(axis=0)

        return {
            'word_indices': torch.tensor(word_indices, dtype=torch.long),
            'word_features': torch.tensor(word_features, dtype=torch.float),
            'eye_tracking_features': torch.tensor(eye_tracking_features, dtype=torch.float),
            'labels': torch.tensor(labels, dtype=torch.float)
        }
