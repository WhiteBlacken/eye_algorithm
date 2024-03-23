# Data处理
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import torch

class TextEyeTrackingDataset(Dataset):
    def __init__(self, csv_file, max_length=512):
        self.data_frame = pd.read_csv(csv_file)
        self.tokenizer = BertTokenizer.from_pretrained("/root/aproj/models/vbert/")
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 假设csv中的列名分别为"text_sequences", "eye_tracking_sequences", "labels"
        text_seq = np.array(eval(self.data_frame.iloc[idx]['text_sequences']))
        eye_tracking_seq = np.array(eval(self.data_frame.iloc[idx]['eye_tracking_sequences']))
        labels = np.array(eval(self.data_frame.iloc[idx]['labels']))

        words = text_seq[:, 1]  # 获取单词
        input_ids, attention_mask = self.encode_words(words, self.tokenizer, self.max_length, self.pad_token_id)
        # print(len(input_ids) == len(attention_mask))
        # print(input_ids)

        word_feature_seq = text_seq[:, 2:].astype(np.float32)  # 假设坐标和大小特征从第2列开始
        # 计算中心点的x坐标
        center_x = word_feature_seq[:, 0] + word_feature_seq[:, 2] / 2.0
        # 计算中心点的y坐标
        center_y = word_feature_seq[:, 1] + word_feature_seq[:, 3] / 2.0
        # 将中心点的x和y坐标组合成一个新的numpy数组
        center_points = np.vstack((center_x, center_y)).T  # .T 是转置操作，使得数组的形状与期望的对齐
        # print(center_points)
        # word_features = (center_points - center_points.mean(axis=0)) /center_points.std(axis=0)
        word_features = center_points
        # print(f"word_features.shape:{word_features.shape}")
        word_features = self.pad_features(word_features, 512, pad_value=0.0)

        # print(f"word_features.shape:{word_features.shape}")
    

        # 标准化眼动追踪数据（这里仅为示例，实际应用中应根据数据分布进行标准化）
        eye_tracking_features = (eye_tracking_seq - eye_tracking_seq.mean(axis=0)) / eye_tracking_seq.std(axis=0)
       
        # print(f"label.shape:{labels.shape}")
        labels = self.pad_labels(labels, 512)
        

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'word_features': torch.tensor(word_features, dtype=torch.float),
            'eye_tracking_features': torch.tensor(eye_tracking_features, dtype=torch.float),
            'labels': torch.tensor(labels, dtype=torch.float)
        }
    
    def encode_words(self, words, tokenizer, max_length, default_input_id):
        """
        对单词序列进行编码。

        参数:
        - words: 单词序列，列表形式。
        - tokenizer: BertTokenizer实例。
        - max_length: 最大序列长度。
        - default_input_id: 无法编码单词时使用的默认input_id。

        返回:
        - input_ids: 编码后的input_ids。
        - attention_mask: 对应的attention_mask。
        """
        input_ids = []
        attention_mask = []

        for word in words:
            # 对每个单词进行编码
            encoded_word = tokenizer.encode(word, add_special_tokens=False)
            # print(encoded_word)
            if len(encoded_word) > 0:
                # 只保留第一个子词的input_id
                input_ids.append(encoded_word[0])
                attention_mask.append(1)  # 对应的attention_mask设置为1
            else:
                # 如果分词器无法处理该单词，分配默认值
                input_ids.append(default_input_id)
                attention_mask.append(0)

        # 确保input_ids不超过最大长度
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        else:
            # 如果长度不足，用[PAD]填充,如果不足512可能产生nan
            input_ids += [default_input_id] * (max_length - len(input_ids))
            attention_mask += [0] * (max_length - len(attention_mask))

        return input_ids, attention_mask

    def pad_features(self, features, pad_length, pad_value=0.0):
        """
        对特征进行填充以达到指定的长度。

        参数:
        - features: 要填充的特征，假设为一个二维数组，形状为[feature_length, num_features]。
        - pad_length: 目标长度。
        - pad_value: 用于填充的值。

        返回:
        - 填充后的特征数组。
        """
        # 计算需要填充的长度
        padding_size = pad_length - features.shape[0]  # 注意这里改为features.shape[0]
        if padding_size > 0:
            # 创建填充数组，注意填充的维度是第一个维度（即行）
            padding = np.full((padding_size, features.shape[1]), pad_value, dtype=features.dtype)
            # 将原始特征和填充数组拼接，注意拼接的轴是第一个维度（即行）
            padded_features = np.concatenate((features, padding), axis=0)
        else:
            # 如果不需要填充，或需要截断，注意截断的维度是第一个维度（即行）
            padded_features = features[:pad_length]

        return padded_features

    def pad_labels(self, labels, pad_length, pad_value=-100):  # 使用-100作为填充值，常用于忽略某些损失计算
        """
        对标签进行填充以达到指定的长度。

        参数:
        - labels: 要填充的标签，假设为一维数组。
        - pad_length: 目标长度。
        - pad_value: 用于填充的值。

        返回:
        - 填充后的标签数组。
        """
        padding_size = pad_length - len(labels)
        if padding_size > 0:
            # 创建填充数组
            padding = np.full(padding_size, pad_value, dtype=labels.dtype)
            # 将原始标签和填充数组拼接
            padded_labels = np.concatenate((labels, padding), axis=0)
        else:
            # 如果不需要填充，或需要截断
            padded_labels = labels[:pad_length]

        return padded_labels
