import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import random

class MultiModalModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(MultiModalModel, self).__init__()
        self.bert = BertModel.from_pretrained("/root/aproj/models/vbert/")
        embedding_dim = self.bert.config.hidden_size

        self.text_lstm = nn.LSTM(embedding_dim + 2, hidden_dim, batch_first=True)
        self.eye_tracking_lstm = nn.LSTM(3, hidden_dim, batch_first=True)
        # 调整全连接层的输入维度，因为我们将融合来自两个LSTM的特征
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask, word_features, eye_tracking_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs.last_hidden_state
        # if random.random() < 0.4:
        # print(text_embeddings)
        # print(word_features)
        text_features = torch.cat((text_embeddings, word_features), dim=-1)
        
        # print(text_features)

        text_out, _ = self.text_lstm(text_features)
        eye_tracking_out, _ = self.eye_tracking_lstm(eye_tracking_features)

        # 使用全局平均池化处理眼动追踪序列的输出，以匹配文本序列的长度
        # 假设eye_tracking_out的形状为[batch_size, seq_len, hidden_dim]
        eye_tracking_avg = eye_tracking_out.mean(dim=1)
        # print(eye_tracking_avg.shape)
        # print(eye_tracking_avg)
        # 扩展维度以匹配text_out的形状[batch_size, seq_len, hidden_dim]
        eye_tracking_expanded = eye_tracking_avg.unsqueeze(1).expand(-1, text_out.size(1), -1)

        # 融合文本特征和眼动追踪特征
        combined_features = torch.cat((text_out, eye_tracking_expanded), dim=2)

        # 对融合后的特征进行分类
        output = self.fc(combined_features)

        return output
