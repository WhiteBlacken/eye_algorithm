import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import random

class MultiModalModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(MultiModalModel, self).__init__()
        self.bert = BertModel.from_pretrained("/root/aproj/models/vbert/")
        embedding_dim = self.bert.config.hidden_size
        
        self.text_lstm = nn.LSTM(embedding_dim + 3, hidden_dim, batch_first=True)
        self.eye_tracking_lstm = nn.LSTM(3, hidden_dim, batch_first=True)
        # 调整全连接层的输入维度，因为我们将融合来自两个LSTM的特征
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask, word_features, eye_tracking_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs.last_hidden_state

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


class MultiModalModelWithDynamicAttention(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(MultiModalModelWithDynamicAttention, self).__init__()
        self.bert = BertModel.from_pretrained("/root/aproj/models/vbert/")
        embedding_dim = self.bert.config.hidden_size
        
        self.text_lstm = nn.LSTM(embedding_dim + 3, hidden_dim, batch_first=True)
        self.eye_tracking_lstm = nn.LSTM(3, hidden_dim, batch_first=True)
        
        self.attention = nn.Linear(hidden_dim * 2, 1)  # 注意力权重计算层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids, attention_mask, word_features, eye_tracking_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs.last_hidden_state

        text_features = torch.cat((text_embeddings, word_features), dim=-1)
        
        text_out, _ = self.text_lstm(text_features)
        eye_tracking_out, _ = self.eye_tracking_lstm(eye_tracking_features)

        # 动态确定eye_tracking_linear层的输入维度
        eye_tracking_dim = eye_tracking_out.size(2)
        self.eye_tracking_linear = nn.Linear(eye_tracking_dim, self.hidden_dim).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"eye_tracking_out.shape:{eye_tracking_out.shape}")
        # 对eye_tracking_out进行线性变换，使其维度与text_out相匹配
        eye_tracking_transformed = self.eye_tracking_linear(eye_tracking_out)

        print(f"text_out.shape:{text_out.shape}")
        print(f"eye_tracking_transformed:{eye_tracking_transformed.shape}")
        # 计算注意力权重
        attention_scores = self.attention(torch.cat((text_out, eye_tracking_transformed), dim=2))
        attention_weights = F.softmax(attention_scores, dim=1)

        # 使用注意力权重加权eye_tracking_transformed
        weighted_eye_tracking = torch.bmm(attention_weights.permute(0, 2, 1), eye_tracking_transformed)

        
        # 融合加权的eye_tracking_transformed和text_out
        combined_features = torch.cat((text_out, weighted_eye_tracking), dim=2)

        # 对融合后的特征进行分类
        output = self.fc(combined_features)

        return output


class MultiModalModelWithAttention(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(MultiModalModelWithAttention, self).__init__()
        self.bert = BertModel.from_pretrained("/root/aproj/models/vbert/")
        embedding_dim = self.bert.config.hidden_size
        
        self.text_lstm = nn.LSTM(embedding_dim + 3, hidden_dim, batch_first=True)
        self.eye_tracking_lstm = nn.LSTM(3, hidden_dim, batch_first=True)
        
        # 添加注意力层
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 调整全连接层的输入维度，因为我们将融合来自两个LSTM的特征
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask, word_features, eye_tracking_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs.last_hidden_state

        text_features = torch.cat((text_embeddings, word_features), dim=-1)
        
        text_out, _ = self.text_lstm(text_features)
        eye_tracking_out, _ = self.eye_tracking_lstm(eye_tracking_features)

        # 使用全局平均池化处理眼动追踪序列的输出，以匹配文本序列的长度
        eye_tracking_avg = eye_tracking_out.mean(dim=1)
        eye_tracking_expanded = eye_tracking_avg.unsqueeze(1).expand(-1, text_out.size(1), -1)

        # 计算注意力权重
        combined_features = torch.cat((text_out, eye_tracking_expanded), dim=2)
        attention_weights = torch.softmax(self.attention(combined_features), dim=1)
        
        # 使用注意力权重对眼动追踪特征进行加权平均
        aligned_eye_tracking = torch.sum(attention_weights * eye_tracking_expanded, dim=1)

        # 融合文本特征和对齐后的眼动追踪特征
        combined_features = torch.cat((text_out, aligned_eye_tracking.unsqueeze(1).expand(-1, text_out.size(1), -1)), dim=2)

        # 对融合后的特征进行分类
        output = self.fc(combined_features)

        return output
