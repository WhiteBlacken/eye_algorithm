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


class MultiModalModelWithAttention(nn.Module):
    def __init__(self, hidden_dim, output_dim, show_dimension=False):
        super(MultiModalModelWithAttention, self).__init__()
        self.bert = BertModel.from_pretrained("/root/aproj/models/vbert/")
        embedding_dim = self.bert.config.hidden_size
        
        self.text_lstm = nn.LSTM(embedding_dim + 3, hidden_dim, batch_first=True)
        self.eye_tracking_lstm = nn.LSTM(3, hidden_dim, batch_first=True)
        
        # 添加注意力层
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 调整全连接层的输入维度，因为我们将融合来自两个LSTM的特征
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.show_dimension = show_dimension

        self.attentionAlignmentLayer = AttentionAlignment(128, 128, 64)

        if show_dimension:
            print(f"hidden_dim:{hidden_dim}")
            print(f"output_dim:{output_dim}")
        

    def forward(self, input_ids, attention_mask, word_features, eye_tracking_features):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs.last_hidden_state
        text_features = torch.cat((text_embeddings, word_features), dim=-1)
        
        text_out, _ = self.text_lstm(text_features)
        eye_tracking_out, _ = self.eye_tracking_lstm(eye_tracking_features)
    
        if self.show_dimension:
            print(f"input_ids:{input_ids.shape}")
            print(f"attention_mask:{attention_mask.shape}")
            print(f"word_features:{word_features.shape}")
            print(f"eye_tracking_features:{eye_tracking_features.shape}")
            
            print(f"text_embeddings:{text_embeddings.shape}")
            print(f"text_features:{text_features.shape}")
            print(f"text_out:{text_out.shape}")
            print(f"eye_tracking_out:{eye_tracking_out.shape}")
            
        output = self.attentionAlignmentLayer(text_out, eye_tracking_out)
        if self.show_dimension:
            print(f"output.shape:{output.shape}")
        # sys.exit(0)
        return output

class AttentionAlignment(nn.Module):
    def __init__(self, word_dim, fixation_dim, hidden_dim):
        super(AttentionAlignment, self).__init__()
        self.word_dim = word_dim
        self.fixation_dim = fixation_dim
        self.hidden_dim = hidden_dim
        
        # 添加线性层调整fixation_seq的维度
        self.linear_fixation = nn.Linear(fixation_dim, word_dim)
        # 定义注意力权重计算层
        self.attention = nn.MultiheadAttention(embed_dim=word_dim, num_heads=1)
        # 定义全连接层
        self.fc = nn.Linear(word_dim * 2, hidden_dim)
        # 定义输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word_seq, fixation_seq):
        # 调整fixation_seq的维度
        # adjusted_fixation_seq = self.linear_fixation(fixation_seq)
        adjusted_fixation_seq = fixation_seq
        # 计算注意力权重
        attention_weights, _ = self.attention(word_seq.permute(1, 0, 2), adjusted_fixation_seq.permute(1, 0, 2), adjusted_fixation_seq.permute(1, 0, 2))
        # print(attention_weights.shape)
        # 对齐word序列和fixation序列
        aligned_word = torch.cat((word_seq, attention_weights.permute(1, 0, 2)), dim=2)
        # 使用全连接层进行特征融合
        fused_features = torch.relu(self.fc(aligned_word))      
        # 输出层
        output = self.sigmoid(self.output_layer(fused_features))
        
        # print(f"adjusted_fixation_seq:{adjusted_fixation_seq.shape}")
        # print(f"word_seq.permute(1, 0, 2):{word_seq.permute(1, 0, 2).shape}")
        # print(f"adjusted_fixation_seq.permute(1, 0, 2):{adjusted_fixation_seq.permute(1, 0, 2).shape}")
        # print(f"attention_weights:{attention_weights.shape}")
        # print(f"aligned_word:{aligned_word.shape}")
        # print(f"fused_features.shape")
        # print(f"output:{output.shape}")
        return output

class MultiModalModelWrapper(nn.Module):
    def __init__(self, hidden_dim, output_dim, pos_weight):
        super().__init__()
        self.model = MultiModalModelWithAttention(hidden_dim, output_dim, show_dimension=False)
        self.pos_weight = pos_weight

    def forward(self, text_input, visual_input):
        return self.model(text_input, visual_input)

    def fit(self, train_loader, test_loader, optimizer, criterion, device, epochs):
        train_model(self.model, train_loader, test_loader, optimizer, criterion, device, tensorboard_dir, epochs)
