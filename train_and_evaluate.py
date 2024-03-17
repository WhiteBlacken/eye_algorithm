import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

def evaluate_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    predictions = []
    actuals = []
    with torch.no_grad():  # 在评估过程中不计算梯度
        for batch in test_loader:
            word_indices = batch['word_indices'].to(device)
            word_features = batch['word_features'].to(device)
            eye_tracking_features = batch['eye_tracking_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(word_indices, word_features, eye_tracking_features)
            outputs = torch.sigmoid(outputs.squeeze(dim=2).view(-1))  # 使用sigmoid激活函数获取概率,因为要算准确率
            predictions.extend(outputs.tolist())
            actuals.extend(labels.view(-1).tolist())

    # 计算性能指标
    predictions = np.round(predictions)  # 将概率值转换为0或1
    accuracy = round(accuracy_score(actuals, predictions), 2)
    auc = round(roc_auc_score(actuals, predictions), 2)
    precision = round(precision_score(actuals, predictions), 2)
    recall = round(recall_score(actuals, predictions), 2)

    return accuracy, auc, precision, recall

def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs=10):
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0
        for batch in train_loader:
            word_indices = batch['word_indices'].to(device)
            word_features = batch['word_features'].to(device)
            eye_tracking_features = batch['eye_tracking_features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(word_indices, word_features, eye_tracking_features)
            outputs = outputs.squeeze(dim=2).view(-1, 1)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 在测试集上评估模型
        accuracy, auc, precision, recall = evaluate_model(model, test_loader, device)
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy}, AUC: {auc}, Precision: {precision}, Recall: {recall}')