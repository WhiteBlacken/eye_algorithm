import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

# torch.autograd.set_detect_anomaly(True)
def evaluate_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    predictions = []
    actuals = []
    with torch.no_grad():  # 在评估过程中不计算梯度
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_features = batch['word_features'].to(device)
            eye_tracking_features = batch['eye_tracking_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, word_features, eye_tracking_features)
            outputs = torch.sigmoid(outputs.squeeze(dim=2).view(-1))  # 使用sigmoid激活函数获取概率,因为要算准确率
            predictions.extend(outputs.tolist())
            actuals.extend(labels.view(-1).tolist())

    # 计算性能指标
    # accuracy, auc, precision, recall = calculate_metrics(actuals, predictions)

     # 去除padding
    idx = len(actuals)
    for i, item in enumerate(actuals):
        if item == -100:
            idx = i
            break
    
    actuals = actuals[:idx]
    predictions = predictions[:idx]

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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_features = batch['word_features'].to(device)
            eye_tracking_features = batch['eye_tracking_features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask,  word_features, eye_tracking_features)
            outputs = outputs.squeeze(dim=2).view(-1, 1)
            labels = labels.view(-1, 1)

            # 去除padding
            idx = len(labels)
            for i, item in enumerate(labels):
                if item == - 100:
                    idx = i
                    break
            outputs = outputs[:idx]
            labels = labels[:idx]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 在测试集上评估模型
        accuracy, auc, precision, recall = evaluate_model(model, test_loader, device)
        # accuracy, auc, precision, recall = 0, 0 , 0 , 0
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy}, AUC: {auc}, Precision: {precision}, Recall: {recall}')


def calculate_metrics(target, predict_proba):
    """
    计算准确率、精确率、召回率和AUC。
    
    参数:
    - target: 目标值列表，由0和1组成。
    - predict_proba: 预测的正类概率列表。
    
    返回:
    - accuracy: 准确率。
    - precision: 精确率。
    - recall: 召回率。
    - auc: AUC值。
    """
    # 去除padding
    idx = len(target)
    for i, item in enumerate(target):
        if item == - 100:
            idx = i
            break
    
    target = target[:idx]
    print(set(target))
    predict_proba = predict_proba[:idx]
    # 将概率预测转换为0和1的预测
    predict = [1 if proba >= 0.5 else 0 for proba in predict_proba]
    
    # 计算TP, TN, FP, FN
    true_positive = sum(p == 1 and t == 1 for p, t in zip(predict, target))
    true_negative = sum(p == 0 and t == 0 for p, t in zip(predict, target))
    false_positive = sum(p == 1 and t == 0 for p, t in zip(predict, target))
    false_negative = sum(p == 0 and t == 1 for p, t in zip(predict, target))
    
    # 计算accuracy, precision, recall
    accuracy = (true_positive + true_negative) / len(predict)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    # 计算AUC
    auc = roc_auc_score(target, predict_proba)
    
    return accuracy, auc, precision, recall