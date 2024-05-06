import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

# torch.autograd.set_detect_anomaly(True)
def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    predictions = []
    actuals = []
    total_loss = 0
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

    outputs = torch.tensor(predictions, device=device)  # 将predictions转换为张量
    labels = torch.tensor(actuals, device=device)  # 将actuals转换为张量
    loss = criterion(outputs, labels)
    total_loss += loss.item()
    predictions = np.round(predictions)  # 将概率值转换为0或1
    window_size = 3
    
    # 将predictions转换为0或1
    predictions_binary = [1 if pred == 1 and any(actuals[i:i+window_size]) == 1 else 0 for i, pred in enumerate(predictions)]
    
    accuracy = round(accuracy_score(actuals, predictions_binary), 2)
    precision = round(precision_score(actuals, predictions_binary), 2)
    recall = round(recall_score(actuals, predictions_binary), 2)
    f1_score = round(2 * (precision * recall) / (precision + recall), 2) if (precision + recall) > 0 else 0
    try:
        auc = round(roc_auc_score(actuals, predictions_binary), 2)
    except:
        auc = 0.5
        
    return loss, accuracy, auc, precision, recall, f1_score

def train_model(model, train_loader, test_loader, optimizer, criterion, device, tensorboard_dir, epochs=10):
    writer = SummaryWriter(tensorboard_dir)
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

        # 在训练集上评估模型
        train_loss, train_accuracy, train_auc, train_precision, train_recall, train_f1_score = evaluate_model(model, train_loader, criterion, device)
        # 在测试集上评估模型
        test_loss, test_accuracy, test_auc, test_precision, test_recall, test_f1_score = evaluate_model(model, test_loader, criterion, device)
        # accuracy, auc, precision, recall = 0, 0 , 0 , 0
        print(f"-----Epoch {epoch + 1}-------")
        print(f'Train metric in Epoch {epoch+1}, Loss: {train_loss/len(train_loader)}, Accuracy: {train_accuracy}, AUC: {train_auc}, Precision: {train_precision}, Recall: {train_recall}, F1-score: {train_f1_score}')
        print(f'Test metric in Epoch {epoch+1}, Loss: {test_loss/len(test_loader)}, Accuracy: {test_accuracy}, AUC: {test_auc}, Precision: {test_precision}, Recall: {test_recall}, F1-score: {test_f1_score}')
        
        # 在每个epoch结束时将训练和测试的损失以及性能指标写入TensorBoard
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)
        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('F1-score/train', train_f1_score, epoch)

        writer.add_scalar('Loss/test', test_loss/len(test_loader), epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        writer.add_scalar('AUC/test', test_auc, epoch)
        writer.add_scalar('Precision/test', test_precision, epoch)
        writer.add_scalar('Recall/test', test_recall, epoch)
        writer.add_scalar('F1-score/test', test_f1_score, epoch)

    # 训练结束时关闭SummaryWriter对象
    writer.close()

