from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import copy

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import Student, Teacher


class DDataset(Dataset):
    def __init__(self, matrix, label):
        super(DDataset, self).__init__()
        self.data_matrix = matrix
        self.label = label

    def __getitem__(self, item):
        return self.data_matrix[item], self.label[item]

    def __len__(self):
        return self.data_matrix.shape[0]
    
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def _read_feature_csv(path, label=None):
    df = pd.read_csv(path,  header=None)
    df = df.rename(columns={0: "apk"})
    if label is not None:
        df["label"] = label
    return df

def _rename_feature_cols(df, prefix):
    feature_cols = [c for c in df.columns if c not in ["apk", "label"]]
    new_names = {c: f"{prefix}_{i}" for i, c in enumerate(feature_cols)}
    return df.rename(columns=new_names)

def _load_one_year_fusion(benign_perm, malicious_perm, benign_api, malicious_api):
    # permission（带label）
    perm_b = _read_feature_csv(benign_perm, label=0)
    perm_m = _read_feature_csv(malicious_perm, label=1)
    perm = pd.concat([perm_b, perm_m], ignore_index=True)
    perm = _rename_feature_cols(perm, "perm")

    # api（不带label）
    api_b = _read_feature_csv(benign_api, label=None)
    api_m = _read_feature_csv(malicious_api, label=None)
    api = pd.concat([api_b, api_m], ignore_index=True)
    api = _rename_feature_cols(api, "api")

    # 按 apk 名 inner join（缺失就丢弃）
    df = perm.merge(api, on="apk", how="inner")

    y = df["label"].to_numpy(dtype=np.float32)

    perm_cols = sorted([c for c in df.columns if c.startswith("perm_")],
                       key=lambda x: int(x.split("_")[1]))
    api_cols = sorted([c for c in df.columns if c.startswith("api_")],
                      key=lambda x: int(x.split("_")[1]))

    # ✅ 核心：顺序改为 api + permission
    X = np.hstack([
        df[api_cols].to_numpy(dtype=np.float32),
        df[perm_cols].to_numpy(dtype=np.float32),
    ])

    return X, y

def load_data_permission_api():
    X_2023, y_2023 = _load_one_year_fusion(
        benign_perm="/home/cl/cl_mac_workspace/process_2023_2024/features/benign_permission_2023.csv",
        malicious_perm="/home/cl/cl_mac_workspace/process_2023_2024/features/malicious_permission_2023.csv",
        benign_api="/home/cl/cl_mac_workspace/process_2023_2024/features/benign_api_2023.csv",
        malicious_api="/home/cl/cl_mac_workspace/process_2023_2024/features/malicious_api_2023.csv",
    )

    X_2024, y_2024 = _load_one_year_fusion(
        benign_perm="/home/cl/cl_mac_workspace/process_2023_2024/features/benign_permission_2024.csv",
        malicious_perm="/home/cl/cl_mac_workspace/process_2023_2024/features/malicious_permission_2024.csv",
        benign_api="/home/cl/cl_mac_workspace/process_2023_2024/features/benign_api_2024.csv",
        malicious_api="/home/cl/cl_mac_workspace/process_2023_2024/features/malicious_api_2024.csv",
    )

    print(f"[2023] X={X_2023.shape}, y={y_2023.shape}")
    print(f"[2024] X={X_2024.shape}, y={y_2024.shape}")

    # 期望维度：api(1683) + perm(1756) = 3439
    assert X_2023.shape[1] == 3439, f"2023 特征维度不是3439，而是 {X_2023.shape[1]}"
    assert X_2024.shape[1] == 3439, f"2024 特征维度不是3439，而是 {X_2024.shape[1]}"

    return X_2023, y_2023, X_2024, y_2024





def teacher_predict(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    tea = Teacher(3439, output_dim_list=[1024, 512, 256, 128, 64, 32])
    pkl_path = "/home/cl/cl_mac_workspace/KD_QW3/Experiment_main/KD_Model/pre_per_api/pre_teacher_0.pth"
    tea.load_state_dict(torch.load(pkl_path))
    tea.to(device)
    tea.eval()

    all_preds = []
    all_probs = []  # 新增：用于存储概率值计算 AUC
    all_labels = []

    # 2. 推理获取结果
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            
            # 模型输出
            outputs = tea(inputs)
            hard_y = outputs[-1] # 假设这是最后一层的 logits
            
            # 计算概率 (如果是二分类，取索引 1 的概率)
            probs = torch.softmax(hard_y, dim=-1)
            
            # 获取预测类别
            preds = torch.argmax(hard_y, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # 提取正例 (class 1) 的概率
            all_labels.extend(labels.numpy())

    # 3. 计算基础指标
    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    # 4. 计算 AUC
    # 注意：AUC 要求 all_labels 必须是 0/1 格式
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # 防止测试集中只有一个类别时报错

    # 5. 计算 FPR (False Positive Rate)
    # confusion_matrix 结构: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # 6. 打印结果
    print(f"--- 教师模型评估结果 ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"FPR:       {fpr:.4f}")

    return acc, pre, rec, f1, auc, fpr
 

@torch.no_grad()
def student_predict(dataloader):
    # ========= 1. 固定配置（工程内常量） =========
    CKPT_PATH = "/home/cl/cl_mac_workspace/KD_QW3/Experiment_main/KD_Model/pre_per_api/pre_student_0.pth"
    INPUT_DIM = 3439
    N_CLASSES = 2
    OUTPUT_DIMS = [512, 128, 32]

    torch.backends.quantized.engine = "fbgemm"

    # ========= 2. 构建 stu1 =========
    stu1 = Student(INPUT_DIM, n_classes=N_CLASSES, output_dim_list=OUTPUT_DIMS)
    stu1.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    stu1.eval()

    # ========= 3. 拷贝并量化 =========
    model = copy.deepcopy(stu1)

    # 层融合 (Fuse)
    for i in range(len(model.model)):
        block = model.model[i]
        if isinstance(block, nn.Sequential):
            if len(block) >= 3 and isinstance(block[1], nn.Linear) and isinstance(block[2], nn.ReLU):
                torch.ao.quantization.fuse_modules(block, ['1', '2'], inplace=True)
            elif len(block) >= 2 and isinstance(block[0], nn.Linear) and isinstance(block[1], nn.ReLU):
                torch.ao.quantization.fuse_modules(block, ['0', '1'], inplace=True)

    model.eval()
    model = torch.ao.quantization.convert(model, inplace=True)

    # ========= 4. 推理获取结果 =========
    y_pred_all = []
    y_true_all = []
    y_probs_all = []  # 新增：用于存储概率值

    for data, target in dataloader:
        # 模型输出：注意这里 model 可能返回多个中间层输出，取最后一个作为分类输出
        outputs = model(data)
        hard_y = outputs[-1]
        
        # 计算概率值 (用于 AUC)
        probs = torch.softmax(hard_y, dim=-1)
        # 获取预测类别
        pred = torch.argmax(hard_y, dim=-1)

        y_pred_all.append(pred.cpu())
        y_probs_all.append(probs[:, 1].cpu()) # 获取正例概率
        y_true_all.append(target.view(-1).cpu())

    y_pred = torch.cat(y_pred_all).numpy()
    y_true = torch.cat(y_true_all).numpy()
    y_probs = torch.cat(y_probs_all).numpy()

    # ========= 5. 计算指标 =========
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1  = f1_score(y_true, y_pred, average="binary", zero_division=0)
    
    # 计算 AUC
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.0
        
    # 计算 FPR (False Positive Rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # ========= 6. 输出 =========
    print(f"--- 学生模型评估结果 ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"FPR:       {fpr:.4f}")

    return acc, pre, rec, f1, auc, fpr


if __name__ == "__main__":
    X_2023, y_2023, X_2024, y_2024 = load_data_permission_api()

    dataset_2023 = DDataset(X_2023, y_2023)
    dataset_2024 = DDataset(X_2024, y_2024)

    dataloader_2023 = torch.utils.data.DataLoader(dataset_2023, batch_size=64, shuffle=False)
    dataloader_2024 = torch.utils.data.DataLoader(dataset_2024, batch_size=64, shuffle=False)


    print("Data loaded successfully.")
    print("2023 data shape:", X_2023.shape, y_2023.shape)
    print("2024 data shape:", X_2024.shape, y_2024.shape)

    
    teacher_predict(dataloader_2023)
    student_predict(dataloader_2023)

    teacher_predict(dataloader_2024)
    student_predict(dataloader_2024)
 