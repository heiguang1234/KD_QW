from model import Student, Teacher
from indicator import Indicator_V2 as Indicator
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import itertools
import os
import sys
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score  # 新增

sys.path.append("/home/cl/cl_mac_workspace/KD_QW")

os.chdir("/home/cl/cl_mac_workspace/KD_QW")

# from knowledge_student import MultiOutputMLP
"""
1.导入数据集
2.划分为kfold 调用train Xiaorong init model
"""


########## 新增 ############
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def evaluate_epoch(student_net, teacher_net, data_loader):
    """
    在一个 data_loader 上评估：返回 (avg_loss, acc, fpr, tpr, auc)
    ROC 采用 “正类概率/score”，而不是 argmax 后的硬标签。
    """
    student_net.eval()
    teacher_net.eval()

    total_loss = 0.0
    total_n = 0

    all_y_true = []
    all_y_score = []  # 正类 score/prob

    with torch.no_grad():
        for data, target in data_loader:
            student_output = student_net(data)
            teacher_output = teacher_net(data)

            loss = distillation_loss(student_output, teacher_output, target)

            # student_output[-1] 是 logits: [N, 2]
            logits = student_output[-1]
            prob = torch.softmax(logits, dim=-1)[:, 1]  # 正类概率

            total_loss += float(loss.item()) * data.size(0)
            total_n += int(data.size(0))

            all_y_true.append(target.detach().cpu().numpy().reshape(-1))
            all_y_score.append(prob.detach().cpu().numpy().reshape(-1))

    avg_loss = total_loss / max(total_n, 1)
    y_true = np.concatenate(all_y_true, axis=0)
    y_score = np.concatenate(all_y_score, axis=0)

    # acc 用阈值 0.5 的预测
    y_pred = (y_score >= 0.5).astype(np.int64)
    acc = float((y_pred == y_true).mean())

    # auc/roc
    # 若某折某次验证集只出现单类别，roc/auc 会报错；这里做保护
    try:
        auc = float(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fpr = fpr.tolist()
        tpr = tpr.tolist()
    except Exception:
        auc = float("nan")
        fpr, tpr = [], []

    return avg_loss, acc, fpr, tpr, auc


def append_history_row(csv_path, row: dict):
    """
    追加写入一行 history 到 csv。不存在则写表头。
    为了方便画 ROC，fpr/tpr 以 json 字符串形式存储。
    """
    import json

    row = dict(row)
    row["fpr"] = json.dumps(row.get("fpr", []), ensure_ascii=False)
    row["tpr"] = json.dumps(row.get("tpr", []), ensure_ascii=False)

    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8-sig")


class DDataset(Dataset):

    def __init__(self, matrix, label):
        super(DDataset, self).__init__()
        self.data_matrix = matrix
        self.label = label

    def __getitem__(self, item):
        return self.data_matrix[item], self.label[item]

    def __len__(self):
        return self.data_matrix.shape[0]


def Save_File(file_name, k, torch_seed, batch_size, dimension, student_net, teacher_net, lr, student_str, teacher_str, epochs, path):
    with open("{}/{}/{}_student_net.txt".format(path, file_name, file_name), "a", encoding="utf-8") as file01:
        file01.write("第{}折************************************".format(k))
        file01.write(time.strftime("%Y-%m-%d %T", time.localtime(time.time())) + "\n")
        file01.write("torch_seed:{} \n".format(torch_seed))
        file01.write("普通迭代次数:{} \n".format(epochs))
        file01.write("batch size:{}, lr:{}".format(batch_size, lr) + "\n")
        file01.write("Number of features:{}".format(dimension) + "\n")
        file01.write("Student param: {}".format(student_net.output_dim_list) + "\n")
        file01.write("Teacher param: {}".format(teacher_net.output_dim_list) + "\n")
        # file.write('student 参数量:{}\n'.format(student_param))
        # file.write('teacher 参数量:{} \n'.format(teacher_naram))
        file01.write("test_____最终{}Student网络结果\n".format(file_name))
        file01.write(student_str)
        file01.write("\n")
    with open("{}/{}/{}_teacher_net.txt".format(path, file_name, file_name), "a", encoding="utf-8") as file_02:
        file_02.write("第{}折************************************".format(k))
        file_02.write(time.strftime("%Y-%m-%d %T", time.localtime(time.time())) + "\n")
        file_02.write("torch_seed:{} \n".format(torch_seed))
        file_02.write("普通迭代次数:{} \n".format(epochs))
        file_02.write("batch size:{}, lr:{}".format(batch_size, lr) + "\n")
        file_02.write("Number of features:{}".format(dimension) + "\n")
        file_02.write("Student param: {}".format(student_net.output_dim_list) + "\n")
        file_02.write("Teacher param: {}".format(teacher_net.output_dim_list) + "\n")
        file_02.write(f"test_____最终{file_name}Teacher网络结果\n")
        file_02.write(teacher_str)
        file_02.write("\n")


def kfold_training(file_name, file_2022, kfold_seed, student_param_list, teacher_param_list, lr, epochs, batch_size, torch_seed, path):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    file_path = "data_set/2021/{}.csv".format(file_name)
    file_path_2022 = "data_set/2022/{}.csv".format(file_2022)
    data_matrix = pd.read_csv(file_path, header=None).values
    data_matrix_2022 = pd.read_csv(file_path_2022, header=None).values
    # 数据集处理

    X = data_matrix[:, 1:]
    Y = data_matrix[:, 0]
    dimension = data_matrix.shape[-1] - 1

    X_2022 = data_matrix_2022[:, 1:]
    Y_2022 = data_matrix_2022[:, 0]
    dimension_2022 = data_matrix_2022.shape[-1] - 1
    # 5折，打乱顺序，随机种子使得随机可以复现
    # n_splits表示划分几等份
    # random_state随机种子数，仅当洗牌时有用，random_state数值相同时，生成的数据集一致
    kfold = KFold(n_splits=5, shuffle=True, random_state=kfold_seed)
    """2021数据集上的测试指标"""
    all_student_indicators = []
    all_teacher_indicators = []

    """2022数据集上的测试指标"""
    all_student_indicators_2022 = []
    all_teacher_indicators_2022 = []

    for k, (train_index, test_index) in enumerate(kfold.split(X)):
        # 准备dataloader
        x_2022_test = X_2022
        y_2022_test = Y_2022
        x_2022_test[x_2022_test >= 1] = 1
        x_2022_test = x_2022_test.astype(np.float32)
        y_2022_test = y_2022_test.astype(np.longlong)
        x_train = X[train_index]
        y_train = Y[train_index]
        x_test = X[test_index]
        y_test = Y[test_index]
        x_train[x_train >= 1] = 1
        x_test[x_test >= 1] = 1
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.longlong)
        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.longlong)
        train_dataset = DDataset(x_train, y_train)
        test_dataset = DDataset(x_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000000, shuffle=True)
        test_dataset_2022 = DDataset(x_2022_test, y_2022_test)
        test_loader_2022 = torch.utils.data.DataLoader(test_dataset_2022, batch_size=10000000, shuffle=True)

        # 新增：用 test_index 这一折作为“val”，用于每 epoch 记录曲线（与最终 test 一样的数据）
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, shuffle=False)

        # 每折 history 输出目录
        history_dir = os.path.join(path, file_name, "history")
        ensure_dir(history_dir)

        # 一次正常知识蒸馏，做一次test
        student_net, teacher_net = train_one_fine(
            k,
            train_loader,
            student_param_list,
            teacher_param_list,
            dimension,
            batch_size,
            lr,
            epochs,
            torch_seed,
            file_name,
            val_loader=val_loader,
            history_dir=history_dir,
        )

        """对正常训练的网络 正常量化的网络 带有中间层损失量化的网络进行2021数据集上的测试"""
        student_net_indicator, teacher_net_indicator = mytest_one(
            k, test_loader, student_net, teacher_net, dimension, batch_size, lr, torch_seed, epochs, file_name, path
        )

        all_student_indicators.append(student_net_indicator)
        all_teacher_indicators.append(teacher_net_indicator)
        """对正常训练的网络 正常量化的网络 带有中间层损失量化的网络进行2022数据集上的测试"""
        (
            student_net_indicator_2022,
            teacher_net_indicator_2022,
        ) = mytest_one_2022(k, test_loader_2022, student_net, teacher_net, dimension, batch_size, lr, torch_seed, epochs, file_2022, path)

        all_student_indicators_2022.append(student_net_indicator_2022)
        all_teacher_indicators_2022.append(teacher_net_indicator_2022)
    fine_acc = get_average_reault(all_student_indicators, all_teacher_indicators, file_name, path)
    get_average_reault(all_student_indicators_2022, all_teacher_indicators_2022, file_2022, path)
    return fine_acc


def get_indicators(pred, label, indicator_name_list):
    """
    :param pred: [0,1,1,...] ndarray(N,)已经是argmax之后的结果
    :param label: [0,1,....] ndarray(N,)真实标签
    :param indicator_name_list: ['acc', 'auc', ...]
    :return: list of indicators [acc, auc, ...]
    """
    # 创建一个指标类,形参为模型预测结果和真实标签
    ind = Indicator(pred, label)
    result_list = []
    for name in indicator_name_list:
        method_name = "get_{}()".format(name)
        # 根据获得的指标名称来ind类调用方法，获得计算结果
        result = eval("ind.{}".format(method_name))
        result_list.append(result)
    return result_list


def format_print(name_list, result):
    for name, result in zip(name_list, result):
        print("{} : {}".format(name, result))


def format_str(name_list, result):
    str = ""
    for name, result in zip(name_list, result):
        str += "{} : {} \n".format(name, result)
    return str


def distillation_loss(student_output, teacher_output, label):
    _, student_hard_y = student_output
    _, teacher_hard_y = teacher_output
    entropy_loss = F.cross_entropy(student_hard_y, label) + F.cross_entropy(teacher_hard_y, label)
    last_loss = F.mse_loss(student_hard_y, teacher_hard_y)
    return entropy_loss + last_loss


def train_one_fine(
    k,
    train_loader,
    student_param_list,
    teacher_param_list,
    dimension,
    batch_size,
    lr,
    epochs,
    torch_seed,
    file_name,
    val_loader=None,  # 新增：用于每个 epoch 记录曲线
    history_dir=None,  # 新增：保存 history 的目录
):
    student_net_fine = Student(dimension, 2, student_param_list)
    teacher_net = Teacher(dimension, 2, teacher_param_list)
    student_net_fine.load_state_dict(torch.load("Experiment_main/KD_Model/{}/pre_student_".format(file_name) + str(k) + ".pth"))
    teacher_net.load_state_dict(torch.load("Experiment_main/KD_Model/{}/pre_teacher_".format(file_name) + str(k) + ".pth"))
    student_net_fine.to("cpu")
    teacher_net.to("cpu")
    """如果模型中存在线性层和relu层 则将他们进行融合"""
    for model in student_net_fine.model:
        if isinstance(model[1], nn.Linear) and isinstance(model[2], nn.ReLU):
            model = torch.quantization.fuse_modules(model, ["1", "2"], inplace=True)
        elif isinstance(model[0], nn.Linear) and isinstance(model[1], nn.ReLU):
            model = torch.quantization.fuse_modules(model, ["0", "1"], inplace=True)

    student_net_fine.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
    torch.ao.quantization.prepare_qat(student_net_fine, inplace=True)
    optimizer = torch.optim.Adam(itertools.chain(student_net_fine.parameters(), teacher_net.parameters()), lr=lr)

    # history 输出位置
    if history_dir is not None:
        ensure_dir(history_dir)
        history_csv = os.path.join(history_dir, f"fold{k}_history.csv")
        # 可选：如果你希望每次重跑覆盖旧文件，取消注释
        # if os.path.exists(history_csv):
        #     os.remove(history_csv)
    else:
        history_csv = None

    print("开始训练不带中间层损失的第{}折 torch seed {}".format(k, torch_seed))
    print("batch size:{}, lr:{}".format(batch_size, lr))
    print("Number of features:{}".format(dimension))
    print("Student param: {}".format(student_net_fine.output_dim_list))
    print("Teacher param: {}".format(teacher_net.output_dim_list))

    for epoch in tqdm(range(epochs), desc="Processing", unit="iteration"):
        student_net_fine.train()
        teacher_net.train()

        epoch_loss_sum = 0.0
        epoch_n = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            student_output = student_net_fine(data)
            teacher_output = teacher_net(data)

            loss = distillation_loss(student_output, teacher_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_sum += float(loss.item()) * data.size(0)
            epoch_n += int(data.size(0))

        train_loss = epoch_loss_sum / max(epoch_n, 1)

        if epoch > 9:
            student_net_fine.apply(torch.ao.quantization.disable_observer)

        # ===== 每个 epoch 做一次验证并记录 =====
        if val_loader is not None and history_csv is not None:
            val_loss, val_acc, fpr, tpr, auc = evaluate_epoch(student_net_fine, teacher_net, val_loader)
            append_history_row(
                history_csv,
                {
                    "fold": k,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auc": auc,
                    "fpr": fpr,
                    "tpr": tpr,
                    "lr": lr,
                    "batch_size": batch_size,
                    "torch_seed": torch_seed,
                    "file_name": file_name,
                },
            )
        elif history_csv is not None:
            # 没 val_loader 时也记录 train_loss，方便画 loss 曲线
            append_history_row(
                history_csv,
                {
                    "fold": k,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": float("nan"),
                    "val_acc": float("nan"),
                    "val_auc": float("nan"),
                    "fpr": [],
                    "tpr": [],
                    "lr": lr,
                    "batch_size": batch_size,
                    "torch_seed": torch_seed,
                    "file_name": file_name,
                },
            )

    student_net_fine.eval()
    student_net_fine = torch.ao.quantization.convert(student_net_fine, inplace=True)

    torch.save(student_net_fine.state_dict(), "Experiment_main/Fine_Model/{}/pre_student_".format(file_name) + str(k) + ".pth")
    torch.save(teacher_net.state_dict(), "Experiment_main/Fine_Model/{}/pre_teacher_".format(file_name) + str(k) + ".pth")

    return student_net_fine, teacher_net


def mytest_one(k, test_loader, student_net, teacher_net, dimension, batch_size, lr, torch_seed, epoch, file_name, path):
    print("开始test 第{}折".format(k))
    #  模型进入验证
    student_net.eval()
    teacher_net.eval()
    name_list = ["acc", "precision", "recall", "fmeature", "specific", "tpr", "fpr", "mcc", "auc"]
    student_indicators = []
    teacher_indicators = []
    for batch_idx, (data, target) in enumerate(test_loader):
        student_output = student_net(data)[-1]
        teacher_output = teacher_net(data)[-1]
        student_output = torch.argmax(student_output, dim=-1, keepdim=False)
        teacher_output = torch.argmax(teacher_output, dim=-1, keepdim=False)
        # get_indicators接受的参数分别是模型的输出结果，真实标签和参数序列
        # 此处获得的student_indicators是一个指标的综合结果集
        student_indicators = get_indicators(student_output.cpu().numpy(), target.cpu().numpy().reshape(-1), name_list)
        teacher_indicators = get_indicators(teacher_output.cpu().numpy(), target.cpu().numpy().reshape(-1), name_list)
    print("test_____最终student网络结果")
    # format_print利用zip分别打印出参数和对应的结果
    format_print(name_list, student_indicators)
    # format_str将所有的结果存到一个字符串里
    student_str = format_str(name_list, student_indicators)
    print("test____最终teacher网络结果")
    format_print(name_list, teacher_indicators)
    teacher_str = format_str(name_list, teacher_indicators)
    Save_File(file_name, k, torch_seed, batch_size, dimension, student_net, teacher_net, lr, student_str, teacher_str, epoch, path)
    return student_indicators, teacher_indicators


def mytest_one_2022(k, test_loader, student_net, teacher_net, dimension, batch_size, lr, torch_seed, epoch, file_name, path):
    print("开始test 第{}折".format(k))
    #  模型进入验证
    student_net.eval()
    teacher_net.eval()
    name_list = ["acc", "precision", "recall", "fmeature", "specific", "tpr", "fpr", "mcc", "auc"]
    student_indicators = []
    teacher_indicators = []
    for batch_idx, (data, target) in enumerate(test_loader):
        student_output = student_net(data)[-1]
        teacher_output = teacher_net(data)[-1]
        student_output = torch.argmax(student_output, dim=-1, keepdim=False)
        teacher_output = torch.argmax(teacher_output, dim=-1, keepdim=False)
        # get_indicators接受的参数分别是模型的输出结果，真实标签和参数序列
        # 此处获得的student_indicators是一个指标的综合结果集
        student_indicators = get_indicators(student_output.cpu().numpy(), target.cpu().numpy().reshape(-1), name_list)
        teacher_indicators = get_indicators(teacher_output.cpu().numpy(), target.cpu().numpy().reshape(-1), name_list)
    print("test_____最终student_2022网络结果")
    # format_print利用zip分别打印出参数和对应的结果
    format_print(name_list, student_indicators)
    # format_str将所有的结果存到一个字符串里
    student_str = format_str(name_list, student_indicators)
    print("test____最终teacher_2022网络结果")
    format_print(name_list, teacher_indicators)
    teacher_str = format_str(name_list, teacher_indicators)
    Save_File(file_name, k, torch_seed, batch_size, dimension, student_net, teacher_net, lr, student_str, teacher_str, epoch, path)
    return student_indicators, teacher_indicators


def get_average_reault(all_student_indicators, all_teacher_indicators, file_name, path):
    student_mean = np.array(all_student_indicators).mean(axis=0)
    teacher_mean = np.array(all_teacher_indicators).mean(axis=0)
    name_list = ["acc", "precision", "recall", "fmeature", "specific", "tpr", "fpr", "mcc", "auc"]
    student_mean_str = format_str(name_list, student_mean)
    teacher_mean_str = format_str(name_list, teacher_mean)
    print("student 平均结果************************************")
    format_print(name_list, student_mean)
    print("teacher 平均结果************************************")
    format_print(name_list, teacher_mean)
    with open("{}/{}/{}_student_net.txt".format(path, file_name, file_name), "a", encoding="utf8") as file:
        file.write("student 平均结果\n")
        file.write(student_mean_str)
    with open("{}/{}/{}_teacher_net.txt".format(path, file_name, file_name), "a", encoding="utf8") as file:
        file.write("teacher 平均结果\n")
        file.write(teacher_mean_str)
    return student_mean[0]


if __name__ == "__main__":
    # for file_name in ['pre_API_1683','pre_per_1757', 'pre_per_api']:
    # for file_name in ['pre_API_1683', 'pre_per_api']:
    for file_name in ["pre_API_1683"]:
        kfold_training(
            file_name=file_name,
            file_2022=file_name + "_2022",
            kfold_seed=2,
            student_param_list=[512, 128, 32],
            teacher_param_list=[1024, 512, 256, 128, 64, 32],
            lr=0.001,
            epochs=40,
            batch_size=256,
            torch_seed=0,
            path="Experiment_main/Fine_result",
        )
