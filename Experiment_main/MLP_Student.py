from model import Student
from indicator import Indicator_V2 as Indicator
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import time
import itertools
import os
import sys
from tqdm import tqdm

sys.path.append('/home/cl/cl_mac_workspace/KD_QW')

os.chdir('/home/cl/cl_mac_workspace/KD_QW')

# from knowledge_student import MultiOutputMLP
'''
1.导入数据集
2.划分为kfold 调用train Xiaorong init model
'''


class DDataset(Dataset):

    def __init__(self, matrix, label):
        super(DDataset, self).__init__()
        self.data_matrix = matrix
        self.label = label

    def __getitem__(self, item):
        return self.data_matrix[item], self.label[item]

    def __len__(self):
        return self.data_matrix.shape[0]


def Save_File(file_name, k, torch_seed, batch_size, dimension, student_net, lr, student_str, epochs, path):
    with open('{}/{}/{}_student_net.txt'.format(path, file_name, file_name), 'a', encoding='utf-8') as file01:
        file01.write('第{}折************************************'.format(k))
        file01.write(
            time.strftime('%Y-%m-%d %T', time.localtime(time.time())) + '\n')
        file01.write('torch_seed:{} \n'.format(torch_seed))
        file01.write('普通迭代次数:{} \n'.format(epochs))
        file01.write('batch size:{}, lr:{}'.format(batch_size, lr) + '\n')
        file01.write('Number of features:{}'.format(dimension) + '\n')
        file01.write('student param: {}'.format(
            student_net.output_dim_list) + '\n')
        file01.write('test_____最终{}student网络结果\n'.format(file_name))
        file01.write(student_str)
        file01.write('\n')


def kfold_training(file_name, kfold_seed, student_param_list, lr, epochs, batch_size, torch_seed, path):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    file_path = 'data_set/2021/{}.csv'.format(file_name)
    data_matrix = pd.read_csv(file_path, header=None).values

    # 数据集处理
    X = data_matrix[:, 1:]
    Y = data_matrix[:, 0]
    dimension = data_matrix.shape[-1] - 1

    # 5折，打乱顺序，随机种子使得随机可以复现
    # n_splits表示划分几等份
    # random_state随机种子数，仅当洗牌时有用，random_state数值相同时，生成的数据集一致
    kfold = KFold(n_splits=5, shuffle=True, random_state=kfold_seed)
    '''2021数据集上的测试指标'''
    all_student_indicators = []

    for k, (train_index, test_index) in enumerate(kfold.split(X)):
        # 准备dataloader
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

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=10000000,
                                                  shuffle=True)
        # 一次正常知识蒸馏，做一次test
        student_net = train_one(k, train_loader,
                                student_param_list, dimension,
                                batch_size, lr, epochs,
                                torch_seed)

        '''对正常训练的网络 正常量化的网络 带有中间层损失量化的网络进行2021数据集上的测试'''
        student_net_indicator = mytest_one(
            k, test_loader, student_net, dimension, batch_size, lr, torch_seed, epochs, file_name, path)

        all_student_indicators.append(student_net_indicator)

    mlp_acc = get_average_reault(all_student_indicators, file_name, path)
    return mlp_acc


def get_indicators(pred, label, indicator_name_list):
    '''
    :param pred: [0,1,1,...] ndarray(N,)已经是argmax之后的结果
    :param label: [0,1,....] ndarray(N,)真实标签
    :param indicator_name_list: ['acc', 'auc', ...]
    :return: list of indicators [acc, auc, ...]
    '''
    # 创建一个指标类,形参为模型预测结果和真实标签
    ind = Indicator(pred, label)
    result_list = []
    for name in indicator_name_list:
        method_name = 'get_{}()'.format(name)
        # 根据获得的指标名称来ind类调用方法，获得计算结果
        result = eval('ind.{}'.format(method_name))
        result_list.append(result)
    return result_list


def format_print(name_list, result):
    for name, result in zip(name_list, result):
        print('{} : {}'.format(name, result))


def format_str(name_list, result):
    str = ''
    for name, result in zip(name_list, result):
        str += '{} : {} \n'.format(name, result)
    return str


def distillation_loss(student_output, label):
    _, student_y = student_output
    entropy_loss = F.cross_entropy(student_y, label)
    return entropy_loss


def train_one(k, train_loader, student_param_list,
              dimension, batch_size, lr, epochs, torch_seed):
    student_net = Student(dimension, 2, student_param_list)
    student_net.to('cuda')
    # 采用默认参数
    optimizer = torch.optim.Adam(
        itertools.chain(student_net.parameters()), lr=lr)
    print('开始训练第{}折 torch seed'.format(k, torch_seed))
    print('batch size:{}, lr:{}'.format(batch_size, lr))
    print('Number of features:{}'.format(dimension))
    print('student param: {}'.format(student_net.output_dim_list))
    for epoch in tqdm(range(epochs), desc="Processing", unit="iteration"):
        # print('epoch: {}'.format(epoch + 1))
        # 表示模型处于训练模式
        student_net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.clone().detach().to('cuda')
            target = target.clone().detach().to('cuda')
            student_output = student_net(data)
            loss = distillation_loss(student_output, label=target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # torch.save(student_net.state_dict(),
    #            'Experiment_main/MLP_Model/{}/pre_student_'.format(file_name) + str(k) + '.pth')
    student_net.to('cpu')
    return student_net


def mytest_one(k, test_loader, student_net, dimension, batch_size, lr, torch_seed, epoch,
               file_name, path):
    print('开始test 第{}折'.format(k))
    #  模型进入验证
    student_net.eval()
    name_list = [
        'acc', 'precision', 'recall', 'fmeature', 'specific', 'tpr', 'fpr',
        'mcc', 'auc'
    ]
    student_indicators = []
    for batch_idx, (data, target) in enumerate(test_loader):
        student_output = student_net(data)[-1]
        student_output = torch.argmax(student_output, dim=-1, keepdim=False)
        # get_indicators接受的参数分别是模型的输出结果，真实标签和参数序列
        # 此处获得的student_indicators是一个指标的综合结果集
        student_indicators = get_indicators(student_output.cpu().numpy(),
                                            target.cpu().numpy().reshape(-1),
                                            name_list)
    print('test_____最终student网络结果')
    # format_print利用zip分别打印出参数和对应的结果
    format_print(name_list, student_indicators)
    # format_str将所有的结果存到一个字符串里
    student_str = format_str(name_list, student_indicators)

    Save_File(file_name, k, torch_seed,
              batch_size, dimension, student_net, lr, student_str, epoch, path)
    return student_indicators


def get_average_reault(all_student_indicators, file_name, path):
    student_mean = np.array(all_student_indicators).mean(axis=0)
    name_list = [
        'acc', 'precision', 'recall', 'fmeature', 'specific', 'tpr', 'fpr',
        'mcc', 'auc'
    ]
    student_mean_str = format_str(name_list, student_mean)
    print('student 平均结果************************************')
    format_print(name_list, student_mean)
    with open('{}/{}/{}_student_net.txt'.format(path, file_name, file_name),
              'a',
              encoding='utf8') as file:
        file.write('student 平均结果\n')
        file.write(student_mean_str)
    return student_mean[0]


if __name__ == '__main__':
    # for file_name in ['pre_API_1683','pre_per_1757', 'pre_per_api']:
    for file_name in [ 'pre_per_1757']:
        kfold_training(
            file_name=file_name,
            kfold_seed=2,
            student_param_list=[512, 128, 32],
            lr=0.001,  
            epochs=80,  
            batch_size=256,  
            torch_seed=0,  
            path='Experiment_main/MLP_result')
