from model import Student, Teacher
from summary import summary
from darts import HardMLPBlock as s_mlpNet
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


def Save_File(path, file_name, k, torch_seed, batch_size, dimension, student_net, teacher_net, lr, student_str,
              teacher_str, epochs, quant_epoch):
    with open('{}/{}_student_net.txt'.format(path, file_name), 'a', encoding='utf-8') as file01:
        file01.write('第{}折************************************'.format(k))
        file01.write(
            time.strftime('%Y-%m-%d %T', time.localtime(time.time())) + '\n')
        file01.write('torch_seed:{} \n'.format(torch_seed))
        file01.write('普通迭代次数:{} \n'.format(epochs))
        file01.write('量化迭代次数:{} \n'.format(quant_epoch))
        file01.write('batch size:{}, lr:{}'.format(batch_size, lr) + '\n')
        file01.write('Number of features:{}'.format(dimension) + '\n')
        file01.write('Student param: {}'.format(student_net.output_dim_list) +
                     '\n')
        file01.write('Teacher param: {}'.format(teacher_net.output_dim_list) +
                     '\n')
        # file.write('student 参数量:{}\n'.format(student_param))
        # file.write('teacher 参数量:{} \n'.format(teacher_naram))
        file01.write('test_____最终{}Student网络结果\n'.format(file_name))
        file01.write(student_str)
        file01.write('\n')
    with open('{}/{}_teacher_net.txt'.format(path, file_name), 'a', encoding='utf-8') as file_02:
        file_02.write('第{}折************************************'.format(k))
        file_02.write(
            time.strftime('%Y-%m-%d %T', time.localtime(time.time())) + '\n')
        file_02.write('torch_seed:{} \n'.format(torch_seed))
        file_02.write('普通迭代次数:{} \n'.format(epochs))
        file_02.write('量化迭代次数:{} \n'.format(quant_epoch))
        file_02.write('batch size:{}, lr:{}'.format(batch_size, lr) + '\n')
        file_02.write('Number of features:{}'.format(dimension) + '\n')
        file_02.write('Student param: {}'.format(student_net.output_dim_list) +
                      '\n')
        file_02.write('Teacher param: {}'.format(teacher_net.output_dim_list) +
                      '\n')
        # file.write('student 参数量:{}\n'.format(student_param))
        # file.write('teacher 参数量:{} \n'.format(teacher_naram))

        file_02.write(f'test_____最终{file_name}Teacher网络结果\n')
        file_02.write(teacher_str)
        file_02.write('\n')


def kfold_training(file_name, file_2022, kfold_seed, student_param_list,
                   teacher_param_list, lr, epochs, quant_epoch, batch_size, torch_seed):
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    file_path = 'data_set/2021/{}.csv'.format(file_name)
    file_path_2022 = 'data_set/2022/{}.csv'.format(file_2022)
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
    '''2021数据集上的测试指标'''
    all_student_indicators = []
    all_teacher_indicators = []
    all_student_fine_indicators = []
    all_teacher_fine_indicators = []
    all_student_middle_indicators = []
    all_teacher_middle_indicators = []

    '''2022数据集上的测试指标'''
    all_student_indicators_2022 = []
    all_teacher_indicators_2022 = []
    all_student_fine_indicators_2022 = []
    all_teacher_fine_indicators_2022 = []
    all_student_middle_indicators_2022 = []
    all_teacher_middle_indicators_2022 = []

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
        # To_Deivce(x_train, y_train, x_test, y_test,
        #           x_2022_test, y_2022_test, device='cuda')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=10000000,
                                                  shuffle=True)
        test_dataset_2022 = DDataset(x_2022_test, y_2022_test)
        test_loader_2022 = torch.utils.data.DataLoader(test_dataset_2022,
                                                       batch_size=10000000,
                                                       shuffle=True)
        # 一次正常知识蒸馏，做一次test
        student_net, teacher_net = train_one(k, train_loader,
                                             student_param_list,
                                             teacher_param_list, dimension,
                                             batch_size, lr, epochs,
                                             torch_seed)
        # 知识蒸馏之后，微调没有中间损失 做一次test
        student_net_fine, teacher_net_fine = train_one_fine(
            k, train_loader, dimension, batch_size, lr, quant_epoch, torch_seed,
            student_param_list, teacher_param_list)

        # 知识蒸馏之后，带有中间层损失做一次test
        student_net_middle, teacher_net_middle = train_one_middle(
            k, train_loader, dimension, batch_size, lr, quant_epoch, torch_seed,
            student_param_list, teacher_param_list)
        '''对正常训练的网络 正常量化的网络 带有中间层损失量化的网络进行2021数据集上的测试'''
        student_net_indicator, teacher_net_indicator, student_net_fine_indicator, teacher_net_fine_indicator, student_net_middle_indicator, teacher_net_middle_indicator = mytest_one(
            file_name, k, test_loader, student_net, teacher_net,
            student_net_fine, teacher_net_fine, student_net_middle,
            teacher_net_middle, dimension, batch_size, lr, torch_seed, epochs, quant_epoch)
        path_common = 'training_result/common_model/'
        path_fine = 'training_result/fine_model/'
        path_middle = 'training_result/middle_model/'
        all_student_indicators.append(student_net_indicator)
        all_teacher_indicators.append(teacher_net_indicator)

        all_student_fine_indicators.append(student_net_fine_indicator)
        all_teacher_fine_indicators.append(teacher_net_fine_indicator)

        all_student_middle_indicators.append(student_net_middle_indicator)
        all_teacher_middle_indicators.append(teacher_net_middle_indicator)
        '''对正常训练的网络 正常量化的网络 带有中间层损失量化的网络进行2022数据集上的测试'''
        student_net_indicator_2022, teacher_net_indicator_2022, student_net_fine_indicator, teacher_net_fine_indicator, student_net_middle_indicator, teacher_net_middle_indicator = mytest_one(
            file_2022, k, test_loader_2022, student_net, teacher_net,
            student_net_fine, teacher_net_fine, student_net_middle,
            teacher_net_middle, dimension_2022, batch_size, lr, torch_seed, epochs, quant_epoch)

        all_student_indicators_2022.append(student_net_indicator_2022)
        all_teacher_indicators_2022.append(teacher_net_indicator_2022)
        all_student_fine_indicators_2022.append(student_net_fine_indicator)
        all_teacher_fine_indicators_2022.append(teacher_net_fine_indicator)

        all_student_middle_indicators_2022.append(student_net_middle_indicator)
        all_teacher_middle_indicators_2022.append(teacher_net_middle_indicator)

    get_average_reault(all_student_indicators,
                       all_teacher_indicators, path_common, file_name)

    get_average_reault(all_student_fine_indicators,
                       all_teacher_fine_indicators, path_fine, file_name)

    get_average_reault(all_student_middle_indicators,
                       all_teacher_middle_indicators, path_middle, file_name)

    get_average_reault(all_student_indicators_2022,
                       all_teacher_indicators_2022, path_common, file_2022)

    get_average_reault(all_student_fine_indicators_2022,
                       all_teacher_fine_indicators_2022, path_fine, file_2022)

    get_average_reault(all_student_middle_indicators_2022,
                       all_teacher_middle_indicators_2022, path_middle, file_2022)


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


def distillation_loss_QAT(student_output, teacher_output, label):
    student_middle_output_list, _, student_y = student_output
    teacher_middle_output_list, _, teacher_y = teacher_output
    # 获得teacher_idx
    idx_list = []
    for middle_output_tensor in student_middle_output_list:
        # 获取学生网络中间层输出的数据格式
        m = middle_output_tensor.shape[-1]
        # 返回教师网络中间层中有与学生网络相同数据格式的层所在的索引号
        idx = [
            t_m_tensor.shape[-1] for t_m_tensor in teacher_middle_output_list
        ].index(m)
        # 将相同数据格式的索引加入数组列表
        idx_list.append(idx)
    true_teacher_middle_output_list = [
        teacher_middle_output_list[idx] for idx in idx_list
    ]
    middle_loss = 0
    for s_m, t_m in zip(student_middle_output_list,
                        true_teacher_middle_output_list):
        middle_loss += F.mse_loss(s_m, t_m)

    entropy_loss = F.cross_entropy(student_y, label) + F.cross_entropy(
        teacher_y, label)
    last_loss = F.mse_loss(student_y, teacher_y)
    return entropy_loss + middle_loss + last_loss


def distillation_loss(student_output, teacher_output, label):
    _, _, student_y = student_output
    _, _, teacher_y = teacher_output
    last_loss = F.mse_loss(student_y, teacher_y)
    entropy_loss = F.cross_entropy(student_y, label) + F.cross_entropy(
        teacher_y, label)
    return entropy_loss + last_loss


def train_one(k, train_loader, student_param_list, teacher_param_list,
              dimension, batch_size, lr, epochs, torch_seed):
    student_net = Student(dimension, 2, student_param_list)
    teacher_net = Teacher(dimension, 2, teacher_param_list)

    student_net.to('cuda')
    teacher_net.to('cuda')
    # 采用默认参数
    optimizer = torch.optim.Adam(itertools.chain(student_net.parameters(),
                                                 teacher_net.parameters()),
                                 lr=lr)
    print('开始训练第{}折 torch seed'.format(k, torch_seed))
    print('batch size:{}, lr:{}'.format(batch_size, lr))
    print('Number of features:{}'.format(dimension))
    print('Student param: {}'.format(student_net.output_dim_list))
    print('Teacher param: {}'.format(teacher_net.output_dim_list))
    for epoch in tqdm(range(epochs)):
        # print('epoch: {}'.format(epoch + 1))
        # 表示模型处于训练模式
        student_net.train()
        teacher_net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.clone().detach().to('cuda')
            target = target.clone().detach().to('cuda')
            student_output = student_net(data)
            teacher_output = teacher_net(data)
            loss = distillation_loss(student_output, teacher_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(student_net.state_dict(),
               'model_fine_/Storage/{}/pre_student_'.format(file_name) + str(k) + '.pth')
    torch.save(teacher_net.state_dict(),
               'model_fine_/Storage/{}/pre_teacher_'.format(file_name) + str(k) + '.pth')
    student_net.to('cpu')
    teacher_net.to('cpu')
    return student_net, teacher_net


def train_one_fine(k, train_loader, dimension, batch_size, lr, quant_epoch,
                   torch_seed, student_param_list, teacher_param_list):
    # To_Deivce(train_loader, device='cpu')
    student_net_fine = Student(dimension, 2, student_param_list)
    teacher_net = Teacher(dimension, 2, teacher_param_list)
    student_net_fine.load_state_dict(
        torch.load('model_fine_/Storage/{}/pre_student_'.format(file_name) + str(k) + '.pth'))
    teacher_net.load_state_dict(
        torch.load('model_fine_/Storage/{}/pre_teacher_'.format(file_name) + str(k) + '.pth'))
    student_net_fine.to('cpu')
    teacher_net.to('cpu')
    '''如果模型中存在线性层和relu层 则将他们进行融合'''
    for model in student_net_fine.model:
        if isinstance(model[1], nn.Linear) and isinstance(model[2], nn.ReLU):
            model = torch.quantization.fuse_modules(model, ['1', '2'],
                                                    inplace=True)
        elif isinstance(model[0], nn.Linear) and isinstance(model[1], nn.ReLU):
            model = torch.quantization.fuse_modules(model, ['0', '1'],
                                                    inplace=True)

    student_net_fine.qconfig = torch.ao.quantization.get_default_qat_qconfig(
        'x86')
    torch.ao.quantization.prepare_qat(student_net_fine, inplace=True)
    optimizer = torch.optim.Adam(itertools.chain(student_net_fine.parameters(),
                                                 teacher_net.parameters()),
                                 lr=lr)
    print('开始训练不带中间层损失的第{}折 torch seed {}'.format(k, torch_seed))
    print('batch size:{}, lr:{}'.format(batch_size, lr))
    print('Number of features:{}'.format(dimension))
    print('Student param: {}'.format(student_net_fine.output_dim_list))
    print('Teacher param: {}'.format(teacher_net.output_dim_list))
    for epoch in tqdm(range(quant_epoch)):
        # print('epoch: {}'.format(epoch + 1))
        # 表示模型处于训练模式
        student_net_fine.train()
        teacher_net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            student_output = student_net_fine(data)
            teacher_output = teacher_net(data)
            '''此为量化不带中间层损失'''
            loss = distillation_loss(student_output, teacher_output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch > 9:
            student_net_fine.apply(torch.ao.quantization.disable_observer)
    student_net_fine = torch.ao.quantization.convert(student_net_fine,
                                                     inplace=True)
    return student_net_fine, teacher_net


def train_one_middle(k, train_loader, dimension, batch_size, lr, quant_epoch,
                     torch_seed, student_param_list, teacher_param_list):
    # To_Deivce(train_loader, device='cpu')
    student_net_middle = Student(dimension, 2, student_param_list)
    teacher_net = Teacher(dimension, 2, teacher_param_list)
    student_net_middle.load_state_dict(
        torch.load('model_fine_/Storage/{}/pre_student_'.format(file_name) + str(k) + '.pth'))
    teacher_net.load_state_dict(
        torch.load('model_fine_/Storage/{}/pre_teacher_'.format(file_name) + str(k) + '.pth'))
    student_net_middle.to('cpu')
    teacher_net.to('cpu')
    '''如果模型中存在线性层和relu层 则将他们进行融合'''
    for model in student_net_middle.model:
        if isinstance(model[1], nn.Linear) and isinstance(model[2], nn.ReLU):
            model = torch.quantization.fuse_modules(model, ['1', '2'],
                                                    inplace=True)
        elif isinstance(model[0], nn.Linear) and isinstance(model[1], nn.ReLU):
            model = torch.quantization.fuse_modules(model, ['0', '1'],
                                                    inplace=True)

    student_net_middle.qconfig = torch.ao.quantization.get_default_qat_qconfig(
        'x86')
    torch.ao.quantization.prepare_qat(student_net_middle, inplace=True)
    optimizer = torch.optim.Adam(itertools.chain(
        student_net_middle.parameters(), teacher_net.parameters()),
        lr=lr)
    print('开始训练携带中间层损失的第{}折 torch seed {}'.format(k, torch_seed))
    print('batch size:{}, lr:{}'.format(batch_size, lr))
    print('Number of features:{}'.format(dimension))
    print('Student param: {}'.format(student_net_middle.output_dim_list))
    print('Teacher param: {}'.format(teacher_net.output_dim_list))
    for epoch in tqdm(range(quant_epoch)):
        # 表示模型处于训练模式
        student_net_middle.train()
        teacher_net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            student_output = student_net_middle(data)
            teacher_output = teacher_net(data)
            '''带有中间层损失的量化'''
            loss = distillation_loss_QAT(student_output, teacher_output,
                                         target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch > 9:
            student_net_middle.apply(torch.ao.quantization.disable_observer)
    student_net_middle = torch.ao.quantization.convert(student_net_middle,
                                                       inplace=True)
    torch.save(student_net_middle.state_dict(),
               'model_q/Storage/{}/pre_student_middle'.format(file_name) + str(k) + '.pth')
    return student_net_middle, teacher_net


def mytest_one(file_name, k, test_loader, student_net, teacher_net,
               student_net_fine, teacher_net_fine, student_net_middle,
               teacher_net_middle, dimension, batch_size, lr, torch_seed, epoch, quant_epoch):
    # To_Deivce(test_loader, device='cpu')
    print('开始test 第{}折'.format(k))
    #  模型进入验证
    student_net.eval()
    teacher_net.eval()
    # 带有中间层损失的量化模型和不带中间层损失的量化模型进入验证
    student_net_fine.eval()
    teacher_net_fine.eval()
    student_net_middle.eval()
    teacher_net_middle.eval()
    # student_param = summary(student_net, input_size=(dimension, ))
    # teacher_naram = summary(teacher_net, input_size=(dimension, ))
    # print('number of student param: {}'.format(student_param))
    # print('number of teacher param: {}'.format(teacher_naram))

    name_list = [
        'acc', 'precision', 'recall', 'fmeature', 'specific', 'tpr', 'fpr',
        'mcc', 'auc'
    ]
    # data_iter = iter(test_loader)
    # data, target = next(data_iter)
    # print(data)
    # print(target)
    for batch_idx, (data, target) in enumerate(test_loader):
        student_output = student_net(data)[-1]
        teacher_output = teacher_net(data)[-1]
        student_fine_output = student_net_fine(data)[-1]
        teacher_fine_output = teacher_net_fine(data)[-1]
        student_middle_output = student_net_middle(data)[-1]
        teacher_middle_output = teacher_net_middle(data)[-1]
        student_output = torch.argmax(student_output, dim=-1, keepdim=False)
        teacher_output = torch.argmax(teacher_output, dim=-1, keepdim=False)
        student_fine_output = torch.argmax(student_fine_output,
                                           dim=-1,
                                           keepdim=False)
        teacher_fine_output = torch.argmax(teacher_fine_output,
                                           dim=-1,
                                           keepdim=False)
        student_middle_output = torch.argmax(student_middle_output,
                                             dim=-1,
                                             keepdim=False)
        teacher_middle_output = torch.argmax(teacher_middle_output,
                                             dim=-1,
                                             keepdim=False)

        # get_indicators接受的参数分别是模型的输出结果，真实标签和参数序列
        # 此处获得的student_indicators是一个指标的综合结果集
        student_indicators = get_indicators(student_output.cpu().numpy(),
                                            target.cpu().numpy().reshape(-1),
                                            name_list)
        teacher_indicators = get_indicators(teacher_output.cpu().numpy(),
                                            target.cpu().numpy().reshape(-1),
                                            name_list)
        student_fine_indicators = get_indicators(
            student_fine_output.cpu().numpy(),
            target.cpu().numpy().reshape(-1), name_list)
        teacher_fine_indicators = get_indicators(
            teacher_fine_output.cpu().numpy(),
            target.cpu().numpy().reshape(-1), name_list)
        student_middle_indicators = get_indicators(
            student_middle_output.cpu().numpy(),
            target.cpu().numpy().reshape(-1), name_list)
        teacher_middle_indicators = get_indicators(
            teacher_middle_output.cpu().numpy(),
            target.cpu().numpy().reshape(-1), name_list)
    print('test_____最终student网络结果')
    # format_print利用zip分别打印出参数和对应的结果
    format_print(name_list, student_indicators)
    # format_str将所有的结果存到一个字符串里
    student_str = format_str(name_list, student_indicators)
    print('test____最终teacher网络结果')
    format_print(name_list, teacher_indicators)
    teacher_str = format_str(name_list, teacher_indicators)

    print('test_____最终fine_student网络结果')
    # format_print利用zip分别打印出参数和对应的结果
    format_print(name_list, student_fine_indicators)
    # format_str将所有的结果存到一个字符串里
    student_fine_str = format_str(name_list, student_fine_indicators)
    print('test____最终fine_teacher网络结果')
    format_print(name_list, teacher_fine_indicators)
    teacher_fine_str = format_str(name_list, teacher_fine_indicators)

    print('test_____最终middle_student网络结果')
    # format_print利用zip分别打印出参数和对应的结果
    format_print(name_list, student_middle_indicators)
    # format_str将所有的结果存到一个字符串里
    student_middle_str = format_str(name_list, student_middle_indicators)
    print('test____最终middle_teacher网络结果')
    format_print(name_list, teacher_middle_indicators)
    teacher_middle_str = format_str(name_list, teacher_middle_indicators)

    # 保存数据
    path_common = 'training_result/common_model'
    Save_File(path_common, file_name, k, torch_seed,
              batch_size, dimension, student_net, teacher_net, lr, student_str, teacher_str, epoch, quant_epoch)
    path_fine = 'training_result/fine_model'
    Save_File(path_fine, file_name, k, torch_seed,
              batch_size, dimension, student_net, teacher_net, lr, student_fine_str, teacher_fine_str, epoch,
              quant_epoch)
    path_middle = 'training_result/middle_model'
    Save_File(path_middle, file_name, k, torch_seed,
              batch_size, dimension, student_net, teacher_net, lr, student_middle_str, teacher_middle_str, epoch,
              quant_epoch)

    # return student_indicators, teacher_indicators, student_param, teacher_naram
    return student_indicators, teacher_indicators, student_fine_indicators, teacher_fine_indicators, student_middle_indicators, teacher_middle_indicators


def get_average_reault(all_student_indicators, all_teacher_indicators,
                       path, file_name):
    student_mean = np.array(all_student_indicators).mean(axis=0)
    teacher_mean = np.array(all_teacher_indicators).mean(axis=0)
    name_list = [
        'acc', 'precision', 'recall', 'fmeature', 'specific', 'tpr', 'fpr',
        'mcc', 'auc'
    ]

    student_mean_str = format_str(name_list, student_mean)
    teacher_mean_str = format_str(name_list, teacher_mean)
    print('student 平均结果************************************')
    format_print(name_list, student_mean)
    print('teacher 平均结果************************************')
    format_print(name_list, teacher_mean)
    with open('{}/{}_student_net.txt'.format(path, file_name),
              'a',
              encoding='utf8') as file:
        # file.write('student 平均参数量:{}\n'.format(student_param_mean))
        # file.write('teacher 参数量:{}\n'.format(teacher_param_mean))
        file.write('student 平均结果\n')
        file.write(student_mean_str)

    with open('{}/{}_teacher_net.txt'.format(path, file_name),
              'a',
              encoding='utf8') as file:
        file.write('teacher 平均结果\n')
        file.write(teacher_mean_str)


if __name__ == '__main__':

    for file_name in ['pre_per_1757', 'pre_API_1683', 'pre_per_api']:
        kfold_training(
            file_name=file_name,
            file_2022=file_name + '_2022',
            kfold_seed=2,
      
            student_param_list=[512, 128, 32],
            teacher_param_list=[1024, 512, 256, 128, 64, 32],
            lr=0.001, 
            epochs=100, 
            quant_epoch=30,
            batch_size=256,  
            torch_seed=0 
        )
