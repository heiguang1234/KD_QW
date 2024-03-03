# 开发者： Hei Guang
# 开发时间：2023/4/13 15:24
import torch.nn as nn
import torch.nn.functional as F
import torch
'''
这是一个使用PyTorch实现的多输出MLP(多层感知机)模型的类定义。该类继承自nn.Module类,重写了__init__方法和forward方法,用于定义模型的结构和前向传播过程。

__init__方法中,定义了模型的输入维度input_dim、输出维度列表output_dim_list和类别数n_classes。其中,output_dim_list是一个列表,包含了每一层的输出维度。
whole_dim_list是一个整个模型的维度列表,包含了输入层、隐藏层和输出层的维度。models_list是一个空列表,
用于存储每一层的模型。在循环中,对于每一层,如果不是输出层,就使用nn.Linear和nn.ReLU定义一个线性层和一个ReLU激活函数层,并将它们封装在nn.Sequential容器中;
如果是输出层,就只使用nn.Linear定义一个线性层。最后,使用nn.ModuleList将所有的模型封装在一个模型列表中。
forward方法中,定义了模型的前向传播过程。对于Ï每一层,将输入x传入模型中,得到输出x,并将其作为下一层的输入。最后,将输出x返回。
该模型的结构为：输入层 -> 隐藏层1 -> ReLU激活函数层1 -> … -> 隐藏层n -> ReLU激活函数层n -> 输出层1 -> … -> 输出层m,其中,隐藏层和输出层的数量和维度由output_dim_list和n_classes参数决定。
'''


class Student(nn.Module):
    # 多层感知器网络隐藏层(256,64)
    def __init__(self, input_dim, n_classes=2, output_dim_list=[256, 64]):
        super(Student, self).__init__()
        self.output_dim_list = output_dim_list
        # dimension是特征的维度=input_dim
        whole_dim_list = [input_dim] + output_dim_list + [n_classes]
        models_list = []

        for i in range(len(whole_dim_list) - 1):
            if i == len(whole_dim_list) - 1 - 1:
                # Sequential容器将语句顺序封装
                # 最后一层的前一层增加一层线性层,不加relu激活函数
                models_list.append(
                    nn.Sequential(
                        nn.Linear(whole_dim_list[i], whole_dim_list[i + 1]),
                        torch.ao.quantization.DeQuantStub()))
            elif i == 0:
                models_list.append(
                    nn.Sequential(
                        torch.ao.quantization.QuantStub(),
                        nn.Linear(whole_dim_list[i], whole_dim_list[i + 1]),
                        nn.ReLU()))
            else:
                # 其他加入一层线性层的同时加入Relu激活函数
                models_list.append(
                    nn.Sequential(
                        nn.Linear(whole_dim_list[i], whole_dim_list[i + 1]),
                        nn.ReLU()))
        self.model = nn.ModuleList(models_list)

    def forward(self, x):
        middle_output_list = []
        for model in self.model:
            x = model(x)
            middle_output_list.append(x)
        y = F.softmax(x, dim=-1)
        return middle_output_list, y


class Teacher(nn.Module):
    # 多层感知器网络隐藏层(256,64)
    def __init__(self, input_dim, n_classes=2, output_dim_list=[256, 64]):
        super(Teacher, self).__init__()
        self.output_dim_list = output_dim_list
        # dimension是特征的维度=input_dim
        whole_dim_list = [input_dim] + output_dim_list + [n_classes]
        models_list = []
        for i in range(len(whole_dim_list) - 1):
            if i == len(whole_dim_list) - 1 - 1:
                # Sequential容器将语句顺序封装
                # 最后一层的前一层增加一层线性层,不加relu激活函数
                models_list.append(
                    nn.Sequential(
                        nn.Linear(whole_dim_list[i], whole_dim_list[i + 1]), ))
            else:
                # 其他加入一层线性层的同时加入Relu激活函数
                models_list.append(
                    nn.Sequential(
                        nn.Linear(whole_dim_list[i], whole_dim_list[i + 1]),
                        nn.ReLU()))
        self.model = nn.ModuleList(models_list)

    def forward(self, x):
        middle_output_list = []
        for model in self.model:
            x = model(x)
            middle_output_list.append(x)
        y = F.softmax(x, dim=-1)
        return middle_output_list, y
