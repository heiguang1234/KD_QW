import torch
from torch import nn
from torch.nn import functional as F


class BaseBlock(nn.Module):
    def __init__(self, input_dim, output_dim, n_classes, theta=1):
        super(BaseBlock, self).__init__()
        '''
        realize block here
        '''
        self.n_classes = n_classes

        self.theta = nn.Parameter(data=torch.tensor(theta, dtype=torch.float32, requires_grad=True).view(1,1),
                                  requires_grad=True)

    def forward(self, x):
        raise NotImplementedError

    def __gt__(self, other):
        if self.theta > other.theta:
            return True
        else:
            return False
'''
BaseBlock 的子类可以有更多种类，以下MLPBlock只是一个典型的子类
'''


class MLPBlock(BaseBlock):
    def __init__(self, input_dim, output_dim, n_classes, theta=1):
        super(MLPBlock, self).__init__(input_dim, output_dim, n_classes, theta)
        self.fc1 = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.fc2 = nn.Linear(in_features=output_dim, out_features=n_classes)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.softmax(self.fc2(y), dim=-1)
        return y


class SoftMLPBlock(BaseBlock):
    def __init__(self, input_dim, output_dim, n_classes, theta=1):
        super(SoftMLPBlock, self).__init__(input_dim, output_dim, n_classes, theta)
        self.fc1 = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.fc2 = nn.Linear(in_features=output_dim, out_features=n_classes)

    def forward(self, x):
        if self.training:
            y = F.relu(self.fc1(x))
            y = self.fc2(y)
            return y
        else:
            y = F.relu(self.fc1(x))
            y = self.fc2(y)
            return F.softmax(y, dim=-1)

class HardMLPBlock(BaseBlock):
    def __init__(self, input_dim, output_dim, n_classes=2, theta=1):
        super(HardMLPBlock, self).__init__(input_dim, output_dim, n_classes, theta)
        self.fc1 = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.fc2 = nn.Linear(in_features=output_dim, out_features=n_classes)

    def forward(self, x):

        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        return F.softmax(y, dim=-1)



class Net(nn.Module):
    def __init__(self, block_list):
        '''
        :param block_list: [BaseBlock, BaseBlock,...]
        '''
        super(Net, self).__init__()
        self.block_list = nn.ModuleList(block_list)
        self.n_blocks = len(self.block_list)
        self.n_classes = block_list[0].n_classes

    def to_device(self, device):
        for block in self.block_list:
            block.to(device)

    def get_final_net(self):
        return max(self.block_list)

    def get_thetas(self):
        return [block.theta for block in self.block_list]

    def forward(self, x):
        if self.training:
            '''
            return the reduce sum of the outputs of blocks
            '''
            thetas = torch.cat([block.theta for block in self.block_list], dim=0) #(n_blocks,1)
            weights = F.softmax(thetas, dim=0) #(n_blocks,1)
            block_outputs = torch.cat([self.block_list[i](x).view(-1,1,self.n_classes)
                       for i in range(self.n_blocks)], dim=1) #(bsz,n_blocks,n_classes)
            return torch.sum(weights * block_outputs, dim=1)

        else:
            return self.get_final_net()(x)


class SoftMLPNet(Net):

    def forward(self, x):
        if self.training:
            '''
            return the reduce sum of the outputs of blocks
            '''
            thetas = torch.cat([block.theta for block in self.block_list], dim=0) #(n_blocks,1)
            weights = F.softmax(thetas, dim=0) #(n_blocks,1)
            block_outputs = torch.cat([self.block_list[i](x).view(-1,1,self.n_classes)
                       for i in range(self.n_blocks)], dim=1) #(bsz,n_blocks,n_classes)

            soft_output = torch.sum(weights * block_outputs, dim=1) / self.n_blocks #(bsz, n_classes) / self.n_blocks
            y = F.softmax(soft_output, dim=-1)
            return soft_output, y
        else:
            return self.get_final_net()(x)


if __name__ == '__main__':
    n_classes = 2
    input_dim = 2293
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = [MLPBlock(input_dim,128,n_classes,theta=1), MLPBlock(input_dim,32,n_classes), MLPBlock(input_dim,16,n_classes)]
    model = Net(block_list=a)
    model.to_device(device)
    for i in model.parameters():
        print(i.shape)
    x = torch.rand((3,input_dim))
    y = model(x)
    print(y)
    final_model =  model.get_final_net()
    y = final_model(x)
    print(y, final_model.theta)