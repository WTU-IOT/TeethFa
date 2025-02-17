import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torchmeta.modules import (MetaModule, MetaConv1d, MetaBatchNorm1d,MetaLinear)


def calculate_mask_index(kernel_length_now,largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now

def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_lenght)
    mask = np.ones((number_of_input_channel,number_of_output_channel,largest_kernel_lenght))
    mask[:,:,0:ind_left]=0
    mask[:,:,ind_right:]=0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l,ind_r= calculate_mask_index(i[2],largest_kernel_lenght)
        big_weight = np.zeros((i[1],i[0],largest_kernel_lenght))
        big_weight[:,:,ind_l:ind_r]= conv.weight.detach().numpy()
        
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        
        mask = creat_mask(i[1],i[0],i[2], largest_kernel_lenght)
        mask_list.append(mask)
        
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)

    
class build_layer_with_layer_parameter(nn.Module):
    def __init__(self,layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()

        os_mask, init_weight, init_bias= creak_layer_mask(layer_parameters)
        
        
        in_channels = os_mask.shape[1] 
        out_channels = os_mask.shape[0] 
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask),requires_grad=False)
        
        self.padding = nn.ConstantPad1d((int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)
         
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight),requires_grad=True)
        self.conv1d.bias =  nn.Parameter(torch.from_numpy(init_bias),requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)
    
    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        #self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        result = F.relu(result_3)
        return result    



class OS_CNN(nn.Module):
    def __init__(self,layer_parameter_list,n_class,few_shot = True):
        super(OS_CNN, self).__init__()
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []


        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)

        self.net = nn.Sequential(*self.layer_list)

        self.averagepool = nn.AdaptiveAvgPool1d(1)

        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr+ final_layer_parameters[1]

        self.hidden = nn.Linear(out_put_channel_numebr, n_class)

    def forward(self, X):
        X = self.net(X)
        X = self.averagepool(X)
        X = X.squeeze_(-1)
        if not self.few_shot:
            X = self.hidden(X)
        return X




class build_layer_with_layer_parameter_meta(MetaModule):
    def __init__(self, layer_parameters):
        super(build_layer_with_layer_parameter_meta, self).__init__()

        os_mask, init_weight, init_bias = creak_layer_mask(layer_parameters)

        in_channels = os_mask.shape[1]
        out_channels = os_mask.shape[0]
        max_kernel_size = os_mask.shape[-1]

        # self.weight_mask = nn.Parameter(torch.from_numpy(os_mask), requires_grad=True)

        self.padding = nn.ConstantPad1d((int((max_kernel_size - 1) / 2), int(max_kernel_size / 2)), 0)

        self.conv1d = MetaConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)

        self.bn = MetaBatchNorm1d(num_features=out_channels)

    def forward(self, X,params=None):
        # self.conv1d.weight.data = self.conv1d.weight * self.weight_mask

        if params is None:
            result_1 = self.padding(X)
            result_2 = self.conv1d(result_1)
            result_3 = self.bn(result_2)
            result = F.relu(result_3)
        else:
            result_1 = self.padding(X)
            result_2 = self.conv1d(result_1,params=self.get_subdict(params, 'conv1d'))
            result_3 = self.bn(result_2,params=self.get_subdict(params, 'bn'))
            result = F.relu(result_3)
        return result



#===================设置OS_CNN中的部分层为元参数=============================
class MetaOS_CNN(MetaModule):
    def __init__(self, layer_parameter_list, n_class):
        super(MetaOS_CNN, self).__init__()
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        self.OS1=  build_layer_with_layer_parameter_meta(layer_parameter_list[0])
        self.OS2 = build_layer_with_layer_parameter_meta(layer_parameter_list[1])
        self.OS3 = build_layer_with_layer_parameter_meta(layer_parameter_list[2])
        self.OS4 = build_layer_with_layer_parameter_meta(layer_parameter_list[3])
        self.dr1 = nn.Dropout(0.1)
        self.averagepool = nn.AdaptiveAvgPool1d(1)

        out_put_channel_number = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_number += final_layer_parameters[1]

        self.hidden = MetaLinear(out_put_channel_number, n_class)
        # self.hidden=nn.Linear(out_put_channel_number, n_class)


    def forward(self, inputs, params=None):
        if params is None:
            x = self.OS1(inputs)
            x=self.OS2(x)
            x = self.OS3(x)
            x = self.OS4(x)
            x = self.dr1(x)
        else:
            x = self.OS1(inputs, params=self.get_subdict(params, 'OS1'))
            x = self.OS2(x, params=self.get_subdict(params, 'OS2'))
            x = self.OS3(x, params=self.get_subdict(params, 'OS3'))
            x = self.OS4(x, params=self.get_subdict(params, 'OS4'))
            x = self.dr1(x)
            #x = self.OS4(x)
        x = self.averagepool(x)
        x = x.squeeze_(-1)
        if params is None:
            x = self.hidden(x)
        else:
            x = self.hidden(x, params=self.get_subdict(params, 'hidden'))
            # x = self.hidden(x)
        return x
    def zero_linear_layers(self):
        # 遍历模型的子模块
        for name, module in self.named_modules():
            if isinstance(module, MetaLinear):
                # 如果模块是全连接层，将其权重和偏差置零
                module.weight.data.fill_(0)
                module.bias.data.fill_(0)





