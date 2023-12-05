import  torch
from    torch import nn
from    torch.nn import functional as F
from torch._VF import lstm as f_lstm
from torch.func import functional_call
import  numpy as np

class Learner(nn.Module):
    """
    Modified for 1d data
    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        # Get the vars/tensors needed for each layer
        for i, (name, param) in enumerate(self.config):

            # CNN layers first
            if name == 'conv1d':
                # param = [ ch_out, ch_in, kernelsz ]
                w = nn.Parameter(torch.ones(*param[:3]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # param = [ ch_out ]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'dense':
                # param = [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # param = [ ch_out ]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'bn':
                # [ ch_out ]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name in ['elu', 'relu', 'globalmax_pool1d', 'softmax', 'dropout']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv1d':
                tmp = 'conv1d:(ch_in%d, ch_out:%d, k:%d, stride:%d, padding:%s)'\
                        %(param[0], param[1], param[2], param[3], param[4])
                info += tmp + '\n'
            elif name == 'dense':
                tmp = 'dense:(ch_in%d, ch_out%d)'\
                        %(param[0], param[1])
                info += tmp + '\n'
            elif name in ['relu', 'elu', 'globalmax_pool1d', 'softmax', 'dropout', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        # Actually utilize the layers now
        for name, param in self.config:
            if name == 'conv1d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv1d(x, w, b, stride=param[3], padding=param[4])
                idx += 2
            elif name == 'dense':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bi-lstm':
                w, b = vars[idx], vars[idx + 1]
                # initial hidden states
                hx = 
                x = f_lstm(x, hx, w, b, 1, #dropout, self.training, True, batchfirst)
                idx += 2
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'elu':
                x = F.elu(inplace=param[0])
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'globalmax_pool1d':
                x = F.adaptive_max_pool1d(x, 1)
            elif name == 'dropout':
                x = F.dropout(x, p=param[0], inplace=param[1])
            elif name == 'softmax':
                x = F.softmax(x)
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
