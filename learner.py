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
            elif name == 'bi-lstm':
                # forward weights inputs
                # [ 4 x hidden, in size ]
                w_ih = nn.Parameter(torch.ones(4 * param[1], param[0]))
                torch.nn.init.kaiming_normal_(w_ih)
                # [ 4 x hidden, hidden size ]
                w_hh = nn.Parameter(torch.ones(4 * param[1], param[1]))
                torch.nn.init.kaiming_normal_(w_hh)
                # backwards inputs, flip dims of forwards direction
                r_w_ih = nn.Parameter(torch.ones(param[0], 4 * param[1]))
                torch.nn.init.kaiming_normal_(r_w_ih)
                r_w_hh = nn.Parameter(torch.ones(param[1], 4 * param[1]])
                torch.nn.init.kaiming_normal_(r_w_hh)

                self.vars.extend([w_ih, w_hh, r_w_ih, r_w_hh])

                # forward biases
                # [ 4 x hidden size ]
                b_ih = nn.Parameter(torch.zeros(4 * param[1]))
                # [ 4 x hidden size ]
                b_hh = nn.Parameter(torch.zeros(4 * param[1]))
                # backwards biases
                r_b_ih = nn.Parameter(torch.zeros(4 * param[1]))
                r_b_hh = nn.Parameter(torch.zeros(4 * param[1]))

                self.vars.extend([b_ih, b_hh, r_b_ih, r_b_hh])
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
                tmp = 'conv1d:(ch_out:%d, ch_in:%d, k:%d, stride:%d, padding:%s)'\
                        %(param[0], param[1], param[2], param[3], param[4])
                info += tmp + '\n'
            elif name == 'dense':
                tmp = 'dense:(ch_out:%d, ch_in:%d)'\
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

        # Here x is of size [1, set/query size, emb size]

        # Actually utilize the layers now
        for name, param in self.config:
            if name == 'conv1d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv1d(x, w, b, stride=param[3], padding=param[4])
                assert not torch.isnan(x).any(), 'nan in layer'
                idx += 2
            elif name == 'dense':
                w, b = vars[idx], vars[idx + 1]
                #print('linear w tensor', w.size(), flush=True)
                #print('linear b tensor', b.size(), flush=True)
                #print('input into linear', x.size(), flush=True)
                x = F.linear(x, w, b)
                #print('output from linear', x.size(), flush=True)
                assert not torch.isnan(x).any(), 'nan in layer'
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bi-lstm':
                w_ih, w_hh, r_w_ih, r_w_hh = vars[idx], vars[idx + 1], vars[idx + 2], vars[idx + 3]
                weights = [w_ih, w_hh, r_w_ih, r_w_hh]
                b_ih, b_hh, r_b_ih, r_b_hh = vars[idx + 4], vars[idx + 5], vars[idx + 6], vars[idx + 7]
                biases = [b_ih, b_hh, r_b_ih, r_b_hh]
                # initial hidden states

                # hidden layers
                batch_size = x.size(0)
                h_zeros = nn.Parameter(torch.zeros(1 * 2, batch_size, param[1]))
                c_zeros = nn.Parameter(torch.zeros(1 * 2, batch_size, param[1]))
                hx = (h_zeros, c_zeros)
               
                flat_w = torch.cat([ torch.flatten(w) for w in weights]
                flat_b = torch.cat(biases)
                # input, hx, flat weights, bias, num layers, deropout, training, bidirectional, batch first
                x, (h_n, c_n) = f_lstm(x, hx, flat_w, flat_b, 1, param[2], self.training, True, False)
                idx += 8
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'elu':
                x = F.elu(x, inplace=param[0])
                assert not torch.isnan(x).any(), 'nan in layer'
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
                assert not torch.isnan(x).any(), 'nan in layer'
            elif name == 'globalmax_pool1d':
                #print('input into global max pool', x.size(), flush=True)
                x = F.adaptive_max_pool1d(x, 1)
                x = torch.flatten(x, start_dim=1)
                assert not torch.isnan(x).any(), 'nan in layer'
                #print('output from global max pool', x.size(), flush=True)
            elif name == 'dropout':
                x = F.dropout(x, p=param[0], inplace=param[1])
                assert not torch.isnan(x).any(), 'nan in layer'
            elif name == 'softmax':
                x = F.softmax(x, dim=0)
                assert not torch.isnan(x).any(), 'nan in layer'
                #print('softmax output', x.size(), flush=True)
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
