import os
import torch
from torch import nn
from torch.autograd import Variable
from agent_critic.actor_network import judgeNet


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: int
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.width = 27
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype
        self.act = nn.Mish()
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.conv_gates = nn.Conv1d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv1d(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)
        self.condition = nn.Conv1d(in_channels=8,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_dim, self.width)).type(self.dtype)

    def forward(self, input_tensor, h_cur, action):
        """
        :param self:
        :param input_tensor: (b, c, 27)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :param action: (b, 8, 27)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state

        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined) + self.condition(action)

        cnm = cc_cnm

        h_next = self.norm(self.act((1 - update_gate) * h_cur + update_gate * cnm))
        return h_next


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype=torch.cuda.FloatTensor, batch_first=False, bias=True):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: int
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(Critic, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = [kernel_size, ] * num_layers
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        self.width = 27
        self.input_dim = input_dim + 7
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        # self.last_conv = nn.Conv1d(in_channels=hidden_dim[-1], out_channels=1, kernel_size=(3,), stride=(1,),
        #                            padding=1, bias=False)
        self.concentrate = nn.Linear(27 * self.hidden_dim[-1], 1, bias=True)
        # self.last_conv = nn.Conv1d(self.hidden_dim[-1], 1, kernel_size=1, padding=0, bias=True)
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]
            cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)
        self.xt_ps = judgeNet(5, num_block=10, hidden_dim=64, out_dim=1)

    def get_punish(self, input_tensor, target=None):
        """
        :param input_tensor: (b, 8, 27)
        """
        # 用于估计相听(因为递归计算太费时间,不方便训练)
        input_tensor = input_tensor.cuda()
        logit = torch.sigmoid(self.xt_ps(input_tensor))

        if target is not None:
            target = target.cuda()
            loss = ((logit - target) ** 2).mean()
            return loss
        else:
            return logit

    def forward(self, input_tensor, mask=None, target=None, hidden_state=None):
        """
        :param input_tensor: (b, t, c, 27)
            extracted features from alexnet
        :param mask: (b, 24)
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        action_feature = input_tensor[:, :, -8:, :]
        if target is not None:
            assert mask is not None

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
        cur_layer_input = input_tensor
        if mask is not None:
            # mask = mask.cuda()
            max_length = int(mask.sum(1).max())
            mask = mask[:, :max_length].cuda()
        else:
            max_length = input_tensor.size(1)
        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(max_length):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :],  # (b,t,c,w)
                                              h_cur=h, action=action_feature[:, t, :, :])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
        bszie, seq_len = layer_output.shape[0], layer_output.shape[1]
        layer_output = layer_output.view(bszie * seq_len, -1)
        # reward = self.last_conv(layer_output)
        reward = self.concentrate(torch.squeeze(layer_output)).view(bszie, seq_len)
        if target is not None:
            loss = ((reward - target[:, None]).pow(2) * mask).sum() / mask.sum()
            return loss
        elif mask is not None:
            reward = reward * mask
            idx = mask.sum(1).int()
            reward = torch.cat((torch.zeros((reward.shape[0], 1)).to(reward), reward), dim=1)
            reward = 0.99 * reward[range(reward.shape[0]), idx.tolist()] - reward[range(reward.shape[0]), (idx-1).tolist()]
            return reward
        else:
            if reward.shape[1] != 1:
                reward = 0.99 * torch.squeeze(reward[:, -1] - reward[:, -2])
            else:
                reward = torch.squeeze(reward[:, -1])
            return reward

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    # set CUDA device
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # detect if CUDA is available or not
    height = width = 6
    channels = 256
    hidden_dim = [32, 64, 1]
    kernel_size = 3  # kernel size for two stacked hidden layer
    num_layers = 3  # number of stacked hidden layer
    model = Critic(input_dim=channels,
                   hidden_dim=hidden_dim,
                   kernel_size=kernel_size,
                   num_layers=num_layers,
                   batch_first=True,
                   bias=True).cuda()

    bsize = 20
    time_steps = 10
    max_time_steps = 24
    mask = torch.zeros(bsize, max_time_steps)
    mask[:, :10] = torch.ones(20, 10).cuda()
    mask[0, :13] = torch.ones(13).cuda()
    feature = torch.rand(1, time_steps, channels, 27).cuda()
    input_tensor = torch.rand(bsize, max_time_steps, channels, 27).cuda()  # (b,t,c,h,w)
    input_tensor[1:, 10:] = torch.zeros(bsize - 1, 14, channels, 27).cuda()
    input_tensor[0, 13:] = torch.zeros(11, channels, 27).cuda()
    target = torch.zeros(20).cuda()
    loss = model(input_tensor, mask, target)
    print(loss)
    reward = model(feature)
    print(reward)
