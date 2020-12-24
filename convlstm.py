'''
Date: 2020/08/30
@author: KimUyen
# The code was revised from repo: https://github.com/ndrplz/ConvLSTM_pytorch
'''
import torch.nn as nn
import torch
import math

class HadamardProduct(nn.Module):
    def __init__(self, shape):
        super(HadamardProduct, self).__init__()
        self.weights = nn.Parameter(torch.rand(shape)).cuda()
        
    def forward(self, x):
        return x*self.weights

class ConvLSTMCell(nn.Module):

    def __init__(self, img_size, input_dim, hidden_dim, kernel_size, 
                 cnn_dropout, rnn_dropout, bias=True, peephole=False,
                 layer_norm=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel for both cnn and rnn.
        cnn_dropout, rnn_dropout: float
            cnn_dropout: dropout rate for convolutional input.
            rnn_dropout: dropout rate for convolutional state.
        bias: bool
            Whether or not to add the bias.
        peephole: bool
            add connection between cell state to gates
        layer_norm: bool
            layer normalization 
        """

        super(ConvLSTMCell, self).__init__()
        self.input_shape = img_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (int(self.kernel_size[0]/2), int(self.kernel_size[1]/2))
        self.stride = (1, 1)
        self.bias = bias
        self.peephole = peephole
        self.layer_norm = layer_norm
        
        self.out_height = int((self.input_shape[0] - self.kernel_size[0] + 2*self.padding[0])/self.stride[0] + 1)
        self.out_width = int((self.input_shape[1] - self.kernel_size[1] + 2*self.padding[1])/self.stride[1] + 1)
        
        self.input_conv = nn.Conv2d(in_channels=self.input_dim, out_channels=4*self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  stride = self.stride,
                                  padding=self.padding,
                                  bias=self.bias)
        self.rnn_conv = nn.Conv2d(self.hidden_dim, out_channels=4*self.hidden_dim, 
                                  kernel_size = self.kernel_size,
                                  padding=(math.floor(self.kernel_size[0]/2), 
                                         math.floor(self.kernel_size[1]/2)),
                                  bias=self.bias)
        
        if self.peephole is True:
            self.weight_ci = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.weight_cf = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.weight_co = HadamardProduct((1, self.hidden_dim, self.out_height, self.out_width))
            self.layer_norm_ci = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
            self.layer_norm_cf = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
            self.layer_norm_co = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
        
            
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        
        self.layer_norm_x = nn.LayerNorm([4*self.hidden_dim, self.out_height, self.out_width])
        self.layer_norm_h = nn.LayerNorm([4*self.hidden_dim, self.out_height, self.out_width])
        self.layer_norm_cnext = nn.LayerNorm([self.hidden_dim, self.out_height, self.out_width])
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        x = self.cnn_dropout(input_tensor)
        x_conv = self.input_conv(x)
        if self.layer_norm is True:
            x_conv = self.layer_norm_x(x_conv)
        # separate i, f, c o
        x_i, x_f, x_c, x_o = torch.split(x_conv, self.hidden_dim, dim=1)
        
        h = self.rnn_dropout(h_cur)
        h_conv = self.rnn_conv(h)
        if self.layer_norm is True:
            h_conv = self.layer_norm_h(h_conv)
        # separate i, f, c o
        h_i, h_f, h_c, h_o = torch.split(h_conv, self.hidden_dim, dim=1)
        

        if self.peephole is True:
            f = torch.sigmoid((x_f + h_f) +  self.layer_norm_cf(self.weight_cf(c_cur)) if self.layer_norm is True else self.weight_cf(c_cur))
            i = torch.sigmoid((x_i + h_i) +  self.layer_norm_ci(self.weight_ci(c_cur)) if self.layer_norm is True else self.weight_ci(c_cur))
        else:
            f = torch.sigmoid((x_f + h_f))
            i = torch.sigmoid((x_i + h_i))
        
        
        g = torch.tanh((x_c + h_c))
        c_next = f * c_cur + i * g
        if self.peephole is True:
            o = torch.sigmoid(x_o + h_o + self.layer_norm_co(self.weight_co(c_cur)) if self.layer_norm is True else self.weight_co(c_cur))
        else:
            o = torch.sigmoid((x_o + h_o))
        
        if self.layer_norm is True:
            c_next = self.layer_norm_cnext(c_next)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        height, width = self.out_height, self.out_width
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.input_conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        cnn_dropout, rnn_dropout: float
            cnn_dropout: dropout rate for convolutional input.
            rnn_dropout: dropout rate for convolutional state.
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_sequence: return output sequence or final output only
        bidirectional: bool
            bidirectional ConvLSTM
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two sequences output and state
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(input_dim=64, hidden_dim=16, kernel_size=(3, 3), 
                               cnn_dropout = 0.2,
                               rnn_dropout=0.2, batch_first=True, bias=False)
        >> output, last_state = convlstm(x)
    """

    def __init__(self, img_size, input_dim, hidden_dim, kernel_size,
                 cnn_dropout=0.5, rnn_dropout=0.5,  
                 batch_first=False, bias=True, peephole=False,
                 layer_norm=False,
                 return_sequence=True,
                 bidirectional=False):
        super(ConvLSTM, self).__init__()

        print(kernel_size)
        self.batch_first = batch_first
        self.return_sequence = return_sequence
        self.bidirectional = bidirectional

        cell_fw = ConvLSTMCell(img_size = img_size,
                                 input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 cnn_dropout=cnn_dropout,
                                 rnn_dropout=rnn_dropout,
                                 bias=bias,
                                 peephole=peephole,
                                 layer_norm=layer_norm)
        self.cell_fw = cell_fw
        
        if self.bidirectional is True:
            cell_bw = ConvLSTMCell(img_size = img_size,
                                     input_dim=input_dim,
                                     hidden_dim=hidden_dim,
                                     kernel_size=kernel_size,
                                     cnn_dropout=cnn_dropout,
                                     rnn_dropout=rnn_dropout,
                                     bias=bias,
                                     peephole=peephole,
                                     layer_norm=layer_norm)
            self.cell_bw = cell_bw

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        layer_output, last_state
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state, hidden_state_inv = self._init_hidden(batch_size=b)
            # if self.bidirectional is True:
            #     hidden_state_inv = self._init_hidden(batch_size=b)

        ## LSTM forward direction
        input_fw = input_tensor
        h, c = hidden_state
        output_inner = []
        for t in range(seq_len):
            h, c = self.cell_fw(input_tensor=input_fw[:, t, :, :, :],
                                             cur_state=[h, c])
            
            output_inner.append(h)
        output_inner = torch.stack((output_inner), dim=1)
        layer_output = output_inner
        last_state = [h, c]
        ####################
        
        
        ## LSTM inverse direction
        if self.bidirectional is True:
            input_inv = input_tensor
            h_inv, c_inv = hidden_state_inv
            output_inv = []
            for t in range(seq_len-1, -1, -1):
                h_inv, c_inv = self.cell_bw(input_tensor=input_inv[:, t, :, :, :],
                                                 cur_state=[h_inv, c_inv])
                
                output_inv.append(h_inv)
            output_inv.reverse() 
            output_inv = torch.stack((output_inv), dim=1)
            layer_output = torch.cat((output_inner, output_inv), dim=2)
            last_state_inv = [h_inv, c_inv]
        ###################################
        
        return layer_output if self.return_sequence is True else layer_output[:, -1:], last_state, last_state_inv if self.bidirectional is True else None

    def _init_hidden(self, batch_size):
        init_states_fw = self.cell_fw.init_hidden(batch_size)
        init_states_bw = None
        if self.bidirectional is True:
            init_states_bw = self.cell_bw.init_hidden(batch_size)
        return init_states_fw, init_states_bw