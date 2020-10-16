import torch
import convlstm
import config


class Flatten(torch.nn.Module):
    def forward(self, input):
        b, seq_len, _, h, w = input.size()
        return input.view(b, seq_len, -1)

class ConvLSTMNetwork(torch.nn.Module):
    def __init__(self, input_channel, hidden_channels, kernel_size, stride, padding, num_layers):
        super(ConvLSTMNetwork, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        convlstm_layer = []
        for i in range(num_layers):
            layer = convlstm.ConvLSTM(input_channel, 
                                         hidden_channels[i],
                                         kernel_size[i],
                                         stride[i],
                                         padding[i],
                                         0.2, 0.,
                                         batch_first=True, 
                                         bias=True, 
                                         peephole=False, 
                                         batch_norm=False,
                                         layer_norm=False,
                                         return_sequence=config.SEQUENCE_OUTPUT,
                                         bidirectional=True)
            convlstm_layer.append(layer)
            input_channel = hidden_channels[i]
           
        self.convlstm_layer = torch.nn.ModuleList(convlstm_layer)
        self.flatten = Flatten()
        self.linear = torch.nn.Linear(256*16*2, 2)
    
    def forward(self, x):
        input_tensor = x
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
       
        out_flatten = self.flatten(input_tensor)
        output = self.linear(out_flatten)
        return output
    
