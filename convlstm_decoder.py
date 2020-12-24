import torch
import convlstm
import config


class Flatten(torch.nn.Module):
    def forward(self, input):
        b, seq_len, _, h, w = input.size()
        return input.view(b, seq_len, -1)

class ConvLSTMNetwork(torch.nn.Module):
    def __init__(self, img_size_list, input_channel, hidden_channels, kernel_size, num_layers, bidirectional = False):
        super(ConvLSTMNetwork, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        convlstm_layer = []
        for i in range(num_layers):
            layer = convlstm.ConvLSTM(img_size_list[i],
                                    input_channel, 
                                    hidden_channels[i],
                                    kernel_size[i],
                                    0.2, 0.,
                                    batch_first=True, 
                                    bias=True, 
                                    peephole=True,
                                    layer_norm=True,
                                    return_sequence=config.SEQUENCE_OUTPUT,
                                    bidirectional=self.bidirectional)
            convlstm_layer.append(layer)
            input_channel = hidden_channels[i] * (2 if self.bidirectional else 1)
           
        self.convlstm_layer = torch.nn.ModuleList(convlstm_layer)
        self.flatten = Flatten()
        self.linear2 = torch.nn.Linear(hidden_channels[-1]*(2 if self.bidirectional else 1)*16, 2)
    
    def forward(self, x):
        input_tensor = x
        for i in range(self.num_layers):
            input_tensor, _, _ = self.convlstm_layer[i](input_tensor)
       
        out_flatten = self.flatten(input_tensor)
        output = self.linear2(out_flatten)
        return output
    
