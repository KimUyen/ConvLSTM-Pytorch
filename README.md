## ConvLSTM Pytorch Implementation
<img align="right" width="567" height="335" src="/figures/BCI_system.png"> <br>
- [Goal](#Goal) <br>
- [Example of using ConvLSTM](#Example-of-using-ConvLSTM) <br>
- [Explaination](#Explaination) <br>
    - [1. ConvLSTM definition](#1-ConvLSTM-definition) <br>
    - [2. Bidirectional ConvLSTM decoder](#2-bidirectional-convlstm-decoder) <br>
    - [3. Input, output for decoder](#3-input-output-for-decoder) <br>
- [Environment](#Environment) <br>
- [References](#References) <br> 

## Goal
The ConvLSTM model is mainly used as skeleton to design a BCI (Brain Computer Interface) decoder for our project (Decode the kinematic signal from neural signal).
This repo is implementation of ConvLSTM in Pytorch. The implemenation is inherited from the paper: Convolutional LSTM Network-A Machine LearningApproach for Precipitation Nowcasting

BCI decoder is a part in BCI system, which is clearly shown in the above figure.

## Example of using ConvLSTM
convlstm_decoder.py contains an example of defining a ConvLSTM decoder.

Here is an example of defining 1 layer bidirectional ConvLSTM:
```
        convlstm_layer = []
        num_layers = 1              # number of layer
        input_channel = 96          # the number of electrodes in Utah array
        hidden_channels = [256]     # the output channels for each layer
        kernel_size = [(7, 7)]      # the kernel size of cnn for each layer
        stride = [(1, 1)]           # the stride size of cnn for each layer
        padding = [(0, 0)]          # padding size of cnn for each layer
        for i in range(num_layers):
            layer = convlstm.ConvLSTM(input_dim=input_channel, 
                                         hidden_dim=hidden_channels[i],
                                         kernel_size=kernel_size[i],
                                         stride=stride[i],
                                         padding=padding[i],
                                         cnn_dropout=0.2, 
                                         rnn_dropout=0.,
                                         batch_first=True, 
                                         bias=True, 
                                         peephole=False, 
                                         batch_norm=False,
                                         layer_norm=False,
                                         return_sequence=True,
                                         bidirectional=True)
            convlstm_layer.append(layer)  
            input_channel = hidden_channels[i]
```

## Explaination
The imlementation firstly was inherited from  [the repo](https://github.com/ndrplz/ConvLSTM_pytorch).

However, I changed the source to have more exactly to the original paper [1].
### 1. ConvLSTM definition
Which are following in the paper definition:
<p align="center">
    <img src="/figures/ConvLSTM_definition.png">
</p>

The ConvLSTM Cell is defined as following figure:
![](/figures/ConvLSTM_cell.png)

### 2. Bidirectional ConvLSTM decoder
Our BCI decoder is a 5 timesteps bidirectional ConvLSTM, which contains two ConvLSTM layer: a forward layer to learn direction from left to right input, a backward layer to learn direction from right to left input. Detail in following figure:
![](/figures/BCI_decoder.png)

### 3. Input, output for decoder
The input of our decoder is spike count or LMP, and output is velocity.
![](/figures/input_output_decoder.png)

## Environment
This repository is tested on Python 3.7.0, Pytorch 1.6.0

## References
[1] Xingjian, S. H. I., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In Advances in neural information processing systems (pp. 802-810).
