import torch
from collections import OrderedDict

class MLP(torch.nn.Module):
    def __init__(self, hidden_sizes, dropout = 0):
        super(MLP, self).__init__()
        order_dict = []
        for i in range(len(hidden_sizes) - 1):
            order_dict.append(('linear_layer_{}'.format(i), torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1])))
            if i < len(hidden_sizes) - 2:
                order_dict.append(('batchnorm_layer_{}'.format(i), torch.nn.BatchNorm1d(hidden_sizes[i+1])))
                order_dict.append(('relu_layer_{}'.format(i), torch.nn.ReLU()))
                if dropout > 0:
                    order_dict.append(('dropout_layer_{}'.format(i), torch.nn.Dropout(dropout)))
        self.mlp = torch.nn.Sequential(OrderedDict(order_dict))
    
    def forward(self, x):
        return self.mlp(x)

class RNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.i2h = torch.nn.Linear(input_size, hidden_size).to(device)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size).to(device)
    
    def forward(self, x, hidden):
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_size).to(x.device)
        inter = self.i2h(x)
        inter_hidden = self.h2h(hidden)
        hidden = torch.nn.functional.tanh(inter + inter_hidden)
        return hidden

class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.i2h = torch.nn.Linear(input_size, 4 * hidden_size).to(device)
        self.h2h = torch.nn.Linear(hidden_size, 4 * hidden_size).to(device)
    
    def forward(self, x, hidden):
        if hidden is None:
            hidden = (torch.zeros(x.shape[0], self.hidden_size).to(x.device), torch.zeros(x.shape[0], self.hidden_size).to(x.device))
        elif type(hidden) == torch.Tensor:
            hidden = (hidden, torch.zeros(x.shape[0], self.hidden_size).to(x.device))
        else:
            hidden = tuple(hidden)
        hx, cx = hidden
        gates = self.i2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.nn.functional.sigmoid(ingate)
        forgetgate = torch.nn.functional.sigmoid(forgetgate)
        cellgate = torch.nn.functional.tanh(cellgate)
        outgate = torch.nn.functional.sigmoid(outgate)
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.nn.functional.tanh(cy)
        return hy, cy

class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, dropout = 0):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnncell_list = [RNNCell(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            self.rnncell_list.append(('rnn_layer_{}'.format(_+1), RNNCell(hidden_size, hidden_size)))
        self.rnncell_list = torch.nn.ModuleList(self.rnncell_list)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden = None):
        if hidden is None:
            hx = [None for _ in range(self.num_layers)]
        elif type(hidden) == torch.Tensor:
            hx= [None for _ in range(self.num_layers)]
            hx[0] = hidden
        else:
            hx = hidden
        hx[0] = self.rnncell_list[0](x, hx[0])
        for i in range(1, self.num_layers):
            input_t = self.dropout(hx[i - 1])
            # hx[i] = self.rnncell_list[i](hx[i - 1], hx[i])
            hx[i] = self.rnncell_list[i](input_t, hx[i])
        x = self.fc(hx[-1])
        return x, hx
    
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, dropout = 0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstmlist = [LSTMCell(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            self.lstmlist.append(LSTMCell(hidden_size, hidden_size))
        self.lstmlist = torch.nn.ModuleList(self.lstmlist)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden = None):
        if hidden is None:
            hx = [None for _ in range(self.num_layers)]
        elif type(hidden) == torch.Tensor:
            hx= [None for _ in range(self.num_layers)]
            hx[0] = hidden
        else:
            hx = hidden
        hx[0] = list(self.lstmlist[0](x, hx[0]))
        for i in range(1, self.num_layers):
            # hx[i] = self.lstmlist[i](hx[i - 1][0], hx[i])
            input_t = self.dropout(hx[i - 1][0])
            hx[i] = list(self.lstmlist[i](input_t, hx[i]))
        x = self.fc(hx[-1][0])
        return x, hx

class LSTMEncoderDecoder2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, window_length, prediction_length, dropout = 0.2, num_layers = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.window_length = window_length
        self.prediction_length = prediction_length
        self.encoder = LSTMModel(input_size, hidden_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMModel(output_size, hidden_size, output_size, num_layers, dropout)
        
    def encoder_forward(self, x):
        hidden = None
        for i in range(self.window_length):
            _, hidden = self.encoder(x[..., i], hidden)
        return hidden
    
    def decoder_forward(self, inter, hidden):
        inter, hidden = self.decoder(inter, hidden)
        return inter, hidden
    
    def forward(self, x, var = False):
        hidden = self.encoder_forward(x)
        decoder_hidden = hidden[self.num_layers- 1][0]
        inter = x[..., -1]
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        predictions = torch.zeros(x.shape[0], self.output_size, self.prediction_length).to(device)
        mc_variance = torch.zeros(x.shape[0], self.output_size, self.prediction_length).to(device) if var else None
        for i in range(self.prediction_length):
            # temp_inters = torch.zeros(x.shape[0], self.output_size, 10)
            inter, decoder_hidden = self.decoder_forward(inter, decoder_hidden)
            if var:
                temp_inters = torch.zeros(x.shape[0], self.output_size, 10)
                for j in range(10):
                    inter_2, decoder_hidden_2 = self.decoder_forward(inter, decoder_hidden)
                    temp_inters[..., j] = inter_2
                mc_variance[..., i] = torch.std(temp_inters, dim=2)
                inter = torch.mean(temp_inters, dim=2)
            predictions[..., i] = inter
        # return predictions, mc_variance
        return {"predictions": predictions, "mc_variance": mc_variance}
    
class LSTMEncoderDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, window_length, prediction_length, dropout = 0.2, num_layers = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.window_length = window_length
        self.prediction_length = prediction_length
        self.encoder = LSTMModel(input_size, hidden_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMModel(output_size, hidden_size, output_size, num_layers, dropout)
        
    def encoder_forward(self, x):
        hidden = None
        for i in range(self.window_length):
            _, hidden = self.encoder(x[..., i], hidden)
        return hidden
    
    def decoder_forward(self, inter, hidden):
        return self.decoder(inter, hidden)
    
    def forward(self, x, var = False, hidden= False):
        hidden_e = self.encoder_forward(x)
        decoder_hidden = hidden_e[self.num_layers- 1][0]
        inter_f = x[..., -1]
        predictions = torch.zeros(x.shape[0], self.output_size, self.prediction_length).to(x.device)
        mc_variance = torch.zeros(x.shape[0], self.output_size, self.prediction_length).to(x.device) if var else None
        for i in range(self.prediction_length):
            if var:
                temp_inters = torch.zeros(x.shape[0], self.output_size, 10)
                for j in range(10):
                    if i != 0:
                        temp_hidden = [[h[0].clone(), h[1].clone()] for h in decoder_hidden]
                    else:
                        temp_hidden = decoder_hidden.clone()
                    temp_inters[..., j], _ = self.decoder_forward(inter_f, temp_hidden)
                mc_variance[..., i] = torch.std(temp_inters, dim=2)
            inter_f, decoder_hidden = self.decoder_forward(inter_f, decoder_hidden)
            predictions[..., i] = inter_f
        output = {"predictions": predictions}
        if var:
            output["mc_variance"] = mc_variance
        if hidden:
            output["encoder_hidden_states"] = hidden_e
            output["decoder_hidden_states"] = decoder_hidden
        return output
