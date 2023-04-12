import torch
import torch.nn as nn
from torch.nn import Parameter

# refer to https://arxiv.org/pdf/1706.08276v1.pdf, equations 4-6 
# x means previous time step, y means current time step
# e.g. `hxy` means h_j-1,t; `hyx` means h_j,t-1
class STLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(5 * hidden_size, input_size))
        self.weight_hxy = Parameter(torch.randn(5 * hidden_size, hidden_size))
        self.weight_hyx = Parameter(torch.randn(5 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(5 * hidden_size))
        self.bias_hxy = Parameter(torch.randn(5 * hidden_size))
        self.bias_hyx = Parameter(torch.randn(5 * hidden_size))

    def forward(self, input, stateS, stateT):
        hxy, cxy = stateS
        hyx, cyx = stateT
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hxy, self.weight_hxy.t()) + self.bias_hxy +
                 torch.mm(hyx, self.weight_hyx.t()) + self.bias_hyx)
        ingate, forgetSgate, forgetTgate, cellgate, outgate = gates.chunk(5, 1)

        ingate = torch.sigmoid(ingate)
        forgetSgate = torch.sigmoid(forgetSgate)
        forgetTgate = torch.sigmoid(forgetTgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cyy = ingate * cellgate + forgetSgate * cxy + forgetTgate * cyx
        hyy = outgate * torch.tanh(cyy)

        return hyy, (hyy, cyy)


# refer to equations 7-10
class STLSTMCell_wTrustGate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(5 * hidden_size, input_size))
        self.weight_hxy = Parameter(torch.randn(5 * hidden_size, hidden_size))
        self.weight_hyx = Parameter(torch.randn(5 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(5 * hidden_size))
        self.bias_hxy = Parameter(torch.randn(5 * hidden_size))
        self.bias_hyx = Parameter(torch.randn(5 * hidden_size))
        
        # new trust gate related parameters
        self.weight_hxy_p = Parameter(torch.randn(hidden_size, hidden_size))
        self.weight_hyx_p = Parameter(torch.randn(hidden_size, hidden_size))
        self.weight_ih_p = Parameter(torch.randn(hidden_size, input_size))
        self.bias_hxy_p = Parameter(torch.randn(hidden_size))
        self.bias_hyx_p = Parameter(torch.randn(hidden_size))
        self.bias_ih_p = Parameter(torch.randn(hidden_size))
        self.lambda_ = 0.5 

    def forward(self, input, stateS, stateT):
        hxy, cxy = stateS
        hyx, cyx = stateT
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hxy, self.weight_hxy.t()) + self.bias_hxy +
                 torch.mm(hyx, self.weight_hyx.t()) + self.bias_hyx)
        ingate, forgetSgate, forgetTgate, cellgate, outgate = gates.chunk(5, 1)

        ingate = torch.sigmoid(ingate)
        forgetSgate = torch.sigmoid(forgetSgate)
        forgetTgate = torch.sigmoid(forgetTgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # equation 7
        p = (torch.mm(hxy, self.weight_hxy_p.t()) + self.bias_hxy_p +
             torch.mm(hyx, self.weight_hyx_p.t()) + self.bias_hyx_p)
        p = torch.tanh(p)

        # equation 8
        x_prime = torch.mm(input, self.weight_ih_p.t()) + self.bias_ih_p
        x_prime = torch.tanh(x_prime)

        # equation 9
        tau = torch.exp(-self.lambda_ * torch.square(p - x_prime))

        # equation 10
        cyy = (tau * ingate * cellgate +
               (torch.ones_like(tau) - tau) * forgetSgate * cxy +
               (torch.ones_like(tau) - tau) * forgetTgate * cyx)
        hyy = outgate * torch.tanh(cyy)

        return hyy, (hyy, cyy)



class STLSTMLayer(nn.Module):
    def __init__(self, cell, input_size, hidden_size, device):
        super().__init__()
        self.device = device
        self.cell = cell(input_size, hidden_size).to(device)
        self.input_size = input_size
        self.hidden_size = hidden_size

    # `input`: tensor(T, J, B, input_size), where T = number of time steps,
    # J = number of joints, B = batch size.
    # return tensor(T, J, B, hidden_size) as output.
    def forward(self, input):
        T = input.shape[0]
        J = input.shape[1]
        B = input.shape[2]

        # We traverse along spatial dimension first, then temporal dimension
        # e.g. state(3,2) --> state(4,2) --> ... --> state(J,2) --> state(1,3) --> ... 
        # We will keep track of the latest J states in a queue `temp_states`.
        # Due to the algorithm of "last-to-first link" (see paper), every state(j,t)
        # will take temp_states[-1] as state(j-1,t) and temp_states[0] as state(j,t-1).
        from collections import deque
        empty_state = (torch.zeros(B, self.hidden_size).to(self.device), torch.zeros(B, self.hidden_size).to(self.device))
        temp_states = deque([empty_state] * J, maxlen=J)
        outputs = []
        for t in range(T):
            temp_outputs = []
            for j in range(J):
                out, state = self.cell(input[t][j], temp_states[-1], temp_states[0])
                temp_states.append(state)
                temp_outputs += [out]
            outputs += [torch.stack(temp_outputs)]
        return torch.stack(outputs)


class StackedSTLSTM(nn.Module):
    
    def __init__(self, num_layers, layer, cell, input_size, hidden_size, class_size, device):
        super().__init__()
        layers = [layer(cell, input_size, hidden_size, device).to(device)] + [layer(cell, hidden_size, 
                                        hidden_size, device).to(device) for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(hidden_size, class_size)

    # `input`: tensor(B, T, J, input_size)
    # `output`: tensor(B, class_size, T, J)
    def forward(self, input):
        output = input.permute(1, 2, 0, 3).contiguous() # (B,T,J,input_size) --> (T,J,B,input_size)
        for layer in self.layers:
            output = layer(output)
        output = self.fc(output)

        # calculate log prob
        log_prob = nn.LogSoftmax(dim=-1)(output)
        log_prob = log_prob.permute(2, 3, 0, 1).contiguous() # (T,J,B,C) --> (B,C,T,J)
        
        # calculate prediction
        prob = nn.Softmax(dim=-1)(output)
        prob = prob.mean(dim=(0,1)) # (T,J,B,C) --> (B,C)
        prediction = prob.argmax(dim=-1) # (B,C) --> (B,)

        return log_prob, prediction
