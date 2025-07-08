import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from utils import *
import multiprocessing as mp
import gymnasium as gym
import ale_py

ALPHA = 0.99
LR = 0.001

"""
ACKTR (Actor Critic Kronecker-Factored Trust Region)
This algorithm uses a similar method to A3C, however it uses an approximation
of the natural gradient update which includes the Fisher Information Matrix (FIM)
in the update. 
The FIM accounts for (encodes) the curvature of the parameter space into the update
where curvature is measured by the policy's output distributions sensitivity to cahnges
in parameters. 
The FIM is useful because vanilla SGD treats the parameter space as though it is a 
flat Euclidian space. Encoding curvature allows for more informed updates
The FIM approximate is calculated on a layer-wise basis as are updates.
"""

"""
Implementation Notes:

We need to first fix our batches.
1. Collect batches with information about 
"""

class LSTM(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.concat_size = hidden_size + input_size
        """
        Using one set of weights is mor efficient and empirically doesn't impact performance significantly.
        The alternative is having a separate set of parameters (in the form of a nn.Linear(self.concat_size, hidden_size) for each gate)
        """
        self.weights = nn.Linear(self.concat_size, 4*hidden_size) # 4x because of chucnking

    def forward(self, x, previous_hidden, previous_cell):
        
        x = torch.cat((x,previous_hidden), dim=-1)

        state = self.weights(x)
        forget, input, output, candidate = torch.chunk(state, chunks=4, dim=-1)

        """
        The candidate will suggest what information the new cell should have (considering information from the previous hidden state and the new input),
        the input gate will decide what and how much of this proposed new state to keep,
        the forget gate will remove the unnecessary parts of the old memory,
        and the output finally decides which parts should be out.
        """
        forget = torch.sigmoid(forget)
        input = torch.sigmoid(input)
        output = torch.sigmoid(output)
        tan = torch.tanh(candidate)

        cell = forget * previous_cell + input * torch.tanh(candidate)
        hidden = output * torch.tanh(cell)
        
        """
        The hidden state is the LSTM layer's current output. It is more relevant to the current input.
        The cell state is the long-term memory that holds the previous information that is used to calculate 
        the hidden state.
        """
        return hidden, cell



class ActorCritic(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        self.hidden_size = 128
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, out_features=256)
        self.lstm = LSTM(self.hidden_size, input_size=256)
        self.value = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.policy_lin = nn.Linear(in_features=self.hidden_size, out_features=6)
        self.policy = nn.Softmax(dim=-1)
        
        # Initialize RMSprop running averages for each parameter
        self.running_avg = {}
        for name, param in self.named_parameters():
            self.running_avg[name] = torch.zeros_like(param.data)
        

        
    def forward(self, x:torch.Tensor, hidden_state=None, cell_state=None):

        if hidden_state is None:
            hidden_state = torch.zeros((x.shape[0], self.hidden_size))
        if cell_state is None:
            cell_state = torch.zeros((x.shape[0], self.hidden_size))

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)
        new_hidden, new_cell = self.lstm(x, hidden_state, cell_state)
        
        value = self.value(new_hidden)
        policy_down = self.policy_lin(new_hidden)
        policy = self.policy(policy_down)
        return value, policy, new_hidden, new_cell
    
global_model = ActorCritic(16, 8, 4)
global_model.share_memory()

# Share the RMSprop running averages across processes
for name, running_avg in global_model.running_avg.items():
    running_avg.share_memory_()

n = 5

def save_input_hook(module, input, output):
    """
    This saves the input to each layer which is the activations of the preivous layer. 
    For example, say we have a simple model with Linear -> ReLU -> Linear,
    The activation for the first Linear is the input data,
    The activiation for the second Linear is the output of the ReLU
    """
    module.input_activation = input[0].detach() 
    
def save_grad_hook(module, grad_input, grad_output):
    """
    This saves the gradient passed down to each layer from the next layer
    Using our Linear -> ReLU -> Linear example,
    the output gradient of each layer is the gradient of the loss wrt the activation of that layer.
    it will be of shape [batch_size, output_size]
    """
    module.output_gradient = grad_output[0].detach()
 

def register_kfac_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.register_forward_hook(save_input_hook)
            module.register_full_backward_hook(save_grad_hook)
        elif isinstance(module, LSTM):
            module.weights.register_forward_hook(save_input_hook)
            module.weights.register_full_backward_hook(save_grad_hook)



def KFac(input, gradient):

    G = gradient.mT @ gradient

    A = input.mT @ input

    G_i = torch.linalg.inv(G + 1e-5 * torch.eye(G.shape[0]))
    A_i = torch.linalg.inv(A + 1e-5 * torch.eye(A.shape[0]))

    return G_i, A_i

def ACKTR(global_params):
    optimizer = torch.optim.RMSprop(global_params.parameters(), lr=1e-4)

    local_model = ActorCritic(16, 8, 4)
    local_model.load_state_dict(global_params.state_dict())
    register_kfac_hooks(local_model)
    
    for name in local_model.running_avg:
        local_model.running_avg[name] = global_params.running_avg[name]
    

    for _ in range(50):


        local_model.load_state_dict(global_params.state_dict())


        hidden_state = torch.zeros(local_model.hidden_size)
        cell_state = torch.zeros(local_model.hidden_size)

        
        values, actions, advantages, log_probs, returns = get_batches_ACKTR(256, local_model, 128)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        values = values.squeeze()

        loss_policy = -(log_probs * advantages).sum()

        loss_value = F.mse_loss(values, returns)  # shape match

        loss = loss_policy + loss_value

        #Compute new gradients
        
        local_model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for name, module in local_model.named_modules():
                if isinstance(module, nn.Linear):
                    input_activation = getattr(module, 'input_activation')
                    output_gradient = getattr(module, 'output_gradient')

                    if input_activation is None or output_gradient is None:
                        continue
                    
                    batch_size = input_activation.shape[0]
                    output_gradient = output_gradient.view(batch_size, -1)
                    input_activation = input_activation.view(batch_size, -1)

                    G_inv, A_inv = KFac(input_activation, output_gradient)

                    for param in module.parameters():
                        raw_grad = param.grad.data        
                        if raw_grad.dim() == 2:
                            natural_grad = G_inv @ raw_grad @ A_inv
                        else:  # bias
                            natural_grad = G_inv @ raw_grad.unsqueeze(1)
                            natural_grad = natural_grad.squeeze(1)
                        param.data -= LR * natural_grad
                    

                elif isinstance(module, nn.Conv2d):
                    input_activation = getattr(module, 'input_activation')
                    output_gradient = getattr(module, 'output_gradient')

                    if input_activation is None or output_gradient is None:
                        continue

                    batch_size = input_activation.shape[0]
                    C_out = output_gradient.shape[1]

                    output_gradient.permute(0, 2, 3, 1)
                    output_gradient = output_gradient.view(-1, C_out)

                    input_activation = torch.nn.functional.unfold(
                        input_activation, 
                        kernel_size=module.kernel_size, 
                        stride=module.stride, 
                        padding=module.padding
                        )
                    input_activation = input_activation.permute(0, 2, 1).contiguous()
                    input_activation = input_activation.view(-1, input_activation.shape[-1])

                    G_inv, A_inv = KFac(input_activation, output_gradient)

                    for param in module.parameters():
                        raw_grad = param.grad.data
                        if raw_grad.dim() == 4:  # Weight tensor
                            # raw_grad: [out_channels, in_channels, kernel_h, kernel_w]
                            # Reshape to: [out_channels, in_channels * kernel_h * kernel_w]
                            original_shape = raw_grad.shape
                            raw_grad_reshaped = raw_grad.view(original_shape[0], -1)
                            
                            # Apply K-FAC update
                            natural_grad = G_inv @ raw_grad_reshaped @ A_inv
                            natural_grad = natural_grad.view(original_shape)
                            
                        else:  # Bias term
                            # raw_grad: [out_channels]
                            natural_grad = G_inv @ raw_grad.unsqueeze(1)
                            natural_grad = natural_grad.squeeze(1)

                        param.data -= LR * natural_grad

                elif isinstance(module, LSTM):
                    lstm_linear = module.weights

                    input_activation = getattr(lstm_linear, 'input_activation')
                    output_gradient = getattr(lstm_linear, 'output_gradient')

                    if input_activation is None or output_gradient is None:
                        continue

                    output_gradient = output_gradient.view(output_gradient.shape[0], -1)
                    input_activation = input_activation.view(input_activation.shape[0], -1)
                    G_inv, A_inv = KFac(input_activation, output_gradient)

                    for param in lstm_linear.parameters():
                        raw_grad = param.grad.data
                        
                        if raw_grad.dim() == 2:
                            natural_grad = G_inv @ raw_grad @ A_inv
                        else:  # bias
                            natural_grad = G_inv @ raw_grad.unsqueeze(1)
                            natural_grad = natural_grad.squeeze(1)
                        param.data -= LR * natural_grad

        optimizer.step()
        optimizer.zero_grad()
        local_model.zero_grad()

        



def learn_async():
    try:
        workers = []
        
        for _ in range(1):
          
            
            
            p = mp.Process(target=ACKTR, args=(global_model,))

            p.start()
            workers.append(p)

        for p in workers:
            p.join()

        
    except Exception as e:
        for p in workers:
            p.terminate()
            p.join()
        print(e)
    
if __name__ == "__main__":
    learn_async()