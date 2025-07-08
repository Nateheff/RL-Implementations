import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *


class Policy(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        # self.hidden_size = 128
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, out_features=256)
        
        self.out = nn.Linear(in_features=256, out_features=6)
        self.probs = nn.Softmax(dim=-1)
        self.adam = optim.Adam(self.parameters())

        
    def forward(self, x:torch.Tensor):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)        
        out = self.out(x)
        probs = self.probs(out)
        return out, probs


policy = Policy(16, 8, 4)
delta = 0.05

def fisher_vector_product(v, states, damping=1e-2): 
    """
    Compute Fisher Information Matrix vector product using KL divergence

    Fisher Information Matrix is the Hessian (matrix of second-order derivatives) of the KL divergence
    of the new and old policies.

    We are computing the FIM vector product which is Hessian of the KL divergences times some v
    F is the second order derivatives of the KL Divergence
    v is a vector (direction) that is used to update x to satisfy Fx = g

    We want this estimation of the FIM since the FIM would be of size # paramters x # parameters.
    Here we are computing the Fisher vector product (Fv) by:
        1. Calculating some scalar function of the two policies.
        NOTE: While it may appear we're only using our current policy for the first-order derivative function,
        we ar using samples according to the old policy thus evaluating the expectation using these samples
        from the old policy
        2. Multiply these gradients by our direction v to get gv
        3. Compute the gradients of gv wrt. the policy once more
        4. Add a regularization factor and return.

    REUTRN: We are returning the Fisher vector product Fv
    """
    # Get current policy probabilities
    _, probs = policy(states)
    old_probs = probs.detach()
    old_logs = torch.log(old_probs + 1e-8)
    # Compute KL divergence 
    log_probs = torch.log(probs + 1e-8)
    kl = (old_probs * (old_logs - log_probs)).sum(dim=-1).mean() #E(a~pi_old)[log(pi_old(a)/pi(a))]
    
    # Compute gradient of KL w.r.t. policy parameters
    kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    g_flat = torch.cat([grad.view(-1) for grad in kl_grad])
    

    gv = torch.dot(g_flat, v)
    #FVP ∇_θ (∇_θ f(θ) · v) = Hessian_f(θ) * v where f = KL divergence
    fvp = torch.autograd.grad(gv, policy.parameters(), retain_graph=True)
    flat_fvp = torch.cat([f.reshape(-1) for f in fvp])

    regularization = damping * v

    return flat_fvp + regularization #Fv

def conjugate_gradient(g, states, n_steps, residual_min=1e-10):
    """
    Solves F x = g using the Conjugate Gradient method.
    
    Args:
        
        g: right-hand side vector (gradient of surrogate loss)
        nsteps: maximum number of iterations
        residual_tol: tolerance for convergence
        v: A direction used to calculate Fv and update x to satisfy Fx = g
        V explores the curavture in F increments x to match g.

    Returns:
        x: solution to Fx = g (the step direction for our update; the natural gradient)
    """

    x = torch.zeros_like(g)
    r = g.clone()
    v = r.clone()
    rs_old = torch.dot(r, r)

    for i in range(n_steps):
        fv = fisher_vector_product(v, states)
        alpha = rs_old / torch.dot(v, fv)
        x += alpha * v
        r -= alpha * fv
        rs_new = torch.dot(r, r)

        if rs_new < residual_min:
            break

        v = r + (rs_new / rs_old) * v # V is encoded with the curvature of F via this update which is used to update x
        rs_old = rs_new
    return x


def TRPO(steps):
    global policy

    """
    1. We collect a batch of states, actions, advantages, log probs, and rewards following the old policy

    2. We then use our current policy to get new log probs for these same states and actions

    3. We then calculate our surrogate loss
    NOTE: This loss does not directly optimize rewards, it instead looks at how much the probability of actions
    changed weighted by how good those actions were. We are using gradient ascent so we are trying to maximize this.
    
    4. We then compute our first order derivatives

    5. We then compute out step direction using Conjugate Gradient and Fisher vector product to
    avoid having to calcualte the full fisher information matrix.
    Our update is bounded by the KL divergence of the two policies and this bound is represented in
    the update as step^T * F * step <= 2 * delta
    
    6.After we calcualte our step, we update our parameters 
    
    We use an additional Adam update since our model is also returning Q_values.
    """
    for i in range(steps):
        states, actions, advantages, log_probs_old, rewards = get_batches_TRPO(2048, policy)
        
        q_values, probs = policy(states)
        
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        #Calculate ratios of new and old action probability distributions and normalize advantages
        ratio = torch.exp(log_probs - log_probs_old)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        #Calculate surrogate loss (negative for gradient ascent and mean due to expectation)
        loss = -(ratio * advantages).mean()

        #compute policy gradient
        g = torch.autograd.grad(loss, policy.parameters())
        #detatch to have g_flat treated as a constant in Pytorch gradient calculation and backprop
        g_flat = torch.cat([grad.view(-1) for grad in g]).detach()

        F_g = conjugate_gradient(g_flat, states ,10) #step direction

        g_Fg = torch.dot(g_flat, F_g) 

        trust_scale = torch.sqrt(2 * delta / (g_Fg + 1e-8))
        step = trust_scale * F_g

        with torch.no_grad():
            offset = 0
            for param in policy.parameters():
                n_param = param.numel()
                step_segment = step[offset: offset + n_param].view_as(param)
                param += step_segment
                offset += n_param
        
        with torch.no_grad():
            states_detached = states.detach()
        

        q_values_new, _ = policy(states_detached)
        values = q_values_new.gather(1, actions.unsqueeze(1)).squeeze()
        returns = torch.tensor(rewards)

        q_loss = F.mse_loss(values, returns)
        policy.adam.zero_grad()
        q_loss.backward()
        policy.adam.step()


TRPO(5)