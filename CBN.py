import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class CBN(nn.Module):

    def __init__(self, n_category, n_hidden, num_features, eps=1e-5, momentum=0.9, is_training=True):
        super(CBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.is_training = is_training

        #Affine transform parameters
        self.gamma = Parameter(torch.Tensor(num_features), requires_grad = True)
        self.beta = Parameter(torch.Tensor(num_features), requires_grad = True)
        
        #Running mean and variance, these parameters are not trained by backprop
        self.running_mean = Parameter(torch.Tensor(num_features), requires_grad = False)
        self.running_var = Parameter(torch.Tensor(num_features), requires_grad = False)
        self.num_batches_tracked = Parameter(torch.Tensor(1), requires_grad = False)      
        
        #Parameter initilization
        self.reset_parameters()
     
        #MLP parameters
        self.n_category = n_category
        self.n_hidden = n_hidden
        
        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.n_category, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.num_features),
            )

        self.fc_beta = nn.Sequential(
            nn.Linear(self.n_category, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.num_features),
            )

        # Initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0.1)

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        self.gamma.data.uniform_()
        self.beta.data.zero_()

    def forward(self, input, category_one_hot):
        
        N, C, H, W = input.size()
        
        exponential_average_factor = 0.0
        if self.is_training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked
            else:  # use exponential moving average
                exponential_average_factor = 1 - self.momentum

        # Obtain delta values from MLP
        delta_gamma = self.fc_gamma(category_one_hot)
        delta_beta = self.fc_beta(category_one_hot)
        
        gamma_cloned = self.gamma.clone()
        beta_cloned = self.beta.clone()
        
        gamma_cloned = gamma_cloned.view(1,C).expand(N,C)
        beta_cloned = beta_cloned.view(1,C).expand(N,C)

        # Update the values
        gamma_cloned += delta_gamma
        beta_cloned += delta_beta

        # Standard batch normalization 
        out, running_mean, running_var = batch_norm(input, self.running_mean, self.running_var, gamma_cloned, beta_cloned,
            self.is_training, exponential_average_factor, self.eps)
        
        if self.is_training:
            self.running_mean.data = running_mean.data
            self.running_var.data = running_var.data

        return out
        
        
def batch_norm(input, running_mean, running_var, gammas, betas,
            is_training, exponential_average_factor, eps):
        # Extract the dimensions
        N, C, H, W = input.size()

        # Mini-batch mean
        mean = torch.mean(input.view(C,-1), dim=1)
        # Mini-batch variance
        variance = torch.mean(((input - mean.view(1,C,1,1).expand((N, C, H, W))) ** 2).view(C,-1), dim=1)

        # Normalize
        if is_training:
            
            #Compute running mean and variance
            running_mean = running_mean*(1-exponential_average_factor) + mean*exponential_average_factor
            running_var = running_var*(1-exponential_average_factor) + variance*exponential_average_factor
        
            # Training mode, normalize the data using its mean and variance
            X_hat = (input - mean.view(1,C,1,1).expand((N, C, H, W))) * 1.0 / torch.sqrt(variance.view(1,C,1,1).expand((N, C, H, W)) + eps)
        else:
            # Test mode, normalize the data using the running mean and variance
            X_hat = (input - running_mean.view(1,C,1,1).expand((N, C, H, W))) * 1.0 / torch.sqrt(running_var.view(1,C,1,1).expand((N, C, H, W)) + eps)
                 
        # Scale and shift
        out = gammas.contiguous().view(N,C,1,1).expand((N, C, H, W)) * X_hat + betas.contiguous().view(N,C,1,1).expand((N, C, H, W))        
        
        return out, running_mean, running_var
        


if __name__ == '__main__':

    model = CBN(2,2,128).cuda()
    x = torch.ones([4,128,16,16])
    one_hot = torch.zeros([4,2])
    one_hot[0,0] = 1
    one_hot[1,1] = 1
    one_hot[2,1] = 1
    one_hot[3,0] = 1
    
    x = Variable(x.cuda())
    one_hot = Variable(one_hot.cuda())
    
    print(model(x,one_hot))
    
