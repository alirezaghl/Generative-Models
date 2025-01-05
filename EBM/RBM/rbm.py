from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class RBMConfig:
    v_dim: int = 784  
    h_dim: int = 256  
    num_steps: int = 1 # CD-1
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-2
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RBM(nn.Module):
    def __init__(self, config):
        super(RBM, self).__init__()
        
        self.config = config
        self.W = nn.Parameter(torch.randn(config.v_dim, config.h_dim).to(config.device) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(config.h_dim).to(config.device))
        self.v_bias = nn.Parameter(torch.zeros(config.v_dim).to(config.device))
    
    def sample_hidden(self, v):
        hidden_prob = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        hidden_samples = torch.bernoulli(hidden_prob)
        return hidden_samples, hidden_prob
    
    def sample_visible(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        v_samples = torch.bernoulli(v_prob)
        return v_samples, v_prob
    
    def gibbs_sampling(self, v):
        v_current = v
        
        for _ in range(self.config.num_steps):
            hidden_samples, _ = self.sample_hidden(v_current)
            v_samples, v_prob = self.sample_visible(hidden_samples)
            v_current = v_samples
        
        hidden_samples, hidden_prob = self.sample_hidden(v_current)
        return v_samples, v_prob, hidden_samples, hidden_prob
    
    def contrastive_divergence(self, v):
        pos_hidden_samples, _ = self.sample_hidden(v)
        neg_visible_samples, _, neg_hidden_samples, _ = self.gibbs_sampling(v)
        
        pos_associations = torch.matmul(v.t(), pos_hidden_samples)
        neg_associations = torch.matmul(neg_visible_samples.t(), neg_hidden_samples)
        
        w_grad = (pos_associations - neg_associations) / v.size(0)
        v_bias_grad = torch.mean(v - neg_visible_samples, dim=0)
        h_bias_grad = torch.mean(pos_hidden_samples - neg_hidden_samples, dim=0)
        
        return w_grad, v_bias_grad, h_bias_grad