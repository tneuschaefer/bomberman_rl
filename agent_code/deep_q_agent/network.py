import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQ(nn.Module):

    def __init__(self, state_shape, actions_shape):
        super(DeepQ, self).__init__()
        
        self.layer1 = nn.Linear(state_shape, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, actions_shape)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def predict(self, state):
        with torch.no_grad():
            return self(state).max(1)[1].view(1, 1)