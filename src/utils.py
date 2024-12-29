import torch.nn as nn

def define_model(input_size, output_size):
    class HealthNet(nn.module):
        def __init__(self, input_size, output_size):
            super(HealthNet, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, output_size)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.softmax(self.fc3(x))
            return x
        



            
