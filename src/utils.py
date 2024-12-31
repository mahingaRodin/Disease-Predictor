import torch.nn as nn


def define_model(input_size, output_size):
    class HealthNet(nn.Module):  # Corrected to 'nn.Module'
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

    # Return an instance of HealthNet
    return HealthNet(input_size, output_size)


def generate_advice(predicted_class):
    advice_map = {
        0: "Maintain a healthy diet and exercise regularly.",
        1: "Consult your doctor about your heart condition.",
        2: "Monitor your cholesterol and blood pressure levels.",
        3: "Consider lifestyle changes to improve your health.",
        4: "Get regular checkups to manage your health effectively."
    }
    return advice_map.get(predicted_class, "No advice available.")
