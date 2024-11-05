import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.LeakyReLU(),

            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.relu = nn.Tanh() #tanh good?
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        proj = self.proj(x)
        x = self.relu(proj)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + proj
        x = self.layer_norm(x)
        return x