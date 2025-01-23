import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
file_path = 'combined_with_mse - Copy.csv'
data = pd.read_csv(file_path)

# Extract input features and output MSE values
response_spectrum_values = data.iloc[:8, :].values  # First 8 rows are input features
mse_values = data.iloc[8, :].values  # 9th row contains MSE values

# Ensure dimensions match
assert mse_values.shape[0] == response_spectrum_values.shape[1],

# Print data shapes
print(f"Input feature data shape: {response_spectrum_values.shape}")
print(f"MSE data shape: {mse_values.shape}")

# Normalize input data
input_scaler = StandardScaler()
response_spectrum_normalized = input_scaler.fit_transform(response_spectrum_values.T).T  # Normalize each column

# Normalize output data
output_scaler = StandardScaler()
mse_values_normalized = output_scaler.fit_transform(mse_values.reshape(-1, 1)).flatten()

# Save normalization parameters
joblib.dump(input_scaler, 'input_scaler.pkl')
joblib.dump(output_scaler, 'output_scaler.pkl')
print("Normalization parameters saved.")

# Convert to PyTorch tensors
response_spectrum_tensor = torch.tensor(response_spectrum_normalized, dtype=torch.float32).to(device)
mse_values_tensor = torch.tensor(mse_values_normalized, dtype=torch.float32).view(-1, 1).to(device)

# Define the proxy model
class ProxyModel(nn.Module):
    def __init__(self):
        super(ProxyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 64),  # 输入层修改为8个特征
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# Instantiate the model and optimizer
proxy_model = ProxyModel().to(device)
proxy_optimizer = optim.Adam(proxy_model.parameters(), lr=0.0002)
criterion = nn.MSELoss()

# Train the proxy model
num_epochs = 15000
for epoch in range(num_epochs):
    proxy_model.train()
    proxy_optimizer.zero_grad()
    predictions = proxy_model(response_spectrum_tensor.T)
    loss = criterion(predictions, mse_values_tensor)
    loss.backward()
    proxy_optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Proxy Loss: {loss.item()}')

print("Proxy model training complete.")