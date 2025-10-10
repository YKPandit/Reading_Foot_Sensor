import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from foot_model import FootForceCNN
from foot_dataset import FootDataset

dataset = FootDataset("./training_data")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FootForceCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/30], Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "models/foot_force_classifier.pth")