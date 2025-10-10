import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from foot_model import FootForceCNN
from foot_dataset import FootDataset

# -----------------------------
# 1. Load dataset and split
# -----------------------------
dataset = FootDataset("./training_data")

torch.manual_seed(42)  # for reproducibility

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 90/10 split within the training set → train/validation
val_size = int(0.1 * len(train_dataset))
actual_train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(train_dataset, [actual_train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -----------------------------
# 2. Model, loss, optimizer
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FootForceCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 3. Training Loop with Accuracy
# -----------------------------
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_train, total_train = 0, 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # compute training accuracy
        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == y).sum().item()
        total_train += y.size(0)
    
    avg_train_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train

    # Validation
    model.eval()
    val_loss = 0
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_val += (preds == y).sum().item()
            total_val += y.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct_val / total_val

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

# -----------------------------
# 4. Save model
# -----------------------------
torch.save(model.state_dict(), "models/foot_force_classifier.pth")

# -----------------------------
# 5. Final Test Evaluation
# -----------------------------
model.eval()
test_loss = 0
correct_test, total_test = 0, 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = criterion(outputs, y)
        test_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct_test += (preds == y).sum().item()
        total_test += y.size(0)

avg_test_loss = test_loss / len(test_loader)
test_acc = correct_test / total_test

print("\n✅ Test Results:")
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")
