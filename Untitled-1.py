# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import math, time

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to [1, 28, 28] and scales to [0, 1]
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE)


# %%
import matplotlib.pyplot as plt

# Get a few sample images and labels
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

# Show first 6 images in the batch
plt.figure(figsize=(10, 3))
for i in range(6):
    plt.subplot(1, 6, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
    plt.title(f"Label: {example_targets[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# %%
print("Image batch shape:", example_data.shape)   # [64, 1, 28, 28]
print("Label batch shape:", example_targets.shape)  # [64]

print("Number of training images:", len(train_data))
print("Number of test images:", len(test_data))


# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


# %%
class MNISTTransformer(nn.Module):
    def __init__(self, input_dim=28, model_dim=128, num_heads=4, num_layers=2, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=256,  batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = x.squeeze(1)              # (B, 28, 28)
        x = self.embedding(x)         # (B, 28, model_dim)
        x = self.pos_enc(x)           # (B, 28, model_dim)
        x = x.permute(1, 0, 2)        # (28, B, model_dim)
        x = self.transformer(x)       # (28, B, model_dim)
        x = x.mean(dim=0)             # (B, model_dim)
        return self.classifier(x)     # (B, 10)


# %%
model = MNISTTransformer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()


# %%
def train():
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


# %%
def evaluate():
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")


# %%
import time

total_start = time.time()  # ⏱ Start total training timer

for epoch in range(EPOCHS):
    start = time.time()  # ⏱ Start epoch timer

    train()  # Train for one epoch

    duration = time.time() - start  # Time taken for this epoch
    print(f"\nEpoch {epoch+1} completed in {duration:.2f} seconds:")

    evaluate()  # Evaluate after each epoch if needed

total_duration = time.time() - total_start  # Total time for all epochs
print(f"\n✅ Training completed in {total_duration:.2f} seconds.")



# %%


# %%


# %%
# defining function to count parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total:,}\n")
    
    print(f"{'Layer':<60} {'Param #':>12}")
    print("="*75)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<60} {param.numel():>12}")


# %%
model = MNISTTransformer().to(DEVICE)
count_parameters(model)


# %%
torch.save(model.state_dict(), "full_precision.pth")
state_dict = torch.load("full_precision.pth")


# %%
for name in state_dict:
    print(name)


# %%
for name, param in state_dict.items():
    print(f"Layer: {name}")
    print(f"Shape: {list(param.shape)}")
    print(f"Values:\n{param}\n")
    print("=" * 60)


# %%
with open("full_precision_parameters.txt", "w") as f:
    for name, param in state_dict.items():
        f.write(f"Layer: {name}\n")
        f.write(f"Shape: {list(param.shape)}\n")
        f.write(f"Values:\n{param.cpu().numpy()}\n")
        f.write("=" * 60 + "\n")


# %%
print(model)


# %%
for name, param in model.named_parameters():
    print(f"{name:40s} | dtype: {param.dtype}")


# %%



