import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


print("PyTorch版本：", torch.__version__)

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)



print("MNIST加载成功！")
print("训练集数量：", len(train_dataset))
print("测试集数量：", len(test_dataset))



image, label = train_dataset[0]

print("图片尺寸：", image.shape)
print("图片标签：", label)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"数字：{label}")
plt.show()


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
print("DataLoader创建成功！")

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)



images, labels = next(iter(train_loader))

print("这一批图片的尺寸：", images.shape)
print("这一批标签的尺寸：", labels.shape)

model = MNISTNet()

output = model(images)

print("模型输出尺寸：", output.shape)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("损失函数和优化器创建成功！")

loss_list = []

epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for images, labels in train_loader:

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 清空梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    loss_list.append(avg_loss)

    print(
        f"第{epoch+1}轮训练，损失：{avg_loss:.4f}"
    )

    # 测试模型

model.eval()  # 切换到测试模式

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total

print("测试准确率：", accuracy, "%")

plt.plot(loss_list)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training Loss Curve")

plt.show()


torch.save(model.state_dict(), "mnist_model.pth")

print("模型保存成功！")


import random

index = random.randint(0, len(test_dataset)-1)

image, label = test_dataset[index]


model.eval()

with torch.no_grad():

    output = model(image.unsqueeze(0))

    prediction = torch.argmax(output, dim=1)


plt.imshow(image.squeeze(), cmap="gray")


plt.title(
    f"Prediction:{prediction.item()}  Label:{label}"
)
plt.show()