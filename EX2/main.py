import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
from torch.optim import lr_scheduler  # 导入学习率调度器模块
from torchvision import datasets, transforms, models  # 导入 torchvision 中的数据集、预处理、模型
from torch.utils.data import random_split  # 导入数据集划分工具
import os  # 操作系统相关库
import time  # 计时相关库
import copy  # 用于复制模型权重

# 设置数据目录路径
data_dir = '/content/Repository/Repository/EX2/flower_dataset'

# 定义数据增强与预处理操作（训练与验证统一处理）
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪到224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(30),  # 随机旋转±30度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机调整颜色
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用ImageNet均值方差进行归一化
])

# 加载整个数据集
full_dataset = datasets.ImageFolder(data_dir, data_transforms)

# 将数据按8:2比例划分为训练集与验证集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 创建训练和验证数据加载器
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# 获取类别名称
class_names = full_dataset.classes

# 加载预训练的 ResNet18 模型
model = models.resnet18(pretrained=True)

# 修改模型最后一层以适应花卉分类（5类）
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# 定义损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()

# 定义优化器为带动量的SGD
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# 设置学习率调度器，每7个epoch衰减一次，衰减率为0.1
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 定义训练函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()  # 记录开始时间
    best_model_wts = copy.deepcopy(model.state_dict())  # 初始化最优模型权重
    best_acc = 0.0  # 初始化最佳准确率

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用GPU或CPU
    model = model.to(device)  # 将模型迁移到计算设备

    for epoch in range(num_epochs):  # 遍历每一个 epoch
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')

        for phase in ['train', 'val']:  # 在训练集与验证集上分别执行
            if phase == 'train':
                model.train()  # 设置为训练模式
            else:
                model.eval()  # 设置为评估模式

            running_loss = 0.0  # 当前 phase 的损失累加
            running_corrects = 0  # 当前 phase 的预测正确数量

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # 梯度清零

                with torch.set_grad_enabled(phase == 'train'):  # 只在训练时记录梯度
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()  # 反向传播
                        optimizer.step()  # 更新参数

                running_loss += loss.item() * inputs.size(0)  # 累加损失
                running_corrects += torch.sum(preds == labels.data)  # 累加正确预测数

            if phase == 'train':
                scheduler.step()  # 学习率调度

            # 计算平均损失和准确率
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 若验证准确率最优，则保存模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())  # 保存最优权重
                save_dir = 'Ex2/work_dir'
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))  # 保存模型文件

        print()  # 换行

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)  # 载入最佳模型
    return model  # 返回训练好的模型

# 启动训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)
