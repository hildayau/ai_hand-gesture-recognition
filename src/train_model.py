import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定義神經網路結構
class GestureNet(nn.Module):
    def __init__(self):
        super(GestureNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # 3 代表辨識 3 種手勢
        )
    def forward(self, x): return self.model(x)

# 讀取數據
try:
    data = np.loadtxt('data/test_gesture_data.csv', delimiter=',')
    X = torch.tensor(data[:, 1:], dtype=torch.float32)
    y = torch.tensor(data[:, 0], dtype=torch.long)

    model = GestureNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練迴圈
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0: print(f'Epoch {epoch}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'models/test_gesture_model.pth')
    print("模型訓練完成並已儲存！")
except:
    print("請先執行 collect_data.py 收集數據再訓練。")