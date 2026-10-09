"""Small demo network for Vistorch.

Run under the tracer:
    PYTHONPATH=extension/python VISTORCH_PORT=8977 python -m vistorch examples/demo_cnn.py
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.act(self.bn(self.conv(x)))
        if identity.shape == out.shape:
            out = out + identity  # residual edge shows up in the graph
        return out


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = Block(3, 16)
        self.body = nn.Sequential(
            Block(16, 16),
            Block(16, 32, stride=2),
            Block(32, 32),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x)
        x = self.dropout(x.flatten(1))
        return self.fc(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    net = Net()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    for step in range(3):  # a few training steps so shapes repeat
        x = torch.randn(4, 3, 32, 32)
        y = net(x)
        loss = y.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[demo] step={step} out={tuple(y.shape)} loss={float(loss):.4f}")
    print("[demo] done")
