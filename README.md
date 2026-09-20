# FL_MP

基于 PyTorch 原生多进程的单机多 GPU 联邦学习框架。

正式入口：

```bash
uv run main.py --algo fedavg --dataset cifar10 --model cnn --devices 0:2
```

每个算法可在共享参数之外注册私有参数。实验产物保存在 `results/`，数据集保存在 `datasets/`。
