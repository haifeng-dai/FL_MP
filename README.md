# FL_MP

基于 PyTorch 原生多进程的单机多 GPU 联邦学习框架。

正式入口：

```bash
uv run main.py --algo fedavg --dataset cifar10 --model cnn --devices 0:2
```

每个算法可在共享参数之外注册私有参数。实验产物保存在 `results/`，数据集保存在 `datasets/`。

框架可用同一 seed 运行多个独立试次；每个试次都会创建独立的 Server、Worker 池和结果目录，运行 ID 与 `args.json` 中会记录试次编号：

```bash
# 运行第 1 至第 5 次
uv run main.py --algo fedavg --seed 42 --times 5

# 仅运行或补跑指定试次
uv run main.py --algo fedavg --seed 42 --trials 2,4
```

多组串行实验可以并发运行：

```bash
uv run scripts/run_groups.py
```

在 `scripts/run_groups.py` 的 `GROUP_COMMANDS` 中直接填写任意数量的组。每组的 `common_args` 放置共同实验条件，`commands` 只放算法名及算法特定参数；各组会同时开始，每组的命令依次执行。单条命令失败后，该组会继续执行下一条；批处理日志和 `summary.json` 保存到 `results/batches/年/月/日/时分秒/`。

查看已完成实验：

```bash
# 列出全部完成运行，或筛选算法
uv run scripts/results.py
uv run scripts/results.py list --algo fedavg,fedprox

# 查看单次运行的参数和终端准确率曲线
uv run scripts/results.py show 103118-fedfm-5b86dc

# 多算法的最新可比运行，或单算法不同参数的快速比较
uv run scripts/results.py compare --algo fedavg,fedprox,fedfm
uv run scripts/results.py compare --algo fedprox --vary mu

# 筛选指定试次，或汇总同一 seed 的多次运行
uv run scripts/results.py list --algo fedavg --trial 3
uv run scripts/results.py aggregate --algo fedavg,fedprox --trials 1,2,3,4,5
```

结果工具只读取 `results/results.sqlite` 和对应运行目录，不生成图像文件。多算法比较会要求数据集、划分、客户端数、轮数、随机种子等基础条件一致；聚合会要求各算法覆盖相同试次，并输出最终和最佳准确率的均值 ± 标准差。
