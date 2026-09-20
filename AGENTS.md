# FL_MP 开发规约

`FL_MP` 是独立的原生多进程联邦学习项目，不适用父项目的 Ray 调度要求。

- 客户端训练和评估使用 `torch.multiprocessing`；不得引入 Ray。
- 所有项目程序、测试和脚本通过 `uv run` 执行。
- 训练仅支持 NVIDIA CUDA GPU；不提供 CPU 训练 fallback。
- Worker 间传输的模型状态必须是独立 CPU Tensor。
- 代码、配置、依赖和设计修改均须先向负责人说明并获得授权。
- 项目输出、文档及沟通使用中文。
