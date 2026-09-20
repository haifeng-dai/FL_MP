"""FedAvg 算法实现。"""

from __future__ import annotations

from algo.core import BaseClient, BaseServer


class FedAvgClient(BaseClient):
    """FedAvg 的本地 SGD 客户端。"""

    def train(self, task):
        """以 SGD 完成标准监督本地训练。"""
        loss = self.train_supervised(task.client_id)
        return self.result(task, loss)


class Server(BaseServer):
    """FedAvg 服务端：仅定义标准加权聚合。"""

    client_class = FedAvgClient

    def apply_results(self, results):
        self.aggregate_weighted(results)
