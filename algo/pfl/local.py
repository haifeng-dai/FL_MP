"""纯本地训练算法实现。"""

from ..core import BaseClient, BaseServer


class LocalClient(BaseClient):
    """仅在客户端私有数据上执行标准监督训练。"""

    def train(self, task):
        loss = self.train_supervised(task.client_id)
        return self.result(task, loss)


class Server(BaseServer):
    """维护独立客户端模型而不执行服务器聚合。"""

    client_class = LocalClient
    pfl = True

    def apply_results(self, results):
        """将参与客户端的新模型写回其本地持久状态。"""
        self.update_client_states(results)
