"""LG-FedAvg 算法实现。"""

from algo.core import BaseClient, BaseServer
from algo.fedper import aggregate_states


class LGFedAvgClient(BaseClient):
    """训练私有特征提取器和全局分类头。"""

    def train(self, task):
        loss = self.train_supervised(task.client_id)
        return self.result(task, loss)


class Server(BaseServer):
    """保留客户端 extractor，仅聚合 classifier。"""

    client_class = LGFedAvgClient
    pfl = True

    def apply_results(self, results):
        self.update_client_states(results)
        total = sum(results[client_id].num_samples for client_id in self.selected)
        weights = [
            results[client_id].num_samples / total for client_id in self.selected
        ]
        heads = [
            {
                name.removeprefix("classifier."): value
                for name, value in results[client_id].state.items()
                if name.startswith("classifier.")
            }
            for client_id in self.selected
        ]
        head = aggregate_states(heads, weights)
        self.model.classifier.load_state_dict(head)
        for state in self.client_states.values():
            for name, value in head.items():
                state[f"classifier.{name}"] = value.clone()
