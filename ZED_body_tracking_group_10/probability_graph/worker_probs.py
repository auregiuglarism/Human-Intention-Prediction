import json
import os

worker_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "worker_probs")


class WorkerProbs:
    def __init__(self):
        self.json_file = "worker_"
        self.worker_dir = "worker_probs"
        self.task_prob = {}

    def clear_probs(self):
        self.task_prob = {}

    def save_json(self, worker_id=None):
        if worker_id is not None:
            file_path = os.path.join(worker_dir, self.json_file + worker_id + ".json")
            with open(file_path, 'w') as json_file:
                json.dump(self.task_prob, json_file)
        else:
            new_worker_id = str(len(os.listdir(worker_dir)) + 1)
            file_path = os.path.join(worker_dir, self.json_file + new_worker_id + ".json")
            with open(file_path, 'w') as json_file:
                json.dump(self.task_prob, json_file)

    def add_task_prob(self, task_from, task_to, prob):
        if task_from in self.task_prob:
            self.task_prob[task_from].append((task_to, prob))
        else:
            self.task_prob[task_from] = []
            self.task_prob[task_from].append((task_to, prob))


if __name__ == "__main__":
    worker = WorkerProbs()
    worker.add_task_prob("Cup0", "Crate0", 0.3)
    worker.add_task_prob("Cup0", "Feeder0", 0.7)

    worker.save_json()
