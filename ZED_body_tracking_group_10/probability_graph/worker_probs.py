import json
import os
from pathlib import Path

worker_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "worker_probs")


class WorkerProbs:
    def __init__(self):
        self.json_file = "worker_"
        self.worker_dir = "worker_probs"
        self.task_prob = {}

    def clear_probs(self):
        self.task_prob = {}

    # The method checks whether the input_value is already in the tuple list
    def check_value(self, input_value, tuple_list):
        for tpl in tuple_list:
            if tpl[0] == input_value:
                return True
        return False

    # function to update the values, the existing values are prioritized, meaning that
    # if new_items have the same node changed, it will keep the value from existing_dict
    def append_to_dict(self, existing_dict, new_items):
        updated_dict = {}
        # for each key in the existing dict
        for key, values in existing_dict.items():
            # append existing_dict values to the list
            # they are already tuples
            updated_dict[key] = values
            # now check in the new_items have this key
            if key in new_items:
                for value in new_items[key]:
                    # convert the list in to tuple
                    tup_value = tuple(value)
                    # if there is no such value add it to the updated dict with @key
                    if not self.check_value(tup_value[0], updated_dict[key]):
                        updated_dict[key].append(tup_value)
                # delete all appended values
                del new_items[key]
        # if there are keys that were not appended from new_items
        # append them in to the list
        for key, values in new_items.items():
            updated_list = []
            for value in values:
                updated_list.append(tuple(value))
            updated_dict[key] = updated_list

        return updated_dict

    def save_json(self, worker_id=None):
        # if worker id was given
        if worker_id is not None:
            file_path = os.path.join(worker_dir, self.json_file + str(worker_id) + ".json")
            worker_file = Path(file_path)
            # check if there already exists a worker file
            if worker_file.is_file():
                # open the json file and update the values
                json_data = json.load(worker_file.open())
                self.task_prob = self.append_to_dict(self.task_prob, json_data)
                with open(file_path, 'w') as json_file:
                    json.dump(self.task_prob, json_file)
            else:
                with open(file_path, 'w') as json_file:
                    json.dump(self.task_prob, json_file)
        # no worker existing
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
    worker.add_task_prob("Cup0", "Crate0", 0.7)
    worker.add_task_prob("Cup0", "Feeder0", 0.3)

    worker.save_json()
