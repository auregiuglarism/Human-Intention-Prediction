import os
import pickle
from pathlib import Path

worker_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "Profiles")


class WorkerProbs:
    def __init__(self, task_prob =None, task_counter = None):
        self.json_file = "worker_"
        self.worker_dir = "Profiles"
        if task_prob is None:
            self.task_prob = {}
        else:
            self.task_prob = {}
        if task_counter is None:
            self.task_counter = {}
        else:
            self.task_counter = {}

    def clear_probs(self):
        self.task_prob = {}

    def check_value(self, input_value, tuple_list):
        """
            The method checks whether the input_value is already in the tuple list
        """
        for tpl in tuple_list:
            if tpl[0] == input_value:
                return True
        return False

    def append_to_dict(self, existing_dict, new_items):
        """
            function to update the values, the existing values are prioritized, meaning that
            if new_items have the same node changed, it will keep the value from existing_dict
        """
        updated_dict = {}
        # for each key in the existing dict
        for key, value in existing_dict.items():
            # append existing_dict values to the list
            # the key is tuple already
            updated_dict[key] = value
            # now check in the new_items have this key
            if key in new_items:
                # delete all appended values
                del new_items[key]
        # if there are keys that were not appended from new_items
        # append them in to the list
        for key, value in new_items.items():
            updated_dict[key] = value

        return updated_dict

    def save_pickle(self, worker_id=None):
        # if worker id was given
        if worker_id is not None:
            file_path = os.path.join(worker_dir, self.json_file + str(worker_id) + ".pkl")
            worker_file = Path(file_path)
            # check if there already exists a worker file
            if worker_file.is_file():
                # open the json file and update the values
                with open(file_path, 'rb') as pickle_file:
                    pickle_data = pickle.load(pickle_file)
                self.task_prob = self.append_to_dict(self.task_prob, pickle_data)
        # no worker existing
        else:
            worker_id = str(len(os.listdir(worker_dir)) + 1)
            file_path = os.path.join(worker_dir, self.json_file + worker_id + ".pkl")

        with open(file_path, 'wb') as pickle_file:
            # combining tasks probs and task counter in to one dict
            worker_data = {"probs": self.task_prob, "counter": self.task_counter}
            pickle.dump(worker_data, pickle_file)

    def add_task_prob(self, task_from, task_to, prob):
        key = (task_from, task_to)
        self.task_prob[key] = prob

    def add_task_counter(self, task_from, task_to, counter):
        key = (task_from, task_to)
        self.task_counter[key] = counter

if __name__ == "__main__":
    worker = WorkerProbs()
    worker.add_task_prob("Cup0", "Crate0", 0.7)
    worker.add_task_prob("Cup0", "Feeder0", 0.3)

    worker.save_pickle()
