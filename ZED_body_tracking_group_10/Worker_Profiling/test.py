import json
import os
import pickle


def check_value(input_value, tuple_list):
    for tpl in tuple_list:
        if tpl[0] == input_value:
            return True
    return False

def append_to_dict(existing_dict, new_items):
    updated_dict = {}
    for key, values in existing_dict.items():
        updated_dict[key] = values
        if key in new_items:
            for value in new_items[key]:
                tup_value = tuple(value)
                if not check_value(tup_value[0], updated_dict[key]):
                    updated_dict[key].append(tup_value)
            del new_items[key]
    for key, values in new_items.items():
        updated_dict[key] = values

    return updated_dict

# # Example usage:
# existing_dict = {'key1': [['value1',3]], 'key2': [['value2',0]]}
# new_items = {'key1': [('value1', 2)], 'key3': [('value3',0)]}
#
# updated_dict = append_to_dict(new_items, existing_dict)
# print(updated_dict)

# dict1 = {"values": [1,2,3]}
# dict2 = {("value1", "value2"): 1}
# worker_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "Profiles")
# file_path = os.path.join(worker_dir, "worker_1.json")
# with open(file_path, 'wb') as pickle_file:
#     pickle.dump(dict2, pickle_file)
#
# To load the pickle file back into a dictionary
with open("/home/kamil/PycharmProjects/Project3-1_WORKING_ZED/ZED_body_tracking_group_10/Worker_Profiling/Profiles/worker_3.pkl", 'rb') as pickle_file:
    loaded_dict = pickle.load(pickle_file)
print(loaded_dict)
#
# for key,values in dict2.items():
#     print(key)
#     print(values)

# s = "Cup0_root"
# s = s.split("_")[0]
# print(s[:-1])

# print({"probs": dict1, "values": dict2})