import math
import numpy as np
import pandas as pd

from seq_utils import shuffle_with_mask, swap_by_indices, generate_seq_pair


SHAPING_OUTPUT_COL_ORDER = [
    "stim",
    "correct_key",
    "block",
    "img_folder",
    "trans_folder",
    "key0_trans",
    "key1_trans",
    "key2_trans",
    "key3_trans",
    "shop0_food",
    "shop1_food",
    "shop2_food",
    "shop3_food",
    "trans0_shop",
    "trans1_shop",
    "trans2_shop",
    "trans3_shop",
    "set_size",
]

def generate_shaping_block(num_keys, num_food, num_iter_per_stim, trans_shop_mapping):
    num_trans = len(trans_shop_mapping)
    trans_stim_seq, correct_key_seq = generate_seq_pair(
        num_trans, num_iter_per_stim, num_keys
    )

    food_seq = []
    for i in range(math.ceil(len(correct_key_seq) / num_food)):
        food_seq.extend(np.random.permutation(num_food))

    seq_data = {
        "stim": [],
        "correct_key": correct_key_seq,
    }
    for i in range(num_trans):
        seq_data[f"shop{i}_food"] = []
        seq_data[f"key{i}_trans"] = []

    all_trans_indexes = np.arange(num_trans)
    all_food_indexes = np.arange(num_food)
    for i, correct_key in enumerate(correct_key_seq):
        seq_data["stim"].append(food_seq[i])

        correct_trans = trans_stim_seq[i]
        base = swap_by_indices(all_trans_indexes, correct_trans, correct_key)
        key_trans_array = shuffle_with_mask(
            base, np.array([i == correct_key for i in range(num_keys)])
        )
        for j, trans in enumerate(key_trans_array):
            seq_data[f"key{j}_trans"].append(trans)

        correct_shop = trans_shop_mapping[correct_trans]
        base = swap_by_indices(all_food_indexes, correct_shop, food_seq[i])
        shop_food_array = shuffle_with_mask(
            base, np.array([i == correct_shop for i in range(num_food)])
        )
        for j, food in enumerate(shop_food_array):
            seq_data[f"shop{j}_food"].append(food)
  
    for k, v in enumerate(trans_shop_mapping):
        seq_data[f"trans{k}_shop"] = [v] * len(trans_stim_seq)

    return pd.DataFrame(seq_data)


def generate_non_shaping_block(
    num_keys, num_food, num_iter_per_stim, trans_shop_mapping, stim_food_mapping
):
    num_villagers = len(stim_food_mapping)
    num_trans = len(trans_shop_mapping)
    villager_seq, correct_key_seq = generate_seq_pair(
        num_villagers, num_iter_per_stim, num_keys
    )

    trans_seq = []
    for i in range(math.ceil(len(correct_key_seq) / num_trans)):
        trans_seq.extend(np.random.permutation(num_trans))

    seq_data = {
        "stim": [],
        "correct_key": correct_key_seq,
    }
    for i in range(num_keys):
        seq_data[f"shop{i}_food"] = []
        seq_data[f"key{i}_trans"] = []

    all_trans_indexes = np.arange(num_trans)
    all_food_indexes = np.arange(num_food)
    for i, correct_key in enumerate(correct_key_seq):
        seq_data["stim"].append(villager_seq[i])

        correct_trans = trans_seq[i]
        base = swap_by_indices(all_trans_indexes, correct_trans, correct_key)
        key_trans_array = shuffle_with_mask(
            base, np.array([i == correct_key for i in range(num_keys)])
        )
        for j, trans in enumerate(key_trans_array):
            seq_data[f"key{j}_trans"].append(trans)

        correct_shop = trans_shop_mapping[correct_trans]
        correct_food = stim_food_mapping[villager_seq[i]]
        base = swap_by_indices(all_food_indexes, correct_shop, correct_food)
        shop_food_array = shuffle_with_mask(
            base, np.array([i == correct_shop for i in range(num_food)])
        )
        for j, food in enumerate(shop_food_array):
            seq_data[f"shop{j}_food"].append(food)

    for k, v in enumerate(trans_shop_mapping):
        seq_data[f"trans{k}_shop"] = [v] * len(villager_seq)

    return pd.DataFrame(seq_data)
