import numpy as np


print("NOTE: This script is meant to be run from the base Yolo directory!!")


# ------------------------------ Setup the files ----------------------------- #
complete_data_file_path  = './scripts/complete_data.txt'
train_file_path          = './scripts/train.txt'
val_file_path            = './scripts/val.txt'


# --------------------------------- Constants -------------------------------- #
TRAIN_SPLIT = 0.8
VAL_SPLIT   = 1 - TRAIN_SPLIT

# --------------- Count number of sample in complete_data_file --------------- #
num_data = 0
with open(complete_data_file_path, "r") as complete_data_file:
    for line in complete_data_file:
        num_data += 1

print(num_data)
# ---------------------------------------------------------------------------- #
#                  Create Train Validation Split Helper Array                  #
# ---------------------------------------------------------------------------- #

# --------------------- Create shuffled array of indices --------------------- #
indices = np.arange(num_data)
np.random.shuffle(indices)

# --- Create split array to determine which data point goes to which split --- #
split = np.zeros((num_data))
for i in range(num_data):
    if i < (num_data*TRAIN_SPLIT):
        split[indices[i]] = 0
    else:
        split[indices[i]] = 1

# ---------------------------------------------------------------------------- #
#                         Create Train/Val Split Output                        #
# ---------------------------------------------------------------------------- #
with open(complete_data_file_path, "r") as complete_data_file:
    with open(train_file_path, "w") as train_file:
        with open(val_file_path, "w") as val_file:
            line_num = 0
            for line in complete_data_file:
                
                # If line in train split, assign to train file
                if split[line_num] == 0:
                    train_file.write(line)
                # If line in val split, assign to val file
                elif split[line_num] == 1:
                    val_file.write(line)
                
                line_num += 1
