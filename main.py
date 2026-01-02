import os
import torch
import numpy as np
import function as ftn
from tqdm import tqdm

# ----------- device config ---------- #
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using Device:', DEVICE)
# ------------------------------------ #

# ----------- user settings ---------- #
learning_rate = 0.001
batch_size = 8
num_training = 200000

lr_decay_factor = 0.78
lr_decay_iter = 10000
decay_start = 20000
# ------------------------------------ #

# ----------- restore options -------- #
brestore = False
restore_lr = 0.001
restore_iter = 20000
# ------------------------------------ #

# ----------- model save options ----- #
model_load_path = ''
model_save_path = ''

model_save_interval = 5000
# ------------------------------------ #

# ------------ load data ------------- #
train_label = ''
test_label = ''

data_path = ''
gt_path = ''

train_data, train_gt, test_data, test_gt = ftn.load_data(
                                                data_path,
                                                gt_path,
                                                train_label,
                                                test_label,
                                                size = 256
                                                )

# ------------ normalization --------- #
normalize = ftn.DataPreprocessing(train_data, test_data)
train_data, test_data = normalize.normalize()
# ------------------------------------ #

# ------------ model config ---------- #
model = ftn.Unet().to(DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(),
    momentum = 0.9, # nesterov = True
    lr = learning_rate if not brestore else restore_lr
)
# ------------------------------------ #

# ------------ change shape ---------- #
test_data = np.transpose(test_data, (0, 3, 1, 2)) # [N, C, H, W]
# ------------------------------------ #

# ------------ training -------------- #
# ------------------------------------ #
