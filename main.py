import pandas as pd
from dataset import SudokuDataset

data = pd.read_csv('data/sudoku.csv')
dataset = SudokuDataset(data)

from torch.utils.data import random_split

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

from torch.utils.data import DataLoader
import multiprocessing

num_workers = multiprocessing.cpu_count()-1
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)

from model import SudokuSolver

hyperparameters = {
    'vocab_size': 10,
    'embed_size': 128,
    'hidden_size': 256,
    'num_layers': 2,
    'lr': 1e-3
}

model = SudokuSolver(hyperparameters)

from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

trainer = Trainer(
    max_epochs=10,
    logger=TensorBoardLogger('logs', name='sudoku-solver', log_graph=True),
    callbacks=[
        ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min'),
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        LearningRateMonitor(logging_interval='step')
    ],
    enable_checkpointing=True,
    enable_model_summary=True,
    enable_progress_bar=True,
)
tuner = Tuner(trainer)

tuner.lr_find(model, train_loader, val_loader)