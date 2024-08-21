import pandas as pd
from dataset import SudokuDataModule
from model import SudokuSolverCNN
from lightning.pytorch.utilities.model_summary import ModelSummary
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

hyperparameters = {
    'num_classes': 10,
    'cnn_channels': [32, 64, 128, 256, 512],
    'lr': 1e-3,
    'batch_size': 64,
    'val_split': 0.2,
    'test_split': 0.2,
    'num_workers': 4,
    'max_epochs': 10,
}

data = pd.read_csv('data/sudoku_small.csv')
dataset = SudokuDataModule(data, hyperparameters)

model = SudokuSolverCNN(hyperparameters)
print(ModelSummary(model, max_depth=2))

trainer = Trainer(
    max_epochs=10,
    logger=TensorBoardLogger('logs', name='sudoku-solver', log_graph=True),
    callbacks=[
        ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min'),
        EarlyStopping(monitor='val_loss', patience=5, mode='min'),
        LearningRateMonitor(logging_interval='step')
    ],
    enable_checkpointing=True,
    enable_progress_bar=True,
)
tuner = Tuner(trainer)

tuner.lr_find(model, datamodule=dataset, max_lr=10, min_lr=1e-10, num_training=9375)
trainer.fit(model, datamodule=dataset)
trainer.test(model, datamodule=dataset)