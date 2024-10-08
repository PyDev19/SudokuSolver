import torch
from typing import Tuple
from torch import Tensor
from torch.nn import Conv2d, Sequential, ReLU, Softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import cross_entropy
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule

class SudokuSolverCNN(LightningModule):
    def __init__(self, hyperparameters: dict[str, int]):
        super(SudokuSolverCNN, self).__init__()
        
        self.num_classes = hyperparameters['num_classes']
        cnn_channels = hyperparameters['cnn_channels']
        self.lr = hyperparameters['lr']
        
        self.cnn = Sequential()
        for i, channel in enumerate(cnn_channels):
            if i == 0:
                self.cnn.add_module(f'conv_{i}', Conv2d(in_channels=1, out_channels=channel, kernel_size=3, padding=1))
            else:
                self.cnn.add_module(f'conv_{i}', Conv2d(in_channels=cnn_channels[i-1], out_channels=channel, kernel_size=3, padding=1))
            
            self.cnn.add_module(f'relu_{i}', ReLU())
        
        self.cnn.add_module('conv_final', Conv2d(cnn_channels[-1], self.num_classes, kernel_size=1))
        self.output = Softmax(dim=1)
        
        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, 1, 9, 9)

    def forward(self, input_tensor: Tensor) -> Tensor:
        cnn_features = self.cnn(input_tensor)
        output = self.output(cnn_features)
        
        return output
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        sudoku_puzzle, sudoku_solution = batch
        predicted_solution = self(sudoku_puzzle)
        
        loss = cross_entropy(predicted_solution, sudoku_solution)
        accuracy_score = accuracy(predicted_solution, sudoku_solution, task='multiclass', num_classes=self.num_classes)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', accuracy_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        sudoku_puzzle, sudoku_solution = batch
        predicted_solution = self(sudoku_puzzle)
        
        loss = cross_entropy(predicted_solution, sudoku_solution)
        accuracy_score = accuracy(predicted_solution, sudoku_solution, task='multiclass', num_classes=self.num_classes)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', accuracy_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        sudoku_puzzle, sudoku_solution = batch
        predicted_solution = self(sudoku_puzzle)
        
        loss = cross_entropy(predicted_solution, sudoku_solution)
        accuracy_score = accuracy(predicted_solution, sudoku_solution, task='multiclass', num_classes=self.num_classes)
        
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', accuracy_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self) -> Tuple[Adam, ReduceLROnPlateau]:
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }
