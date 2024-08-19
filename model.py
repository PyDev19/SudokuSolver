import torch
from typing import Tuple
from torch import Tensor
from torch.nn import Linear, Conv2d, Sequential, ReLU
from torch.nn.init import xavier_normal_, zeros_, kaiming_normal_, constant_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import cross_entropy, relu, softmax
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule

class SudokuSolverCNN(LightningModule):
    def __init__(self, hyperparameters: dict[str, int]):
        super(SudokuSolverCNN, self).__init__()
        
        self.num_classes = hyperparameters['num_classes']
        hidden_dim = hyperparameters['hidden_dim']
        cnn_channels = hyperparameters['cnn_channels']
        self.lr = hyperparameters['lr']
        
        self.cnn_layers = Sequential()
        
        for i, out_channels in enumerate(cnn_channels):
            in_channels = 1 if i == 0 else cnn_channels[i - 1]
            self.cnn_layers.add_module(
                f'conv_{i}', Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.cnn_layers.add_module(f'relu_{i}', ReLU())
        
        self.fc_hidden = Linear(cnn_channels[-1] * 9 ** 2, hidden_dim)
        self.fc_output = Linear(hidden_dim, self.num_classes * 9 ** 2)
        
        self.init_weights()
        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, 1, 9, 9)
        
    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, Conv2d):
                kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    zeros_(module.bias)
            elif isinstance(module, Linear):
                xavier_normal_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0.1)

    def forward(self, input_tensor: Tensor) -> Tensor:
        cnn_output = self.cnn_layers(input_tensor)
        
        cnn_output = cnn_output.view(-1, 256 * 9 * 9)
        
        hidden_layer_output = relu(self.fc_hidden(cnn_output))
        output_logits = self.fc_output(hidden_layer_output)
                
        output_probs = output_logits.view(-1, self.num_classes, 9, 9)
        
        return softmax(output_probs, dim=3)
    
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
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
        return optimizer
