import torch
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Embedding, LSTM, Linear
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule

class SudokuSolver(LightningModule):
    def __init__(self, hyperparameters: dict[str, int]):
        super(SudokuSolver, self).__init__()
        
        vocab_size = hyperparameters['vocab_size']
        embed_size = hyperparameters['embed_size']
        hidden_size = hyperparameters['hidden_size']
        num_layers = hyperparameters['num_layers']
        self.lr = hyperparameters['lr']
        
        self.embedding = Embedding(vocab_size, embed_size)
        
        self.encoder = LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.decoder = LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.output_layer = Linear(hidden_size, vocab_size)
        
        self.save_hyperparameters()
        self.example_input_array = torch.randint(0, 10, (1, 81), dtype=torch.long)

    def forward(self, puzzles: Tensor, solutions: Optional[Tensor] = None, teacher_forcing_ratio=0.5):
        embedded_puzzle = self.embedding(puzzles)
        _, (hidden, cell) = self.encoder(embedded_puzzle)
        
        decoder_input = puzzles[:, 0].unsqueeze(1) if solutions is None else solutions[:, 0].unsqueeze(1)
        outputs = []
        
        for t in range(81):
            embedded_input = self.embedding(decoder_input)
            output, (hidden, cell) = self.decoder(embedded_input, (hidden, cell))
            output = self.output_layer(output.squeeze(1))
            outputs.append(output.unsqueeze(1))
            
            # Generate next input
            _, top1 = output.topk(1)
            if solutions is None:
                decoder_input = top1  # If no solutions provided, use model output
            else:
                # Use teacher forcing or the model output based on probability
                decoder_input = top1 if torch.rand(1).item() > teacher_forcing_ratio else solutions[:, t].unsqueeze(1)
    
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        puzzles, solutions = batch
        outputs: Tensor = self(puzzles, solutions)
        
        loss = cross_entropy(outputs.view(-1, 10), solutions.view(-1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        outputs = outputs.permute(0, 2, 1)
        
        acc = accuracy(outputs, solutions, task='multiclass', num_classes=10)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        puzzles, solutions = batch
        outputs: Tensor = self(puzzles)
        
        loss = cross_entropy(outputs.view(-1, 10), solutions.view(-1))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        outputs = outputs.permute(0, 2, 1)
        
        acc = accuracy(outputs, solutions, task='multiclass', num_classes=10)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        puzzles, solutions = batch
        outputs: Tensor = self(puzzles)
        
        loss = cross_entropy(outputs.view(-1, 10), solutions.view(-1))
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        outputs = outputs.permute(0, 2, 1)
                
        acc = accuracy(outputs, solutions, task='multiclass', num_classes=10)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
