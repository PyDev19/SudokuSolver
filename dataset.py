import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
from typing import Optional, Union
from pandas import DataFrame

class SudokuDataset(Dataset):
    def __init__(self, data: DataFrame):
        self.puzzles = data['quizzes']
        self.solutions = data['solutions']
    
    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        puzzle = [int(digit) for digit in self.puzzles[idx]]
        solution = [int(digit) for digit in self.solutions[idx]]
        
        puzzle_grid = torch.tensor(puzzle, dtype=torch.float32).view(1, 9, 9)
        solution_grid = torch.tensor(solution, dtype=torch.long).view(9, 9) - 1
        
        return puzzle_grid, solution_grid


class SudokuDataModule(LightningDataModule):
    def __init__(self, data: DataFrame, hyperparameters: dict[str, Union[int, float]]):
        super(SudokuDataModule, self).__init__()
        self.data = data
        
        self.batch_size = hyperparameters['batch_size']
        self.val_split = hyperparameters['val_split']
        self.test_split = hyperparameters['test_split']
        self.num_workers = hyperparameters['num_workers']
        
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        self.dataset = SudokuDataset(self.data)

    def setup(self, stage: Optional[str] = None):
        if self.dataset is None:
            raise RuntimeError("Dataset not prepared. Please run `prepare_data` before `setup`.")

        total_size = len(self.dataset)
        test_size = int(self.test_split * total_size)
        val_size = int(self.val_split * total_size)
        train_size = total_size - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        
        print(f"Train size: {len(self.train_dataset)}")
        print(f"Val size: {len(self.val_dataset)}")
        print(f"Test size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
