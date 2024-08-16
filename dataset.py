import torch
from torch.utils.data import Dataset

class SudokuDataset(Dataset):
    def __init__(self, data):
        self.puzzles = data['puzzle']
        self.solutions = data['solution']
    
    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        puzzle = torch.tensor([int(digit) for digit in self.puzzles[idx]], dtype=torch.long)
        solution = torch.tensor([int(digit) for digit in self.solutions[idx]], dtype=torch.long)
        
        return puzzle, solution
