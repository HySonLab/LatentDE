import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Tuple, Union


class ProteinDataset(Dataset):

    def __init__(self,
                 csv_data: Union[str, pd.DataFrame],
                 max_length: int = None):
        """
        Args:
            csv_data (str | pd.DataFrame): Path to the csv file or pd.DataFrame.
        """
        super(ProteinDataset, self).__init__()
        if isinstance(csv_data, str):
            self.data = pd.read_csv(csv_data)
        elif isinstance(csv_data, pd.DataFrame):
            self.data = csv_data
        else:
            raise ValueError("csv_data has to be string or dataframe.")
        self.max_length = max_length or max(self.data["sequence"].apply(lambda x: len(x)).to_list())
        self.min_fitness = self.data["fitness"].min()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequences = self.data.iloc[idx, 0]
        fitnesses = self.data.iloc[idx, 1]
        if isinstance(sequences, pd.Series):
            sequences = sequences.tolist()
            fitnesses = fitnesses.tolist()
        return {"sequences": sequences,
                "fitness": torch.tensor(fitnesses, dtype=torch.float32)}


class ProteinsDataModule(LightningDataModule):

    def __init__(self,
                 csv_data: str,
                 max_length: int = None,
                 train_val_split: Tuple[float, float] = (0.9, 0.1),
                 train_batch_size: int = 32,
                 valid_batch_size: int = 32,
                 num_workers: int = 64,
                 seed: int = 0):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_dataset = None
        self.valid_dataset = None
        self.datasets = ProteinDataset(self.hparams.csv_data,
                                       self.hparams.max_length)

    @property
    def min_fitness(self):
        return self.datasets.min_fitness

    @property
    def max_length(self):
        return self.datasets.max_length

    def set_max_length(self, max_len: int):
        self.datasets.max_length = max_len

    def setup(self, stage):
        self.train_dataset, self.valid_dataset = random_split(
            dataset=self.datasets,
            lengths=self.hparams.train_val_split,
            generator=torch.Generator().manual_seed(self.hparams.seed)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.valid_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
