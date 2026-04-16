# Copyright (c) 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import InteractionDIGIRDataset
from transforms import TargetBuilder


class InteractionDIGIRDataModule(pl.LightningDataModule):

    def __init__(
            self,
            interaction_data_path: str,
            train_batch_size: int,
            val_batch_size: int,
            test_batch_size: int,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = True,
            persistent_workers: bool = True,
            num_historical_steps: int = 8,
            num_future_steps: int = 12,
            train_max_samples: Optional[int] = None,
            val_max_samples: Optional[int] = None,
            test_max_samples: Optional[int] = None,
            use_kg: bool = True,
            train_transform: Optional[Callable] = None,
            val_transform: Optional[Callable] = None,
            test_transform: Optional[Callable] = None,
            **kwargs) -> None:
        super(InteractionDIGIRDataModule, self).__init__()
        self.interaction_data_path = interaction_data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.train_max_samples = train_max_samples
        self.val_max_samples = val_max_samples
        self.test_max_samples = test_max_samples
        self.use_kg = use_kg
        self.train_transform = (
            train_transform if train_transform is not None else TargetBuilder(num_historical_steps, num_future_steps))
        self.val_transform = (
            val_transform if val_transform is not None else TargetBuilder(num_historical_steps, num_future_steps))
        self.test_transform = test_transform

    def prepare_data(self) -> None:
        InteractionDIGIRDataset(
            data_path=self.interaction_data_path,
            split='train',
            transform=self.train_transform,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            max_samples=1,
            use_kg=self.use_kg)
        InteractionDIGIRDataset(
            data_path=self.interaction_data_path,
            split='val',
            transform=self.val_transform,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            max_samples=1,
            use_kg=self.use_kg)
        InteractionDIGIRDataset(
            data_path=self.interaction_data_path,
            split='test',
            transform=self.test_transform,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            max_samples=1,
            use_kg=self.use_kg)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = InteractionDIGIRDataset(
            data_path=self.interaction_data_path,
            split='train',
            transform=self.train_transform,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            max_samples=self.train_max_samples,
            use_kg=self.use_kg)
        self.val_dataset = InteractionDIGIRDataset(
            data_path=self.interaction_data_path,
            split='val',
            transform=self.val_transform,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            max_samples=self.val_max_samples,
            use_kg=self.use_kg)
        self.test_dataset = InteractionDIGIRDataset(
            data_path=self.interaction_data_path,
            split='test',
            transform=self.test_transform,
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            max_samples=self.test_max_samples,
            use_kg=self.use_kg)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers)
