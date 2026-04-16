# Copyright (c) 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import math
import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import pytorch_lightning as pl
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

from datasets import InteractionDIGIRDataset
from transforms import TargetBuilder


class _LocationBatchSampler(Sampler[List[int]]):

    def __init__(self, locations: List[str], batch_size: int, shuffle: bool) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        loc_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, loc in enumerate(locations):
            loc_to_indices[str(loc)].append(idx)
        self.loc_to_indices = dict(loc_to_indices)

    def __iter__(self):
        loc_items = list(self.loc_to_indices.items())
        if self.shuffle:
            random.shuffle(loc_items)

        all_batches: List[List[int]] = []
        for _, indices in loc_items:
            local_indices = indices.copy()
            if self.shuffle:
                random.shuffle(local_indices)
            for i in range(0, len(local_indices), self.batch_size):
                all_batches.append(local_indices[i:i + self.batch_size])

        if self.shuffle:
            random.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for _, indices in self.loc_to_indices.items():
            total += math.ceil(len(indices) / self.batch_size)
        return total


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
            batch_by_location: bool = False,
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
        self.batch_by_location = batch_by_location
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
        if self.batch_by_location:
            batch_sampler = _LocationBatchSampler(
                locations=self.train_dataset.sample_locations,
                batch_size=self.train_batch_size,
                shuffle=self.shuffle)
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers)
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        if self.batch_by_location:
            batch_sampler = _LocationBatchSampler(
                locations=self.val_dataset.sample_locations,
                batch_size=self.val_batch_size,
                shuffle=False)
            return DataLoader(
                self.val_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers)
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        if self.batch_by_location:
            batch_sampler = _LocationBatchSampler(
                locations=self.test_dataset.sample_locations,
                batch_size=self.test_batch_size,
                shuffle=False)
            return DataLoader(
                self.test_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers)
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers)
