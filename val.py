# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser
from argparse import ArgumentTypeError

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from datasets import InteractionDIGIRDataset
from predictors import QCNet
from transforms import TargetBuilder


def _str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if val in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise ArgumentTypeError(f'Invalid boolean value: {v}')


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, default='.')
    parser.add_argument('--interaction_data_path', type=str, default=None)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--use_kg', type=_str2bool, default=True)
    parser.add_argument('--allow_test_as_val', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=_str2bool, default=True)
    parser.add_argument('--persistent_workers', type=_str2bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
    dataset_cls = {
        'argoverse_v2': ArgoverseV2Dataset,
        'interaction_digir': InteractionDIGIRDataset,
    }[model.dataset]
    if model.dataset == 'interaction_digir':
        if args.interaction_data_path is None:
            raise ValueError('--interaction_data_path is required for interaction_digir validation')
        val_dataset = dataset_cls(
            data_path=args.interaction_data_path,
            split=args.split,
            transform=TargetBuilder(model.num_historical_steps, model.num_future_steps),
            num_historical_steps=model.num_historical_steps,
            num_future_steps=model.num_future_steps,
            max_samples=args.max_samples,
            use_kg=args.use_kg,
            allow_test_as_val=args.allow_test_as_val)
    else:
        val_dataset = dataset_cls(
            root=args.root,
            split='val',
            transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory,
                            persistent_workers=args.persistent_workers and args.num_workers > 0)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
    trainer.validate(model, dataloader)
