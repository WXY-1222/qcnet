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
from datetime import datetime
import json
import os
import subprocess

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from datamodules import InteractionDIGIRDataModule
from predictors import QCNet


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
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='.')
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=_str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=_str2bool, default=True)
    parser.add_argument('--persistent_workers', type=_str2bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--interaction_data_path', type=str, default=None)
    parser.add_argument('--locations', type=str, default=None,
                        help='Comma-separated location names for filtering interaction samples')
    parser.add_argument('--train_max_samples', type=int, default=None)
    parser.add_argument('--val_max_samples', type=int, default=None)
    parser.add_argument('--test_max_samples', type=int, default=None)
    parser.add_argument('--use_kg', type=_str2bool, default=True)
    parser.add_argument('--batch_by_location', action='store_true')
    parser.add_argument('--location_batch_seed', type=int, default=None)
    parser.add_argument('--ddp_even_strategy', type=str, default='drop', choices=['drop', 'pad'])
    parser.add_argument('--allow_test_as_val', action='store_true')
    parser.add_argument('--require_test_split', action='store_true')
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--eval_batches', type=int, default=0,
                        help='0 for full validation; >0 limits number of val batches per epoch')
    parser.add_argument('--monitor_metric', type=str, default='val_minADE')
    parser.add_argument('--monitor_mode', type=str, default='min', choices=['min', 'max'])
    QCNet.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.location_batch_seed is None:
        args.location_batch_seed = args.seed
    pl.seed_everything(args.seed, workers=True)

    if args.dataset == 'interaction_digir' and args.interaction_data_path is None:
        raise ValueError('--interaction_data_path is required when --dataset interaction_digir')

    model = QCNet(**vars(args))
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
        'interaction_digir': InteractionDIGIRDataModule,
    }[args.dataset](**vars(args))

    if args.save_root is not None:
        os.makedirs(args.save_root, exist_ok=True)
        git_commit = ''
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True).strip()
        except Exception:
            git_commit = ''
        run_meta = {
            'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'git_commit': git_commit,
            'argv': vars(args),
        }
        meta_file = os.path.join(args.save_root, f'run_meta_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.json')
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(run_meta, f, ensure_ascii=False, indent=2)
        print(f'[RunMeta] saved to {meta_file}')

    model_checkpoint = ModelCheckpoint(monitor=args.monitor_metric, save_top_k=5, mode=args.monitor_mode)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer_kwargs = dict(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )
    if args.save_root is not None:
        trainer_kwargs['default_root_dir'] = args.save_root
    if args.eval_batches > 0:
        trainer_kwargs['limit_val_batches'] = args.eval_batches
    else:
        trainer_kwargs['limit_val_batches'] = 1.0
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule)
