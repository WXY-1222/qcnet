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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datamodules import ArgoverseV2DataModule
from datamodules import InteractionDIGIRDataModule
from predictors import QCNet

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='.')
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--interaction_data_path', type=str, default=None)
    parser.add_argument('--train_max_samples', type=int, default=None)
    parser.add_argument('--val_max_samples', type=int, default=None)
    parser.add_argument('--test_max_samples', type=int, default=None)
    parser.add_argument('--use_kg', type=bool, default=True)
    parser.add_argument('--batch_by_location', action='store_true')
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--eval_batches', type=int, default=0,
                        help='0 for full validation; >0 limits number of val batches per epoch')
    QCNet.add_model_specific_args(parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)

    if args.dataset == 'interaction_digir' and args.interaction_data_path is None:
        raise ValueError('--interaction_data_path is required when --dataset interaction_digir')

    model = QCNet(**vars(args))
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
        'interaction_digir': InteractionDIGIRDataModule,
    }[args.dataset](**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer_kwargs = dict(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=args.max_epochs,
    )
    if args.save_root is not None:
        trainer_kwargs['default_root_dir'] = args.save_root
    if args.eval_batches > 0:
        trainer_kwargs['limit_val_batches'] = args.eval_batches
    else:
        trainer_kwargs['limit_val_batches'] = 1.0
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule)
