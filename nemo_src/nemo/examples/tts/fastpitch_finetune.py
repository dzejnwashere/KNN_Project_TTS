# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import lightning.pytorch as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

# We changed the config name from fastpitch_align_44100 to fastpitch_emotion_22050
@hydra_runner(config_path="conf", config_name="fastpitch_emotion_22050")
def main(cfg):
    if hasattr(cfg.model.optim, 'sched'):
        logging.warning("You are using an optimizer scheduler while finetuning. Are you sure this is intended?")
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logging.warning("The recommended learning rate for finetuning is 2e-4")
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    # we added this lines to match our configuration
    pretrained = FastPitchModel.restore_from(
        "/home/dzejn/PycharmProjects/KNN_Project_TTS/tts_en_fastpitch.nemo",
        map_location='cpu',
        strict=False
    )
    model = FastPitchModel(cfg=cfg.model, trainer=trainer)
    pretrained_dict = pretrained.state_dict()
    model_dict = model.state_dict()

    matched = {k: v for k, v in pretrained_dict.items()
               if k in model_dict and v.shape == model_dict[k].shape}
    unmatched = [k for k in model_dict if k not in matched]

    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)

    print(f"\nLoaded {len(matched)} layers from pretrained checkpoint")
    print(f"New randomly initialized layers ({len(unmatched)}):")
    for k in unmatched:
        print(f"  {k}")

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


if __name__ == '__main__':
    main()
