#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import torch

__all__ = ['ReplayMemory']


class ReplayMemory(object):
    def __init__(self, max_size, max_critic_size):
        """ create a replay memory for off-policy RL or offline RL.

        Args:
            max_size (int): max size of replay memory
            obs_dim (list or tuple): observation shape
            act_dim (list or tuple): action shape
        """
        self.max_size = int(max_size)
        self.max_critic_size = int(max_critic_size)
        self.experience = []
        self.global_feature = {"feature": None, "target": None}
        self.all_global_feature = {"feature": [], "target": []}
        self.xt_ps_feature = {"feature": [], "target": []}
        self._curr_size = 0
        self._curr_pos = 0
        self._curr_critic_pos = 0
        self._curr_ps_pos = 0

    def reset(self):
        self.experience = []
        self._curr_size = 0
        self._curr_pos = 0

    def sample_batch(self, batch_size):
        """ sample a batch from replay memory

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        batch_size = min(batch_size, self._curr_size)
        batch_idx = np.random.choice(range(self._curr_size), size=batch_size, replace=False)
        return [self.experience[i] for i in batch_idx]

    def sample_ps_feature(self, batch_size):
        batch_size = min(batch_size, len(self.xt_ps_feature["feature"]))
        batch_idx = np.random.choice(len(self.xt_ps_feature["feature"]), size=batch_size, replace=False)
        feature = [self.xt_ps_feature["feature"][i] for i in batch_idx]
        target = [torch.tensor(self.xt_ps_feature["target"][i]) for i in batch_idx]
        x = torch.stack(feature, dim=0)
        y = torch.squeeze(torch.stack(target, dim=0))
        return x, y

    def sample_global_feature(self, batch_size):
        batch_size = min(batch_size, len(self.all_global_feature["feature"]))
        batch_idx = np.random.choice(range(len(self.all_global_feature["feature"])), size=batch_size, replace=False)
        x, y = [self.all_global_feature["feature"][i] for i in batch_idx], [self.all_global_feature["target"][i] for i in batch_idx]
        mask = []
        all_feature = []
        for idx, feature in enumerate(x):
            if feature is None:
                y.pop(idx)
                continue
            feature = torch.squeeze(feature)
            if feature.dim() < 3:
                feature = feature[None, :, :]
            pad = 40 - feature.shape[0]
            m = torch.zeros(40)
            m[:feature.shape[0]] = torch.ones(feature.shape[0])
            mask.append(m)
            zeros = torch.zeros(pad, feature.shape[1], 27)
            feature = torch.cat((feature, zeros), dim=0)
            all_feature.append(feature)
        x = torch.stack(all_feature, dim=0)
        mask = torch.stack(mask, dim=0)
        y = torch.squeeze(torch.stack(y, dim=0))
        return x, y, mask

    def append(self, experience, global_feature):
        if experience is not None:
            if self._curr_size < self.max_size:
                self._curr_size += 1
                self.experience.append(experience)
            else:
                self._curr_pos = (self._curr_pos + 1) % self.max_size
                self.experience[self._curr_pos] = experience
        if global_feature is not None:
            if self.global_feature["feature"] is not None:
                self.global_feature["feature"] = torch.cat((self.global_feature["feature"], global_feature), dim=1)
            else:
                self.global_feature["feature"] = global_feature

    def append_xt_ps(self, xt_ps_feature):
        if len(self.xt_ps_feature) < self.max_size:
            self._curr_ps_pos += 1
            self.xt_ps_feature["feature"].append(xt_ps_feature["feature"])
            self.xt_ps_feature["target"].append(xt_ps_feature["target"])
        else:
            self._curr_ps_pos = (self._curr_ps_pos + 1) % self.max_size
            self.xt_ps_feature["feature"][self._curr_ps_pos] = xt_ps_feature["feature"]
            self.xt_ps_feature["target"][self._curr_ps_pos] = xt_ps_feature["target"]

    def append_epoch_feature(self, target):
        if len(self.all_global_feature["feature"]) < self.max_critic_size:
            self._curr_critic_pos = self._curr_critic_pos + 1
            self.all_global_feature["feature"].append(self.global_feature["feature"])
            self.all_global_feature["target"].append(target)
            self.global_feature = {"feature": None, "target": None}
        else:
            self._curr_critic_pos = (self._curr_critic_pos + 1) % self.max_critic_size
            self.all_global_feature["feature"][self._curr_critic_pos] = self.global_feature["feature"]
            self.all_global_feature["target"][self._curr_critic_pos] = target
            self.global_feature = {"feature": None, "target": None}

    def size(self):
        """ get current size of replay memory.
        """
        return self._curr_size

    def __len__(self):
        return self._curr_size
