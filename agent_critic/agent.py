import copy
import os
from torch.optim import Adam
import torch
import numpy as np
from .replay_memory import ReplayMemory


class Agent(object):
    def __init__(self, id, model, save_path, batch_size, critic_batch_szie, lr, warmup_epoch):
        super(Agent, self).__init__()
        self.alg = model
        self.memory_size = 1e4
        self.critic_memory_size = 250
        self.rpm = ReplayMemory(max_size=self.memory_size, max_critic_size=self.critic_memory_size)
        self.global_train_step = 0
        self.min_memory_size = batch_size * 25
        self.batch_size = batch_size
        self.critic_batch_szie = critic_batch_szie
        self.n = 4
        self.gamma = 0.95
        self.epoch = 0
        self.save_path = save_path
        self.id = id
        self.actor_optimizer = Adam(self.alg.parameters(), lr=lr)
        self.lr = lr
        self.warmup_epoch = warmup_epoch

    def warm_up(self, optimizer, epoch):
        if epoch < self.warmup_epoch:
            iter_frac = min(float(epoch + 1) / max(self.warmup_epoch, 1), 1.0)
            lr = self.lr * iter_frac
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr

    def save(self, only_feature=False):
        if not only_feature:
            path = os.path.join(self.save_path, "agent", "checkpt_{:}.pth".format(self.epoch))
            torch.save({
                'state_dict': self.alg.state_dict(),
                'optim_state_dict': self.actor_optimizer.state_dict(),
                'epoch': self.epoch
            }, path)
        path = os.path.join(self.save_path, "global_feature", "global_feature_uid{:}.pth".format(self.id))
        torch.save({
            'feature': self.rpm.all_global_feature["feature"],
            'target': self.rpm.all_global_feature["target"],
            'experience': self.rpm.experience,
            'curr_size': self.rpm._curr_size,
            'curr_pos': self.rpm._curr_pos,
            'curr_critic_pos': self.rpm._curr_critic_pos,
            'ps_feature': self.rpm.xt_ps_feature,
            'curr_ps_pos': self.rpm._curr_ps_pos
        }, path)

    def resume(self, path_model, path_feature=None, load_optimizer=True):
        checkpt = torch.load(path_model, map_location=lambda storage, loc: storage)
        self.alg.load_state_dict(checkpt["state_dict"], strict=True)
        self.epoch = (checkpt["epoch"])
        if load_optimizer:
            self.actor_optimizer.load_state_dict(checkpt["optim_state_dict"])
        if path_feature is not None:
            checkpt = torch.load(path_feature, map_location=lambda storage, loc: storage)
            self.rpm.all_global_feature["feature"] = (checkpt["feature"])
            self.rpm.all_global_feature["target"] = (checkpt["target"])
            self.rpm._curr_size = checkpt["curr_size"]
            self.rpm._curr_pos = checkpt["curr_pos"]
            self.rpm.experience = checkpt["experience"]
            self.rpm._curr_critic_pos = checkpt["curr_critic_pos"]
            self.rpm.xt_ps_feature = checkpt["ps_feature"]
            self.rpm._curr_ps_pos = checkpt["curr_ps_pos"]

    def pretrain(self, path):
        checkpt = torch.load(os.path.join(path, "play", "checkpt.pth"))
        self.alg.actor_model.PlayNet.load_state_dict(checkpt["state_dict"])
        checkpt = torch.load(os.path.join(path, "gang", "checkpt.pth"))
        self.alg.actor_model.GangNet.load_state_dict(checkpt["state_dict"])
        checkpt = torch.load(os.path.join(path, "peng", "checkpt.pth"))
        self.alg.actor_model.PengNet.load_state_dict(checkpt["state_dict"])
        checkpt = torch.load(os.path.join(path, "chi", "checkpt.pth"))
        self.alg.actor_model.ChiNet.load_state_dict(checkpt["state_dict"])
        checkpt = torch.load(os.path.join(path, "win", "checkpt.pth"))
        self.alg.actor_model.WinNet.load_state_dict(checkpt["state_dict"])
        checkpt = torch.load(os.path.join(path, "richi", "checkpt.pth"))
        self.alg.actor_model.RichiNet.load_state_dict(checkpt["state_dict"])

    def predict(self, obs, guzhang_first, determine=True):
        action_feature, action, input = self.alg.policy(obs, guzhang_first, determine=determine)
        action = torch.squeeze(action)
        return input, action, action_feature

    def add_experience(self, experience, global_feature):
        self.rpm.append(experience, global_feature)

    def learn_fast(self, critic, ps_weight=2.0):
        """ sample batch, compute q_target and train
        """
        loss = 0.
        self.alg.train()
        self.global_train_step += 1
        experience = self.rpm.sample_batch(self.batch_size)
        bszie = len(experience)
        play_batch = {"target": "play", "feature": [], "mask": [], "global_feature": [],
                      "previous_global_feature": [], "ps_feature": []}
        richi_batch = {"target": "richi", "feature": [], "mask": [], "global_feature": [],
                       "previous_global_feature": []}
        others_win_batch = {"target": "others_win", "feature": [], "global_feature": [],
                            "previous_global_feature": []}
        others_gang_batch = {"target": "others_gang", "feature": [], "global_feature": [],
                             "previous_global_feature": [], "id": []}
        others_peng_batch = {"target": "others_peng", "feature": [], "global_feature": [],
                             "previous_global_feature": [], "id": []}
        # chi与gang难以并行 每次决策数量都不固定
        for exp in experience:
            obj = exp["target"]
            target = obj["target"]

            if target == "gang":
                input, action, action_feature = self.predict(obj, guzhang_first=False, determine=False)
                global_feature = torch.squeeze(exp["global_feature"])
                previous_global_feature = exp["previous_global_feature"]
                global_feature = torch.cat((global_feature, action_feature), dim=0)[None, None, :, :]
                if previous_global_feature is not None:
                    global_feature = torch.cat((previous_global_feature, global_feature), dim=1)
                with torch.no_grad():
                    global_feature = global_feature.cuda()
                    reward = critic(global_feature)
                    reward = reward.detach()
                flag = True if input["id"] is not None else False
                logp = torch.log(action[0] if flag else action[1])
                assert not torch.isinf(logp) and not torch.isnan(logp)
                loss = loss + (-logp * reward)

            elif target == "others_chi":
                input, action, action_feature = self.predict(obj, guzhang_first=False, determine=False)
                global_feature = torch.squeeze(exp["global_feature"])
                previous_global_feature = exp["previous_global_feature"]
                global_feature = torch.cat((global_feature, action_feature), dim=0)[None, None, :, :]
                if previous_global_feature is not None:
                    global_feature = torch.cat((previous_global_feature, global_feature), dim=1)
                with torch.no_grad():
                    global_feature = global_feature.cuda()
                    reward = critic(global_feature)
                reward = reward.detach()
                ids = input["ids"]
                if ids is not None:
                    logp = torch.log(action[ids[1]])
                    assert not torch.isinf(logp) and not torch.isnan(logp)
                    loss = loss + (-logp * reward)
                else:
                    logp = torch.log(torch.cumprod(torch.ones_like(action) - action, dim=-1)[-1])
                    assert not torch.isinf(logp) and not torch.isnan(logp)
                    loss = loss + (-logp * reward)

            elif target == "richi":
                richi_batch["feature"].append(obj["feature"])
                richi_batch["mask"].append(obj["mask"])
                richi_batch["global_feature"].append(exp["global_feature"])
                richi_batch["previous_global_feature"].append(exp["previous_global_feature"])
            elif target == "play":
                play_batch["feature"].append(obj["feature"])
                play_batch["mask"].append(obj["mask"])
                play_batch["global_feature"].append(exp["global_feature"])
                play_batch["previous_global_feature"].append(exp["previous_global_feature"])
                play_batch["ps_feature"].append(exp["xt_feature"]["feature"][:4, :])
            elif target == "others_win" or target == "qianggang":
                others_win_batch["feature"].append(obj["feature"])
                others_win_batch["global_feature"].append(exp["global_feature"])
                others_win_batch["previous_global_feature"].append(exp["previous_global_feature"])
            elif target == "others_gang":
                feature = obj["feature"]
                id = obj["id"][0]
                target = torch.zeros((1, 27)).to(feature)
                target[:, id] = 1
                feature = torch.cat((feature, target), dim=0)
                others_gang_batch["feature"].append(feature)
                others_gang_batch["global_feature"].append(exp["global_feature"])
                others_gang_batch["previous_global_feature"].append(exp["previous_global_feature"])
            elif target == "others_peng":
                feature = obj["feature"]
                id = obj["id"][0]
                target = torch.zeros((1, 27)).to(feature)
                target[:, id] = 1
                feature = torch.cat((feature, target), dim=0)
                others_peng_batch["feature"].append(feature)
                others_peng_batch["global_feature"].append(exp["global_feature"])
                others_peng_batch["previous_global_feature"].append(exp["previous_global_feature"])

        loss = loss + self.alg.policy_batch(play_batch, critic, ps_weight=ps_weight)
        if len(richi_batch["feature"]) > 0:
            loss = loss + self.alg.policy_batch(richi_batch, critic)
        if len(others_gang_batch["feature"]) > 0:
            loss = loss + self.alg.policy_batch(others_gang_batch, critic)
        if len(others_peng_batch["feature"]) > 0:
            loss = loss + self.alg.policy_batch(others_peng_batch, critic)
        if len(others_win_batch["feature"]) > 0:
            loss = loss + self.alg.policy_batch(others_win_batch, critic)
        loss = loss / bszie
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise EOFError
        if self.epoch < self.warmup_epoch:
            self.warm_up(self.actor_optimizer, self.epoch)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss

    def learn(self, critic):
        """ sample batch, compute q_target and train
        """
        self.alg.train()
        self.global_train_step += 1
        experience = self.rpm.sample_batch(self.batch_size)
        batch_logp = []
        batch_reward = []
        for exp in experience:
            obj = exp["target"]
            target = obj["target"]
            input, action, action_feature = self.predict(obj)
            global_feature = exp["global_feature"]
            previous_global_feature = exp["previous_global_feature"]
            global_feature = torch.cat((global_feature, action_feature), dim=0)[None, None, :, :]
            if previous_global_feature is not None:
                global_feature = torch.cat((previous_global_feature, global_feature), dim=1)
            with torch.no_grad():
                global_feature = global_feature.cuda()
                reward = critic(global_feature)
                if target == "play":
                    ps_feature = exp["xt_feature"]["feature"][:4, :]
                    id = input["play"]
                    _ps_feature = torch.zeros((1, 27))
                    _ps_feature[0, id] = 1
                    ps_feature = torch.cat((ps_feature, _ps_feature), dim=0)[None, :, :]
                    reward = reward - critic.get_punish(ps_feature) * 2
                reward = reward.detach()
            batch_reward.append(reward)
            if target == "win":
                batch_logp.append(torch.tensor(0.).cuda())
            elif target == "gang":
                flag = True if input["id"] is not None else False
                logp = torch.log(action[0] if flag else action[1])
                assert not torch.isinf(logp) and not torch.isnan(logp)
                batch_logp.append(logp)
            elif target == "richi":
                flag = input["flag"]
                logp = torch.log(action[0] if flag else action[1])
                assert not torch.isinf(logp) and not torch.isnan(logp)
                batch_logp.append(logp)
            elif target == "play":
                assert action.sum() > 1. - 1e-5
                id = input["play"]
                logp = torch.log(action[id])
                assert not torch.isinf(logp) and not torch.isnan(logp)
                batch_logp.append(logp)
            elif target == "others_win":
                flag = input["flag"]
                logp = torch.log(action[0] if flag else action[1])
                assert not torch.isinf(logp) and not torch.isnan(logp)
                batch_logp.append(logp)
            elif target == "others_gang":
                flag = input["flag"]
                logp = torch.log(action[0] if flag else action[1])
                assert not torch.isinf(logp) and not torch.isnan(logp)
                batch_logp.append(logp)
            elif target == "others_peng":
                flag = input["flag"]
                logp = torch.log(action[0] if flag else action[1])
                assert not torch.isinf(logp) and not torch.isnan(logp)
                batch_logp.append(logp)
            elif target == "others_chi":
                ids = input["ids"]
                if ids is not None:
                    logp = torch.log(action[ids[1]])
                    assert not torch.isinf(logp) and not torch.isnan(logp)
                    batch_logp.append(logp)
                else:
                    logp = torch.log(torch.cumprod(torch.ones_like(action) - action, dim=-1)[-1])
                    assert not torch.isinf(logp) and not torch.isnan(logp)
                    batch_logp.append(logp)

        batch_logp = torch.stack(batch_logp, dim=0)
        batch_reward = torch.stack(batch_reward, dim=0)
        loss = -batch_logp * batch_reward
        loss = loss.mean()
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise EOFError
        # if self.epoch < self.warmup_epoch:
        #     self.warm_up(self.actor_optimizer, self.epoch)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss

    def learn_critic(self, critic, optimizer):
        # if self.epoch < self.warmup_epoch:
        #     self.warm_up(optimizer, self.epoch)
        x, y, mask = self.rpm.sample_global_feature(self.critic_batch_szie)
        x = x.cuda()
        y = y.cuda()
        mask = mask.cuda()
        loss = critic(x, mask, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ps_x, ps_y = self.rpm.sample_ps_feature(self.batch_size)
        loss_ps = critic.get_punish(ps_x, ps_y)
        optimizer.zero_grad()
        loss_ps.backward()
        optimizer.step()
        return loss, loss_ps
