import copy
from torch.optim import Adam

from .actor_network import judgeNet, playNet
import torch
import numpy as np
from torch.nn import functional as F
from torch import nn


class MAModel(nn.Module):
    def __init__(self, obs_dim, uid, num_blocks):
        super(MAModel, self).__init__()
        self.uid = uid
        self.actor_model = ActorModel(obs_dim, num_blocks)

    def policy(self, obs):
        return self.actor_model(obs)

    def policy_batch(self, batch, critic, ps_weight=2):
        return self.actor_model.learn_batch(batch, critic, ps_weight=ps_weight)

    def forawrd(self, obs):
        return self.actor_model(obs)

    def get_actor_params(self):
        return self.actor_model.parameters()


class ActorModel(nn.Module):
    def __init__(self, obs_dim, num_blocks):
        super(ActorModel, self).__init__()
        self.WinNet = judgeNet(obs_dim, num_block=num_blocks, hidden_dim=256).cuda()
        self.RichiNet = judgeNet(obs_dim, num_block=num_blocks, hidden_dim=256).cuda()
        self.GangNet = judgeNet(obs_dim + 1, num_block=num_blocks, hidden_dim=256).cuda()
        self.PengNet = judgeNet(obs_dim + 1, num_block=num_blocks, hidden_dim=256).cuda()
        self.ChiNet = judgeNet(obs_dim + 1, num_block=num_blocks, hidden_dim=256).cuda()
        self.PlayNet = playNet(obs_dim, num_block=num_blocks, hidden_dim=256).cuda()

    # 除了吃与杠以外的批次计算
    def learn_batch(self, batch, critic, ps_weight=4.0):
        target = batch["target"]
        logp = []
        global_mask = []
        all_global_feature = []
        feature = torch.stack(batch["feature"], dim=0).cuda()
        last_reward = torch.stack(batch["last_reward"], dim=0).cuda()
        for idx, global_feature in enumerate(batch["previous_global_feature"]):
            if global_feature is None:
                global_feature = torch.zeros(40, 409, 27)
                m = torch.zeros(40)
                m[:1] = torch.ones(1)
                all_global_feature.append(global_feature)
                global_mask.append(m)
            else:
                global_feature = torch.squeeze(global_feature)
                if global_feature.dim() < 3:
                    global_feature = global_feature[None, :, :]
                pad = 40 - global_feature.shape[0]
                m = torch.zeros(40)
                m[:global_feature.shape[0]+1] = torch.ones(global_feature.shape[0]+1)
                global_mask.append(m)
                zeros = torch.zeros(pad, global_feature.shape[1], 27).to(global_feature)
                global_feature = torch.cat((global_feature, zeros), dim=0)
                all_global_feature.append(global_feature)
        global_mask = torch.stack(global_mask, dim=0)
        if target == "richi":
            mask = torch.stack(batch["mask"], dim=0).to(feature)
            p_richi = self.RichiNet(feature)
            action = torch.squeeze(p_richi)
            if action.dim() < 2:
                action = action[None, :]
            with torch.no_grad():
                p = self.PlayNet(feature, mask).detach()
            for i in range(action.shape[0]):
                action_feature = torch.zeros(8, 27)
                if action[i, 0] > action[i, 1]:
                    logp.append(torch.log(action[i, 0]))
                    id = torch.argmax(p[i, :])
                    action_feature[2, :] = torch.ones(27)
                    action_feature[3, id] = 1
                else:
                    logp.append(torch.log(action[i, 1]))
                    action_feature[2, :] = torch.ones(27) * -1
                n = int(global_mask[i].sum() - 1)
                now_global_feature = torch.cat((batch["global_feature"][i], action_feature), dim=0)
                all_global_feature[i][n, :] = now_global_feature
        if target == "play":
            mask = torch.stack(batch["mask"], dim=0).to(feature)
            p = self.PlayNet(feature, mask)
            action = torch.squeeze(p)
            for i in range(action.shape[0]):
                action_feature = torch.zeros(8, 27)
                id = torch.argmax(action[i, :])
                zeros = torch.zeros(1, 27)
                zeros[0, id] = 1
                batch["ps_feature"][i] = torch.cat((batch["ps_feature"][i], zeros), dim=0)
                logp.append(torch.log(action[i, id]))
                action_feature[3, id] = 1.
                n = int(global_mask[i].sum() - 1)
                now_global_feature = torch.cat((batch["global_feature"][i], action_feature), dim=0)
                all_global_feature[i][n, :] = now_global_feature

        if target == "others_win" or target == "qianggang":
            p = self.WinNet(feature)
            action = torch.squeeze(p)
            if action.dim() < 2:
                action = action[None, :]
            for i in range(action.shape[0]):
                action_feature = torch.zeros(8, 27)
                if action[i, 0] > action[i, 1]:
                    logp.append(torch.log(action[i, 0]))
                    action_feature[4, :] = torch.ones(27)
                else:
                    logp.append(torch.log(action[i, 1]))
                    action_feature[4, :] = torch.ones(27) * -1
                n = int(global_mask[i].sum() - 1)
                now_global_feature = torch.cat((batch["global_feature"][i], action_feature), dim=0)
                all_global_feature[i][n, :] = now_global_feature

        if target == "others_gang":
            p = self.GangNet(feature)
            action = torch.squeeze(p)
            if action.dim() < 2:
                action = action[None, :]
            for i in range(action.shape[0]):
                action_feature = torch.zeros(8, 27)
                if action[i, 0] > action[i, 1]:
                    logp.append(torch.log(action[i, 0]))
                    action_feature[5, :] = torch.ones(27)
                else:
                    logp.append(torch.log(action[i, 1]))
                    action_feature[5, :] = torch.ones(27) * -1
                n = int(global_mask[i].sum() - 1)
                now_global_feature = torch.cat((batch["global_feature"][i], action_feature), dim=0)
                all_global_feature[i][n, :] = now_global_feature
        if target == "others_peng":
            p = self.PengNet(feature)
            action = torch.squeeze(p)
            if action.dim() < 2:
                action = action[None, :]
            for i in range(action.shape[0]):
                action_feature = torch.zeros(8, 27)
                if action[i, 0] > action[i, 1]:
                    logp.append(torch.log(action[i, 0]))
                    action_feature[6, :] = torch.ones(27)
                else:
                    logp.append(torch.log(action[i, 1]))
                    action_feature[6, :] = torch.ones(27) * -1
                n = int(global_mask[i].sum() - 1)
                now_global_feature = torch.cat((batch["global_feature"][i], action_feature), dim=0)
                all_global_feature[i][n, :] = now_global_feature
        logp = torch.stack(logp, dim=0)
        all_global_feature = torch.stack(all_global_feature, dim=0).cuda()
        with torch.no_grad():
            reward = critic(all_global_feature, global_mask) - last_reward
            if target == "play":
                ps_feature = torch.stack(batch["ps_feature"], dim=0).cuda()
                ps = critic.get_punish(ps_feature)
                reward = reward - ps * ps_weight
        loss = -logp * reward
        loss = loss.sum()
        return loss

    # 输出的action形状为(7, 27) 对应7种动作 写成(7,27)的形状是因为方便输入critic
    def forward(self, obj, determine=True):
        target = obj["target"]
        feature = obj["feature"].cuda()
        if len(feature.shape) == 2:
            feature = torch.unsqueeze(feature, dim=0)
        out = torch.zeros(8, 27)
        if target == "win":
            return torch.zeros(8, 27), torch.tensor([1., 0.]), True
        elif target == "gang":
            angang_ids = obj["angang_ids"]
            bugang_ids = obj["bugang_ids"]
            output = {"id": None, 'type': None}
            action = torch.zeros(27).to(feature)

            for id in angang_ids:
                target = torch.zeros((1, 1, 27)).to(feature)
                target[:, :, id] = 4
                feat = copy.deepcopy(feature)
                feat = torch.cat((feat, target), dim=1)
                p = self.GangNet(feat)
                action = torch.squeeze(p)
                if action[0] > action[1]:
                    out[0, id] = 4
                    output["id"] = id
                    output["type"] = "angang"
                else:
                    out[0, id] = -1
                    return out.detach().cpu(), action, output

            for id in bugang_ids:
                target = torch.zeros((1, 1, 27)).to(feature)
                target[:, :, id] = 1
                feat = copy.deepcopy(feature)
                feat = torch.cat((feat, target), dim=1)
                p = self.GangNet(feat)
                action = torch.squeeze(p)
                if action[0] > action[1]:
                    out[1, id] = 1
                    output["id"] = id
                    output["type"] = "bugang"
                    return out.detach().cpu(), action, output

            return out.detach().cpu(), action, output
        elif target == "richi":
            p_richi = self.RichiNet(feature)
            action = torch.squeeze(p_richi)
            output = {"flag": False}
            mask = obj["mask"].cuda()
            if action[0] > action[1]:
                output["flag"] = True
                p = self.PlayNet(feature, mask)
                assert p.sum() > 1. - 1e-5
                id = torch.argmax(p)
                output["play"] = id
                out[2, :] = torch.ones(27)
                out[3, id] = 1.
                return out.detach().cpu(), action, output
            else:
                out[2, :] = torch.ones(27) * -1
                return out.detach().cpu(), action, output


        elif target == "play":
            mask = obj["mask"].cuda()
            p = self.PlayNet(feature, mask)
            assert p.sum() > 1. - 1e-5
            action = torch.squeeze(p)
            # out[3, :] = None
            output = {"p": p}
            id = torch.argmax(p)
            output["play"] = id
            out[3, id] = 1.

        elif target == "others_win" or target == "qianggang":
            id = obj["id"]
            output = {"flag": False, "id": id}
            # target = torch.zeros((1, 1, 27)).to(feature)
            # target[:, :, id] = 3
            # feature = torch.cat((feature, target), dim=1)
            p = self.WinNet(feature)
            action = torch.squeeze(p)
            if action[0] > action[1]:
                output["flag"] = True
                out[4, :] = torch.ones(27)
                return out.detach().cpu(), action, output
            else:
                out[4, :] = torch.ones(27) * -1
                return out.detach().cpu(), action, output
        elif target == "others_gang":
            id = obj["id"]
            output = {"flag": False, "id": id}
            target = torch.zeros((1, 1, 27)).to(feature)
            target[:, :, id] = 1
            feat = copy.deepcopy(feature)
            feat = torch.cat((feat, target), dim=1)
            p = self.GangNet(feat)
            action = torch.squeeze(p)
            if action[0] > action[1]:
                output["flag"] = True
                out[5, id] = 1
                return out.detach().cpu(), action, output
            else:
                out[5, id] = -1
                return out.detach().cpu(), action, output
        elif target == "others_peng":
            id = obj["id"]
            output = {"flag": False, "id": id}
            target = torch.zeros((1, 1, 27)).to(feature)
            target[:, :, id] = 1
            feat = copy.deepcopy(feature)
            feat = torch.cat((feat, target), dim=1)
            p = self.PengNet(feat)
            action = torch.squeeze(p)
            if action[0] > action[1]:
                output["flag"] = True
                out[6, :] = torch.ones(27)
                return out.detach().cpu(), action, output
            else:
                out[6, :] = torch.ones(27) * -1
                return out.detach().cpu(), action, output
        elif target == "others_chi":
            idlist = obj["idlist"]
            for idx, ids in enumerate(idlist):
                target = torch.zeros((len(idlist), 1, 27)).to(feature)
                for id in ids:
                    target[idx, :, id] = 1
            feature = feature.expand(len(idlist), -1, -1)
            feature = torch.cat((feature, target), dim=1)
            p = self.ChiNet(feature)  # len(idlist), 2
            for i, ids in enumerate(idlist):
                idx = ids[1]
                action = torch.zeros(27).to(p)
                action[idx] = p[i, 0]
            output = {"ids": None}
            p = copy.copy(action.detach().cpu())
            while p.sum() != 0:
                idx = int(torch.argmax(action))
                if action[idx] > 0.5:
                    out[7, idx] = 1
                    output["ids"] = [idx - 1, idx, idx + 1]
                    return out.detach().cpu(), action, output
                else:
                    out[7, idx] = -1
                    p[idx] = 0
            return out.detach().cpu(), action, output
        else:
            raise AssertionError
