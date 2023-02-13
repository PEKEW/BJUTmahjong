#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import logging
import os
import random
import time
import argparse
import numpy as np
import torch
from torch.backends import cudnn
from torch.optim import Adam
from agent_critic.agent import Agent
from agent_critic.model import MAModel
from env.env import game
from agent_critic.critic_network import Critic

CRITIC_LR = 0.001  # learning rate for the critic model
ACTOR_LR = 0.001  # learning rate of the actor model
BATCH_SIZE = 1024
UPDATE_EPOCH = 1
ps_weight = 0.
save_per_epochs = 10
keep_xiangting = True  # 规则:保证相听不会下降
guzhang_first = True   # 规则:有限出完全不相邻的牌
logging.basicConfig(filename='rf_log.txt',
                    filemode='w')
log = logging.getLogger()
log.setLevel(logging.INFO)


# 设定种子
def common_init(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


# 单独训练critic
def train_critic_episode(agents, epoch, critic, critic_optimizer):
    mean_critic_loss = 0
    mean_ps_loss = 0
    for i, agent in enumerate(agents):
        critic_loss, ps_loss = agent.learn_critic(critic, critic_optimizer)
        mean_critic_loss += critic_loss / 4
        mean_ps_loss += ps_loss / 4
    log.info("critic的损失为{:.4f}, 相听惩罚项的损失为{:.4f}".format(float(mean_critic_loss), float(mean_ps_loss)))
    path = os.path.join("save/critic", "checkpt_{:}.pth".format(int(epoch)))
    torch.save({
        'state_dict': critic.state_dict(),
        'optim_state_dict': critic_optimizer.state_dict(),
        'epoch': epoch
    }, path)


# 完成一局比赛 并训练actor与critic
def run_episode(env, agents, critic, critic_optimizer, epoch, train_agent=True, train_critic=True):
    env.reset()
    steps = 0
    while True:
        with torch.no_grad():
            target = env.step()
            if target["target"] == "draw":
                break
            steps += 1
            uid = target["uid"]
            input, action, action_feature = agents[uid].predict(target, guzhang_first=guzhang_first)
            experience, global_feature = env.step(input)
            # 检查这里是否为 1 1 D 27
            if target["target"] == "play":
                xt_ps_feature = experience["xt_feature"]
                agents[uid].rpm.append_xt_ps(xt_ps_feature)

            global_feature = torch.cat((global_feature, action_feature), dim=0)[None, None, :, :]
            previous_global_feature = agents[uid].rpm.global_feature["feature"]
            if previous_global_feature is not None:
                experience["previous_global_feature"] = previous_global_feature
                experience["target"] = target
                experience["action"] = action
                agents[0].add_experience(experience, None)
                agents[uid].add_experience(None, global_feature)
            else:
                experience["previous_global_feature"] = None
                # agents[0].add_experience(experience, None)  # 训练第一步会产生干扰 这里就不训练第一步了
                agents[uid].add_experience(None, global_feature)
            terminal = experience["terminal"]
            if terminal:
                break

    if len(agents[0].rpm.all_global_feature["feature"]) > 10:
        epoch += 1
    score = env.history.score
    # max_score = score.max()
    for i in range(4):
        # reward = (score[i] - max_score).view(1, 1)
        reward = score[i]  # 这里将奖励由分数变为与均值的差
        # reward = score[i]
        agents[i].rpm.append_epoch_feature(reward)
        # learn policy
    for i in range(4):
        log.info("玩家{:}得分为:{:}".format(i, env.history.score[i]))
    # for i, agent in enumerate(agents):
    if train_critic:
        mean_critic_loss = 0
        mean_ps_loss = 0
        if len(agents[0].rpm.all_global_feature["feature"]) > 10:
            for i, agent in enumerate(agents):
                critic_loss, ps_loss = agent.learn_critic(critic, critic_optimizer)
                mean_critic_loss += critic_loss / 4
                mean_ps_loss += ps_loss / 4
            log.info("critic的损失为{:.4f}, 相听惩罚项的损失为{:.4f}".format(float(mean_critic_loss), float(mean_ps_loss)))
            if epoch % save_per_epochs == 0:
                path = os.path.join("save/critic", "checkpt_{:}.pth".format(int(epoch)))
                torch.save({
                    'state_dict': critic.state_dict(),
                    'optim_state_dict': critic_optimizer.state_dict(),
                    'epoch': epoch
                }, path)

    if train_agent:
        if len(agents[0].rpm.all_global_feature["feature"]) > 10:
            agent_loss = agents[0].learn_fast(critic, ps_weight=ps_weight)
            log.info("agent的损失为{:.4f}".format(float(agent_loss)))
            agents[0].epoch += 1
            if epoch % save_per_epochs == 0:
                for i in range(1, 4):
                    path_model = "save/agent/checkpt_{:}.pth".format(epoch-save_per_epochs)
                    agents[i].resume(path_model, load_optimizer=False)
                    for i in range(4):
                        agents[i].save(only_feature=False if i == 0 else True)
    return steps, epoch


def train_agent(resume=True, epoch=0, seed=0, load_global_feature=True, train_agent=True, train_critic=True,
                train_agent_only=False):
    path = "save"
    # seed = int(time.time())
    common_init(seed)
    env = game(1, path)
    # build agents
    agents = []
    for i in range(env.n):
        model = MAModel(obs_dim=401, uid=i, num_blocks=30)
        agent = Agent(model=model, id=i, batch_size=BATCH_SIZE, critic_batch_szie=128, save_path=path, lr=ACTOR_LR,
                      warmup_epoch=200)
        agents.append(agent)
    critic = Critic(409, hidden_dim=128, kernel_size=3, num_layers=8, bias=True).cuda()
    optimizer = Adam(critic.parameters(), lr=CRITIC_LR)
    if not resume:
        path = "save/pretrain"
        for i in range(4):
            agents[i].pretrain(path)
        checkpt = torch.load("save/pretrain/critic/checkpt.pth", map_location=lambda storage, loc: storage)
        critic.load_state_dict(checkpt["state_dict"])
        epoch = 0.
    else:
        for i in range(4):
            path_model = "save/agent/checkpt_{:}.pth".format(epoch)
            if load_global_feature:
                path_feature = "save/global_feature/global_feature_uid{:}.pth".format(i)
            else:
                path_feature = None
            agents[i].resume(path_model, path_feature, load_optimizer=True if i == 0 else False)
        try:
            path_critic = "save/critic/checkpt_{:}.pth".format(epoch)
            checkpt = torch.load(path_critic, map_location=lambda storage, loc: storage)
            critic.load_state_dict(checkpt["state_dict"], strict=True)
            optimizer.load_state_dict(checkpt["optim_state_dict"])
        except:
            raise IOError("load critic error!!!")
    total_steps = 0
    total_episodes = epoch
    while True:
        log.info("epoch:{:} start".format(total_episodes))
        # run an episode
        if train_agent_only:
            train_critic_episode(agents, epoch, critic, optimizer)
        else:
            steps, total_episodes = run_episode(env, agents, critic, optimizer, total_episodes, train_agent,
                                                train_critic)
            # Record reward
            total_steps += steps


if __name__ == '__main__':
    epoch = 14050
    resume = True
    seed = int(time.time())
    train_agent(resume, epoch, seed, load_global_feature=True, train_agent=True, train_critic=True,
                train_agent_only=False)
