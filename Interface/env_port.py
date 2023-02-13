import logging
import time
from .history import History
import copy
import torch
import numpy as np
from .utils import clc_xiangting, _set, msg2id, id2msg


class players:
    def __init__(self, id):
        self.hand = torch.zeros((27,))
        self.inhand = torch.zeros((27,))
        self.id = id
        self.chi = []
        self.peng = []
        self.gang = []
        self.xt = 8
        self.richi = False

    def reset(self):
        self.hand = torch.zeros((27,))
        self.inhand = torch.zeros((27,))
        self.chi = []
        self.peng = []
        self.gang = []
        self.xt = 8
        self.richi = False

    def update(self, meta_data):
        self.id = meta_data["seat"]
        self.inhand = msg2id(meta_data["hand"], tensor=True)
        self.hand = copy.deepcopy(self.inhand)
        if meta_data["pon_hand"] != "":
            self.peng = meta_data["pon_hand"].split(",")
        if meta_data["chow_hand"] != "":
            self.chi = meta_data["chow_hand"].split(",")
        if meta_data["kon_hand"] != "":
            self.gang = meta_data["kon_hand"].split(",")
        for data in self.peng:
            self.hand = self.hand + msg2id(data, tensor=True)
        for data in self.chi:
            self.hand = self.hand + msg2id(data, tensor=True)
        for data in self.gang:
            self.hand = self.hand + msg2id(data, tensor=True)
        self.action_after_draw()

    def action_after_draw(self):
        minus = len(self.peng) * 2 + len(self.gang) * 2 + len(self.chi) * 2
        self.xt = clc_xiangting(self.inhand, minus)

    def get_priv_feature(self):
        inhand = torch.zeros(4, 27)
        for id, nums in enumerate(self.inhand):
            for i in range(nums.int()):
                inhand[i, id] += 1
        xt = torch.tensor(self.xt).view(1, 1).expand(1, 27)
        uid = torch.tensor(0).view(1, 1).expand(1, 27)  # 训练的时候没换uid...
        feature_private = torch.cat([inhand, xt, uid], dim=0)
        return feature_private

    def get_feature(self, history):
        features = history.get_global_feature(know_all=False)
        feature_private = self.get_priv_feature()
        feature = torch.cat((feature_private, features), dim=0)
        return feature

    def play_vaild_check(self, id):
        cards = self.inhand
        if cards[id] >= 1:
            return True
        return False

    def peng_vaild_check(self, id):
        cards = self.inhand
        if cards[id] >= 2:
            return True
        return False

    def chi_vaild_check(self, id):
        cards = self.inhand
        res = []
        for i in range(-1, 2):
            idlist = [id - 1 + i, id + i, id + 1 + i]
            mean = id + i
            if mean % 9 in [0, 8] or mean < 0 or mean > 27:
                continue
            flag = True
            for _id in idlist:
                if _id != id and cards[_id] < 1:
                    flag = False
                    break
            if flag:
                res.append(idlist)
        return res

    def richi_vaild_check(self, check=False):
        cards = self.inhand
        minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
        if check:
            self.xt = clc_xiangting(cards, minus)
        if self.xt == 0:
            mask = torch.zeros(27, )
            ids = torch.arange(27)[self.inhand > 0].tolist()
            for id in ids:
                tmp = copy.deepcopy(cards)
                tmp[id] -= 1
                xt = clc_xiangting(tmp, minus)
                if xt == 0:
                    mask[id] = 1.
            return mask
        return None

    # TODO 删除多余的相听计算
    def win_vaild_check(self, check=False, id=None):
        xt = 8
        if id is not None:
            cards = copy.deepcopy(self.inhand)
            cards[id] += 1
            minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
            xt = clc_xiangting(cards, minus)
        if check:
            cards = self.inhand
            minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
            self.xt = xt = clc_xiangting(cards, minus)
        if self.xt == -1 or xt == -1:
            return True
        else:
            return False

    # FIXME 这里发现了bug 当没有碰 而是吃了3次相同牌的情况下 也会触发补杠 已修复
    def gang_vaild_check(self, id=None):
        cards = self.inhand
        if id is not None:
            if cards[id] >= 3:
                if self.richi:
                    minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
                    tmp = copy.deepcopy(cards)
                    tmp[id] -= 1
                    xt = clc_xiangting(tmp, minus)
                    if xt != 0:
                        return False
                return True
        else:
            angang_ids = torch.arange(0, 27)[(self.inhand == 4)].tolist()
            bugang_ids = list(set(torch.arange(0, 27)[(self.hand == 4)].tolist()) & set(self.peng))
            if self.richi:
                _angang_ids, _bugang_ids = [], []
                for idx, id in enumerate(angang_ids):
                    minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
                    tmp = copy.deepcopy(cards)
                    tmp[id] -= 4
                    xt = clc_xiangting(tmp, minus + 2)
                    if xt == 0:
                        _angang_ids.append(angang_ids[idx])
                for idx, id in enumerate(angang_ids):
                    minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
                    tmp = copy.deepcopy(cards)
                    tmp[id] -= 1
                    xt = clc_xiangting(tmp, minus)
                    if xt == 0:
                        _angang_ids.append(_bugang_ids[idx])
                angang_ids = _angang_ids
                bugang_ids = _bugang_ids
            return angang_ids, bugang_ids


logging.basicConfig(filename='rf_log.txt',
                    filemode='w', )


class game:
    def __init__(self):
        self.zhuang = -1
        self.player = players(-1)
        self.richi = False
        self.history = History()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def reset(self, meta_data):
        self.richi = False
        self.zhuang = meta_data["dealer"]
        self.player.reset()
        self.player.update(meta_data)
        if self.player.inhand.sum() + (len(self.player.chi) + len(self.player.peng) + len(self.player.gang)) * 3 == 14:
            now = self.player.id
        else:
            now = int(meta_data["history"][-1].split(",")[0])
        self.history.reset()
        self.history.update(meta_data, now)
        self.richi = self.history.richi[self.player.id]

    def _win_vaild_check(self, id=None):
        return self.player.win_vaild_check(check=False, id=id)

    def _richi_vaild_check(self):
        if self.richi:
            return None
        return self.player.richi_vaild_check()

    def _gang_vaild_check(self, id=None):
        return self.player.gang_vaild_check(id)

    def _peng_vaild_check(self, id=None):
        if self.richi:
            return False
        return self.player.peng_vaild_check(id)

    def _chi_vaild_check(self, id=None):
        if self.richi:
            return []
        return self.player.chi_vaild_check(id)

    def step(self, meta_data, agent):
        ReqMess = {'code': 200, 'action_type': "Pass", 'action_content': None}
        exp, id = None, None
        self.reset(meta_data)
        if self.player.inhand.sum() + (len(self.player.chi) + len(self.player.peng) + len(self.player.gang)) * 3 == 14:
            target = 'win'
        else:
            target = 'others_win'
            exp = meta_data["history"][-1].split(",")
            id = msg2id(exp[2], tensor=False)[0]
        feature = self.player.get_feature(self.history).cuda()
        if target == 'win':
            if self._win_vaild_check():
                object = {"target": "win", "feature": feature}
                input, _, _ = agent.predict(object)
                if input["flag"]:
                    ReqMess['action_type'] = "Win"
                    ReqMess['action_content'] = None
                    return ReqMess
            target = "gang"

        if target == 'gang':
            angang_ids, bugang_ids = self._gang_vaild_check()
            if len(angang_ids) or len(bugang_ids) != 0:
                object = {"target": "gang", "angang_ids": angang_ids, "bugang_ids": bugang_ids, "feature": feature}
                input, _, _ = agent.predict(object)
                if input["id"] is not None:
                    ReqMess['action_type'] = "Gang"
                    ReqMess['action_content'] = id2msg(input["id"])
                    return ReqMess
            if self.richi:
                ReqMess['action_type'] = "Pass"
                ReqMess['action_content'] = None
                return ReqMess
            target = "richi"

        if target == "richi":
            mask = self._richi_vaild_check()
            if mask is not None:
                object = {"target": "richi", "mask": mask, "feature": feature}
                input, _, _ = agent.predict(object)
                if input["flag"]:
                    ReqMess['action_type'] = "Listen"
                    ReqMess['action_content'] = id2msg(input["play"])
                    return ReqMess
            target = "play"

        if target == 'play':
            mask = (self.player.inhand > 0).int()
            object = {"target": "play", "mask": mask, "feature": feature}
            input, _, _ = agent.predict(object)
            ReqMess['action_type'] = "Discard"
            ReqMess['action_content'] = id2msg(input["play"])
            return ReqMess

        if target == 'others_win':
            if self._win_vaild_check(id):
                object = {"target": "others_win", "id": id, "feature": feature}
                input, _, _ = agent.predict(object)
                if input["flag"]:
                    ReqMess['action_type'] = "Win"
                    ReqMess['action_content'] = None
                    return ReqMess
            if exp[1] == "Kon":
                ReqMess['action_type'] = "Pass"
                ReqMess['action_content'] = None
                return ReqMess
            else:
                target = 'others_gang'

        if target == 'others_gang':
            flag = self._gang_vaild_check(id)
            if flag is not None:
                object = {"target": "others_gang", "id": id, "feature": feature}
                input, _, _ = agent.predict(object)
                flag = input["flag"]
                if flag:
                    ReqMess['action_type'] = "Kon"
                    ReqMess['action_content'] = id2msg([id, ] * 4)
                    return ReqMess
            target = 'others_peng'

        if target == 'others_peng':
            if self._peng_vaild_check(id):
                object = {"target": "others_peng", "id": id, "feature": feature}
                input, _, _ = agent.predict(object)
                flag = input["flag"]
                if flag:
                    ReqMess['action_type'] = "Pon"
                    ReqMess['action_content'] = id2msg([id, ] * 3)
                    return ReqMess
            target = 'others_chi'

        if target == "others_chi":
            idlist = self._chi_vaild_check(id)
            if len(idlist) != 0:
                object = {"target": "others_chi", "idlist": idlist, "feature": feature}
                input, _, _ = agent.predict(object)
                ids = input["ids"]
                if ids is not None:
                    ReqMess['action_type'] = "Chow"
                    ReqMess['action_content'] = id2msg(ids)
                    return ReqMess
            return ReqMess
