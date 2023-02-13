import torch

# 通过history里的信息来获得特征
from Interface.utils import msg2id


class History:
    def __init__(self):
        self.discard = torch.zeros((4, 27))  # 已经打出的牌
        self.chi = torch.zeros((4, 27))
        self.peng = torch.zeros((4, 27))
        self.bugang = torch.zeros((4, 27))
        self.angang = torch.zeros((4, 27))
        self.zhigang = torch.zeros((4, 27))
        self.cr = 0  # 回合数
        self.matches = 0  # 局数
        self.now = -1  # 当前玩家
        self.richi = [False, ] * 4
        self.zhuang_id = -1
        self.discard_timeline = torch.zeros(4, 80, 27)
        self.score = torch.zeros(4)
        self.num_cards_left = 56
        # feature unseen
        self.inhand = torch.zeros((4, 27))
        self.cards_indeck = torch.zeros((27,))

    def reset(self):
        self.discard = torch.zeros((4, 27))  # 已经打出的牌
        self.chi = torch.zeros((4, 27))
        self.peng = torch.zeros((4, 27))
        self.bugang = torch.zeros((4, 27))
        self.angang = torch.zeros((4, 27))
        self.zhigang = torch.zeros((4, 27))
        self.richi = [False, ] * 4
        self.zhuang_id = -1
        self.discard_timeline = torch.zeros(4, 80, 27)
        self.cr = 0
        self.now = -1
        self.matches = 0
        self.score = torch.zeros(4)
        self.num_cards_left = 56
        self.inhand = torch.zeros((4, 27))
        self.cards_indeck = torch.zeros((27,))

    def update(self, meta_data, now):
        meta_data["history"].pop(0)
        history = meta_data["history"]
        self.now = now
        self.zhuang_id = meta_data["dealer"]
        last_Discard = {"uid": None, "id": None}
        last_Chow = {"uid": None, "id": None, "target": None}
        for exp in history:
            exp = exp.split(",")
            uid = int(exp[0])
            action = exp[1]
            content = exp[2]
            if action == "Discard" or action == "Listen":
                id = msg2id(content, tensor=False)[0]
                self.discard[uid, id] += 1
                self.discard_timeline[uid, self.cr, id] = 1
                self.cr += 1
                self.num_cards_left -= 1
                if action == "Listen":
                    self.score[uid] += 3
                    for i in range(3):
                        self.score[(uid + i) % 4] -= 1
                    self.richi[uid] = True
                if last_Chow["uid"] is not None:
                    if last_Chow["uid"] == uid and last_Chow["id"] == id:
                        self.score[uid] -= 1
                        self.score[last_Chow["target"]] += 1
                    else:
                        last_Chow = {"uid": None, "id": None, "target": None}
                last_Discard["uid"] = uid
                last_Discard["id"] = id

            elif action == "Pon":
                self.num_cards_left += 1
                id = msg2id(content, tensor=False)[0]
                self.peng[uid, id] += 1
                self.score[uid] += 2
                self.score[last_Discard["uid"]] -= 2
            elif action == "Chow":
                self.num_cards_left += 1
                ids = msg2id(content, tensor=False)
                for id in ids:
                    self.chi[uid, id] += 1
                self.score[uid] += 1
                self.score[last_Discard["uid"]] -= 1
                last_Chow = {"uid": uid, "id": last_Discard["id"], "target": last_Discard["uid"]}
            elif action == "Kon":
                id = msg2id(content, tensor=False)[0]
                if self.discard_timeline[:, self.cr - 1, id].sum() == 1:
                    self.zhigang[uid, id] += 1
                    self.score[uid] += 4
                    self.score[last_Discard["uid"]] -= 4
                elif self.peng[uid, id] == 1:
                    self.bugang[uid, id] += 1
                    self.score[uid] += 3
                    for i in range(3):
                        self.score[(uid + i) % 4] -= 1
                else:
                    self.angang[uid, id] += 1
                    self.score[uid] += 9
                    for i in range(3):
                        self.score[(uid + i) % 4] -= 3

    def get_global_feature(self, know_all=False):
        discard = torch.zeros(4, 4, 27)
        for uid in range(4):
            for id, nums in enumerate(self.discard[uid, :]):
                for i in range(nums.int()):
                    discard[uid, i, id] += 1
        discard = discard.view(16, 27)
        discard_time_line = self.discard_timeline.view(320, 27)
        feature_base = torch.cat(
            [discard, self.chi, self.peng, self.zhigang,
             self.bugang, self.angang, discard_time_line], dim=0)
        richi = torch.tensor(self.richi).to(int).view(4, 1).expand(4, 27)
        cr = torch.tensor(self.cr).view(1, 1).expand(1, 27)
        matches = torch.tensor(self.matches).view(1, 1).expand(1, 27)
        score = self.score.view(4, 1).expand(4, 27)
        num_cards_left = torch.tensor(self.num_cards_left).view(1, 1).expand(1, 27)
        zhuang = torch.zeros(4, )
        zhuang[self.zhuang_id] += 1
        zhuang = zhuang.view(4, 1).expand(4, 27)
        now = torch.zeros(4, )
        now[self.now] += 1
        now = now[:, None].expand(4, 27)
        feature_discrete = torch.cat([richi, cr, matches, score, num_cards_left, zhuang, now], dim=0)
        if know_all:
            inhand = torch.zeros(4, 4, 27)
            for uid in range(4):
                for id, nums in enumerate(self.inhand[uid, :]):
                    for i in range(nums.int()):
                        inhand[uid, i, id] += 1
            inhand = inhand.view(16, 27)
            cards_indeck = torch.zeros(4, 27)
            for id, nums in enumerate(self.cards_indeck):
                for i in range(nums.int()):
                    cards_indeck[i, id] += 1
        else:
            inhand = torch.zeros(16, 27)
            cards_indeck = torch.zeros(4, 27)
        feature_unseen = torch.cat([inhand, cards_indeck], dim=0)
        feature = torch.cat([feature_base, feature_discrete, feature_unseen], dim=0)
        return feature
