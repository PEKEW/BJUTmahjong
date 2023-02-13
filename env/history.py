import torch

# NOTE 这里的信息都应该是公开的
# NOTE 通过history里的信息来获得特征
class History:
    def __init__(self, zhuang_id):
        self.discard = torch.zeros((4, 27))  # 已经打出的牌
        self.chi = torch.zeros((4, 27))
        self.peng = torch.zeros((4, 27))
        self.bugang = torch.zeros((4, 27))
        self.angang = torch.zeros((4, 27))
        self.zhigang = torch.zeros((4, 27))
        self.cr = 0  # 回合数
        self.matches = -1  # 局数
        self.now = 0  # 当前玩家
        self.richi = [False, ] * 4
        self.zhuang_id = zhuang_id
        self.discard_timeline = torch.zeros(4, 80, 27)  # 每一轮都有一名玩家弃1张牌 最大假设100轮
        self.score = torch.zeros(4)
        self.num_cards_left = 56
        # feature unseen
        self.inhand = torch.zeros((4, 27))
        self.cards_indeck = torch.zeros((27,))
        self.experience = []
    def reset_match(self, zhuang_id):
        self.discard = torch.zeros((4, 27))  # 已经打出的牌
        self.chi = torch.zeros((4, 27))
        self.peng = torch.zeros((4, 27))
        self.bugang = torch.zeros((4, 27))
        self.angang = torch.zeros((4, 27))
        self.zhigang = torch.zeros((4, 27))
        self.richi = [False, ] * 4
        self.zhuang_id = zhuang_id
        self.discard_timeline = torch.zeros(4, 80, 27)
        self.cr = 0
        self.now = self.zhuang_id
        self.matches += 1
        self.action_rec = []
        self.score = torch.zeros(4)
        self.num_cards_left = 56
        self.inhand = torch.zeros((4, 27))
        self.cards_indeck = torch.zeros((27,))

    def reset_game(self):
        self.reset_match(0)
        self.matches = 0

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
