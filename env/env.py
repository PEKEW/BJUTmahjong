import logging
import time
from .history import History
import copy
import torch
import numpy as np

from .util.utils import random_choice
from .utils import clc_xiangting, id2card, _set, random_draw, clc_yxjz


# NOTE 本接口为基本玩家接口 必须重构的函数为
# NOTE 1.action_after_draw获得牌后进行的操作,当下必须得计算相听
# NOTE 2.get_play_feature获取判断出牌与是否立直的特征
# NOTE 3.get_judge_feature获取判断是否吃碰杠赢牌的特征
# NOTE 4.get_play_p 根据特征返回出牌的概率 (27,)
# NOTE 5.if_***根据特征判断是否立直碰杠赢牌
class players:
    def __init__(self, id):
        # NOTE 抓牌放到了game里 因为card_indeck也在game里
        self.hand = torch.zeros((27,))
        self.inhand = torch.zeros((27,))
        self.id = id
        self.chi = []
        self.peng = []
        self.gang = []
        self.xt = 8
        self.ef_draw = []
        self.num_ef_cards = 0
        self.reward_memory = []
        self.last_draw = -1
        self.last_reward = 0.
        self.last_chi = {"id": None, "target": None}

    def reset(self):
        self.hand = torch.zeros((27,))
        self.inhand = torch.zeros((27,))
        self.chi = []
        self.peng = []
        self.gang = []
        self.xt = 8
        self.ef_draw = []
        self.num_ef_cards = 0
        self.reward_memory = []
        self.last_draw = -1
        self.last_reward = 0.
        self.last_chi = {"id": None, "target": None}

    def action_after_draw(self, history, draw=False):
        minus = len(self.peng) * 2 + len(self.gang) * 2 + len(self.chi) * 2
        self.xt = clc_xiangting(self.inhand, minus)
        # card_available = torch.ones(27) * 4 - self.inhand - (history.chi + history.peng * 3 + history.bugang +
        #                                                      (history.angang + history.zhigang) * 4 + history.discard).sum(0)
        # 太慢了 受不了
        # self.xt, rec, qidui = clc_xiangting(self.inhand, minus, getoptim=True)
        # if not draw:
        #     self.ef_draw, self.num_ef_cards = clc_yxjz(rec, minus, self.inhand, card_available)

    # NOTE 8维 或 6维(当14张牌时,知道哪些牌进相听没有意义)
    def get_priv_feature(self, play=False):
        inhand = torch.zeros(4, 27)
        for id, nums in enumerate(self.inhand):
            for i in range(nums.int()):
                inhand[i, id] += 1
        xt = torch.tensor(self.xt).view(1, 1).expand(1, 27)
        uid = torch.tensor(self.id).view(1, 1).expand(1, 27)
        # 太慢不要了
        # if not play:
        #     ef_draw = torch.zeros(1, 27)
        #     for id in self.ef_draw:
        #         ef_draw[0, id] += 1
        #     num_ef_draw = torch.tensor(self.num_ef_cards).view(1, 1).expand(1, 27)
        #     feature_private = torch.cat([inhand, xt, uid, ef_draw, num_ef_draw], dim=0)
        # else:
        feature_private = torch.cat([inhand, xt, uid], dim=0)
        return feature_private

    # NOTE 403
    def get_feature(self, history, play=False, know_all=False):
        features = history.get_global_feature(know_all)
        feature_private = self.get_priv_feature(play)
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

    # TODO 之前有bug现在改好了 输入是某张牌 输出为可行的吃法
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

    # TODO 检测立直是否合法 id为牌的id 如果id为None则只检验相听是否为0 check表示是否重新计算相听 如果之前计算过就不用重新算了
    def richi_vaild_check(self, id=None, check=False):
        cards = self.inhand
        minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
        if id is not None:
            vaild = self.play_vaild_check(id)
            if vaild:
                tmp = copy.deepcopy(cards)
                tmp[id] -= 1
                xt = clc_xiangting(tmp, minus)
                if xt == 0:
                    return True
            return False
        else:
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

    # TODO 检查是否能赢 如果id为入手的牌(或别人打出的牌) 为None时只检验想听 check同上面的
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

    # TODO id为None表示是自己摸的
    def gang_vaild_check(self, id=None):
        cards = self.inhand
        if id is not None:
            if cards[id] >= 3:
                return True
        else:
            angang_ids = torch.arange(0, 27)[(self.inhand == 4)].tolist()
            bugang_ids = torch.arange(0, 27)[(self.hand == 4) & (self.inhand == 1)].tolist()
            return angang_ids, bugang_ids

    def play(self, p=None):
        if p is None:
            p = [1 / 27, ] * 27
        vaild = False
        while not vaild:
            id = np.random.choice(range(27), size=1, p=p.detach().cpu().numpy())
            vaild = self.play_vaild_check(id)
        _set(id, 0, self.inhand)
        _set(id, 0, self.hand)
        # minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
        # self.xt = clc_xiangting(self.inhand, minus)
        return id

    def _peng(self, id):
        self.peng.append(id)
        ids = [id, ] * 2
        _set(ids, 0, self.inhand)
        _set(id, 1, self.hand)
        minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
        self.xt = clc_xiangting(self.inhand, minus)

    def _chi(self, id, ids):
        self.chi.append(ids)
        tmp = copy.deepcopy(ids)
        tmp.remove(id)
        _set(tmp, 0, self.inhand)
        _set(id, 1, self.hand)
        # minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
        # self.xt = clc_xiangting(self.inhand, minus)

    # TODO 这里把杠后的摸排弃牌去了 因为杠后相当于重新进行本轮 直接回到本轮开始就好了
    def _gang(self, id, type):
        if type == "bugang":
            self.peng.remove(id)
            self.gang.append(id)
            _set(id, 0, self.inhand)
            # minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
            # self.xt = clc_xiangting(self.inhand, minus)
            return 3

        elif type == "angang":
            ids = [id, ] * 4
            _set(ids, 0, self.inhand)
            self.gang.append(id)
            # minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
            # self.xt = clc_xiangting(self.inhand, minus)
            return 9
        else:
            self.gang.append(id)
            ids = [id, ] * 3
            _set(ids, 0, self.inhand)
            _set(id, 1, self.hand)
            # minus = len(self.chi) * 2 + len(self.peng) * 2 + len(self.gang) * 2
            # self.xt = clc_xiangting(self.inhand, minus)
            return 4

    # TODO 判断赢牌的分数 这里应该加个自摸的 还没写
    def _win(self):
        num_dui = (self.inhand == 2).sum()
        num_ke = (self.inhand == 3).sum() + len(self.peng) + len(self.gang)
        num_tong = self.hand[:9].sum()
        num_tiao = self.hand[9:18].sum()
        num_wan = self.hand[18:].sum()
        if num_dui == 7:
            return 12
        if num_tiao == 14 + len(self.gang) or num_tong == 14 + len(self.gang) or num_wan == 14 + len(self.gang):
            return 12
        if num_ke == 4:
            return 8
        return 6


'''
这里为游戏本体
card_indeck表示牌库中剩余牌
zhuang表示庄家id
history为历史信息(没用到)
num_matches表示游戏场数
score记录4人分数
richi记录四人立直状态
'''

# TODO 1.写一个历史记录每一轮中所有玩家动作 并能够打印出来
# TODO 2.本接口的if_xxx 都应当输入特征 得让他们适配
# TODO 3.特征尚未定义
# TODO 4.需要把可以获取的信息都保存到History这样才方便提取特征

logging.basicConfig(filename='rf_log.txt',
                    filemode='w', )


# NOTE 1.num_matches总共游戏局数 2.players:四名玩家 3.savepath保存路径
class game:
    def __init__(self, num_matches, savepath):
        self.card_indeck = torch.ones((27, 4))
        self.card_indeck_perm = np.arange(0, 108).tolist()
        np.random.shuffle(self.card_indeck_perm)
        self.zhuang = -1
        self.players = [players(0), players(1), players(2), players(3)]
        self.num_matches = num_matches
        self.richi = [False, ] * 4
        self.savepath = savepath
        self.seed = int(time.time())
        # self.seed = 0
        self.history = History(self.zhuang)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.target = "draw"
        self.now = 0
        self.tmp_uid = 0  # 用来临时保存当前吃碰杠玩家
        self.tmp_id = -1  # 用来临时保存最近出的一张牌
        self.last_match_score = torch.zeros(4)
        self.n = 4

    # def get_env_features(self, know_all):
    #     global_feature = self.history.get_global_feature(know_all=True)[None, :, :]
    #     return global_feature
    #
    #
    # # 返回的是 (1, cr, channels, 27) cr回合数 用于critic评分网络判断当下奖励
    # # NOTE 共402维
    # def get_global_features(self, uid, know_all):
    #     now_env_feature = self.get_env_features(know_all)
    #     now_priv_feature = self.players[uid].get_priv_feature()
    #     global_feature = torch.cat((now_env_feature, now_priv_feature), dim=1)[:, None, :, :]
    #     return global_feature

    def get_experience(self):
        experience = {
            "card_indeck": self.card_indeck,
            "card_indeck_perm": self.card_indeck_perm,
            "zhuang": self.zhuang,
            "players": copy.deepcopy(self.players),
            "num_matches": self.num_matches,
            "richi": self.richi,
            "seed": self.seed,
            "history": copy.deepcopy(self.history),
            "target": self.target,
            "now": self.now,
            "tmp_uid": self.tmp_uid,
            "tmp_id ": self.tmp_id,
            # "global_feature": self.global_feature,
            "last_match_score": self.last_match_score
        }
        return experience

    def replay_experience(self, experience):
        self.card_indeck = experience["card_indeck"]
        self.card_indeck_perm = experience["card_indeck_perm"]
        self.zhuang = experience["zhuang"]
        self.players = experience["players"]
        self.num_matches = experience["num_matches"]
        self.richi = experience["richhi"]
        self.seed = experience["seed"]
        self.history = experience["history"]
        self.target = experience["target"]
        self.now = experience["now"]
        self.tmp_uid = experience["tmp_uid"]
        self.tmp_id = experience["tmp_id"]
        self.global_feature = experience["global_feature"]

    # NOTE 表示重置本局
    def reset_match(self):
        self.zhuang = (self.zhuang + 1) % 4
        self.card_indeck = torch.ones((27, 4))
        # FIXED
        self.card_indeck_perm = np.arange(0, 108).tolist()
        np.random.shuffle(self.card_indeck_perm)
        self.history.reset_match(self.zhuang)
        self.richi = [False, ] * 4
        self.target = "draw"
        self.now = self.zhuang
        self.tmp_uid = 0
        self.tmp_id = -1
        for id, player in enumerate(self.players):
            player.reset()
            self.card_indeck, ids = random_draw(self.card_indeck, 13, self.card_indeck_perm)
            _set(ids, 1, player.hand)
            _set(ids, 1, player.inhand)
            player.action_after_draw(self.history)
            message = "玩家{:}手牌:".format(id) + id2card(ids, zhangshu=True)
            self.logger.info(message)
        for uid, player in enumerate(self.players):
            self.history.inhand[uid, :] = player.inhand
        self.history.cards_indeck = self.card_indeck.sum(1)
        self.last_match_score = self.history.score

    # NOTE 表示重置游戏
    def reset(self):
        self.zhuang = np.random.randint(0, 4)
        self.history.reset_game()
        self.richi = [False, ] * 4
        self.card_indeck = torch.ones((27, 4))
        self.card_indeck_perm = np.arange(0, 108).tolist()
        self.target = "draw"
        self.now = self.zhuang
        self.tmp_uid = 0
        self.tmp_id = -1
        np.random.shuffle(self.card_indeck_perm)
        for id, player in enumerate(self.players):
            player.reset()
            self.card_indeck, ids = random_draw(self.card_indeck, 13, self.card_indeck_perm)
            _set(ids, 1, player.hand)
            _set(ids, 1, player.inhand)
            player.action_after_draw(self.history)
            message = "玩家{:}手牌:".format(id) + id2card(ids, zhangshu=True)
            self.logger.info(message)
        for uid, player in enumerate(self.players):
            self.history.inhand[uid, :] = player.inhand
        self.history.cards_indeck = self.card_indeck.sum(1)
        self.seed = int(time.time())
        self.last_match_score = torch.zeros(4)

    # NOTE 表示摸牌的过程
    def _draw(self, uid, num_cards=1):
        player = self.players[uid]
        card_left, ids = random_draw(self.card_indeck, num_cards, self.card_indeck_perm)
        _set(ids, 1, player.inhand)
        _set(ids, 1, player.hand)
        player.last_draw = ids
        player.action_after_draw(self.history, draw=True)
        message = "{:}号玩家,抓牌".format(uid) + id2card(ids)
        self.logger.info(message)
        self.history.num_cards_left -= 1
        self.history.cards_indeck[ids] -= 1
        return ids

    def _win_vaild_check(self, uid, id=None):
        return self.players[uid].win_vaild_check(check=False, id=id)

    def _win(self, uid, target, type="zimo"):
        player = self.players[uid]
        score = player._win()
        if type == "zimo":
            for i in range(3):
                _uid = (uid + i + 1) % 4
                self.history.score[_uid] -= score
            self.history.score[uid] += score * 3
        else:
            self.history.score[target] -= score
            self.history.score[uid] += score
        message = "{:}号玩家,赢,回合结束".format(uid)
        self.logger.info(message)
        return score

    def _richi_vaild_check(self, uid):
        if self.richi[uid]:
            return None
        return self.players[uid].richi_vaild_check()

    def _richi(self, uid, p=None, play=None):
        player = self.players[uid]
        assert p is not None or play is not None
        if play is not None:
            assert player.inhand[play] > 0
            cards = player.inhand
            minus = len(player.chi) * 2 + len(player.peng) * 2 + len(player.gang) * 2
            tmp = copy.deepcopy(cards)
            tmp[play] -= 1
            xt = clc_xiangting(tmp, minus)
            assert xt == 0
            p = torch.zeros((27,))
            p[play] = 1.0
        score = 3
        self.history.score[uid] += score
        for i in range(3):
            _uid = (uid + i + 1) % 4
            self.history.score[_uid] -= 1
        self.richi[uid] = True
        self.history.richi[uid] = True
        message = "{:}号玩家,立直".format(uid)
        self.logger.info(message)
        id = player.play(p)
        message = "{:}号玩家,出牌".format(uid) + id2card(id)
        self.history.discard[uid, id] += 1
        self.history.discard_timeline[uid, self.history.cr, id] = 1
        self.logger.info(message)
        return id

    def _gang_vaild_check(self, uid, id=None):
        return self.players[uid].gang_vaild_check(id)

    # NOTE 杠之前还要检测其他3名玩家是否能赢牌
    def _gang(self, uid, target, id, type):
        player = self.players[uid]
        if type == "zhigang":
            score = player._gang(id, "zhigang")
            self.history.score[uid] += score
            self.history.zhigang[uid, id] += 1
            self.history.score[target] -= score
            message = "{:}号玩家,杠".format(uid) + id2card(id)
            self.logger.info(message)
        elif type == "bugang":
            score = player._gang(id, "bugang")
            self.history.score[uid] += score
            self.history.bugang[uid, id] += 1
            for i in range(3):
                _uid = (uid + i + 1) % 4
                self.history.score[uid] -= 1

            message = "{:}号玩家,杠".format(uid) + id2card(id)
            self.logger.info(message)
        elif type == "angang":
            score = player._gang(id, "angang")
            self.history.angang[uid, id] += 1
            message = "{:}号玩家,杠".format(uid) + id2card(id)
            self.logger.info(message)
            self.history.score[uid] += score
            for i in range(3):
                _uid = (uid + i + 1) % 4
                self.history.score[uid] -= 3
        else:
            raise TypeError()
        player.action_after_draw(self.history, draw=True)
        return score

    def _peng_vaild_check(self, uid, id=None):
        if self.richi[uid]:
            return False
        return self.players[uid].peng_vaild_check(id)

    def _peng(self, uid, target, id):
        player = self.players[uid]
        player._peng(id)
        self.history.peng[uid, id] = 1
        message = "{:}号玩家,碰".format(uid) + id2card(id)
        self.logger.info(message)
        self.history.score[uid] += 2
        self.history.score[target] -= 2
        player.action_after_draw(self.history, draw=True)

    # NOTE 这里返回的是列表 不是bool 因而需要对列表中所有目标判断是否吃
    def _chi_vaild_check(self, uid, id=None):
        if self.richi[uid]:
            return []
        return self.players[uid].chi_vaild_check(id)

    def _chi(self, uid, id, target, ids):
        player = self.players[uid]
        player._chi(id, ids)
        for id in ids:
            self.history.chi[uid, id] += 1
        message = "{:}号玩家,吃".format(uid) + id2card(ids[0]) + id2card(ids[1]) + id2card(ids[2])
        self.history.score[uid] += 1
        self.history.score[target] -= 1
        self.players[uid].last_chi["id"] = id
        self.players[uid].last_chi["target"] = target
        self.logger.info(message)
        player.action_after_draw(self.history, draw=True)

    def _play(self, uid, p=None, play=None):
        player = self.players[uid]
        assert p is not None or play is not None
        if play is not None:
            assert player.inhand[play] > 0
            p = torch.zeros((27,))
            p[play] = 1.0
        id = player.play(p)
        if self.players[uid].last_chi["id"] is not None:
            if self.players[uid].last_chi["id"] == id:
                self.history.score[uid] -= 1
                self.history.score[self.players[uid].last_chi["target"]] += 1
            else:
                self.last_chi = {"id": None, "target": None}
        self.history.discard[uid, id] += 1
        self.history.discard_timeline[uid, self.history.cr, id] = 1
        message = "{:}号玩家,出牌".format(uid) + id2card(id)
        self.logger.info(message)
        self.history.inhand[uid, :] = self.players[uid].inhand
        player.action_after_draw(self.history, draw=False)
        return id

    # def save(self, matches, turns):
    #     path = os.path.join(self.savepath, "{:}".format(matches))
    #     if not os.path.exists(path):
    #         os.mkdir(path)
    #     path = os.path.join(self.savepath, "{:}".format(matches), 'checkpt-turns-{:}.pth'.format(turns))
    #     torch.save({
    #         'zhuang': self.zhuang,
    #         'player1': self.players[0],
    #         'player2': self.players[1],
    #         'player3': self.players[2],
    #         'player4': self.players[3],
    #         'score': self.score,
    #         'richi': self.richi,
    #         'num_matches': self.num_matches,
    #         'card_indeck': self.card_indeck,
    #         'seed': self.seed,
    #         'history': self.history
    #     }, path)
    #
    # def resume(self, matches, turns):
    #     path = os.path.join(self.savepath, "{:}".format(matches), 'checkpt-turns:{:}.pth'.format(turns))
    #     checkpt = torch.load(path)
    #     self.zhuang = checkpt['zhuang']
    #     self.players = [checkpt['player1'], checkpt['player2'], checkpt['player3'], checkpt['player4']]
    #     self.score = checkpt['score']
    #     self.richi = checkpt['richi']
    #     self.num_matches = checkpt['num_matches']
    #     self.card_indeck = checkpt['card_indeck']
    #     self.seed = checkpt['seed']
    #     self.history = checkpt['history']
    #     random.seed(self.seed)
    #     np.random.seed(self.seed)
    #     torch.random.manual_seed(self.seed)

    # FIXME 问题在于每一次只能处理一个样本 不能并行处理
    def step(self, input=None, experience=None, know_all=False, critic_know_all=True, keep_xiangting=False):
        if experience is not None:
            self.replay_experience(experience)
        while True:
            if self.target == 'draw':
                if len(self.card_indeck_perm) == 0:
                    self.logger.info("平局")
                    return {"target": "draw"}
                id = self._draw(self.now)
                self.target = 'win'

            if self.target == 'win':
                if input is None:
                    if self._win_vaild_check(self.now):
                        feature = self.players[self.now].get_feature(self.history, know_all=know_all)
                        object = {"target": "win", "uid": self.now, "feature": feature}
                        return object
                else:
                    global_feature = self.players[self.now].get_feature(self.history, know_all=critic_know_all)
                    flag = True
                    if flag:
                        score = self._win(self.now, target=None)
                        reward = score
                        # self.reset_match()
                    else:
                        self.target = "gang"
                        reward = 0
                    self.players[self.now].reward_memory.append(reward)
                    # TODO 补全 问题是这个动作的评分是判断了就给 还是说只有做动作才给
                    experience = {"target": "win", "experience": self.get_experience(), "act": input, "reward": reward,
                                  "uid": self.now, "terminal": flag, 'global_feature': global_feature}
                    return experience, global_feature

                self.target = "gang"
            # Fixme 头疼 补杠还能抢杠 那么补杠的一个reward反而取决于之后几个人是否胡牌 胡牌reward为0 不胡为3 好恶心
            if self.target == 'gang':
                terminal = False
                if input is None:
                    angang_ids, bugang_ids = self._gang_vaild_check(self.now)
                    if len(angang_ids) or len(bugang_ids) != 0:
                        feature = self.players[self.now].get_feature(self.history, know_all=know_all)
                        object = {"target": "gang", "angang_ids": angang_ids, "bugang_ids": bugang_ids, "uid": self.now,
                                  "feature": feature}
                        return object
                    else:
                        self.target = "richi"
                else:
                    # input["id"]应该为None或者为id
                    id = input["id"]
                    global_feature = self.players[self.now].get_feature(self.history, know_all=critic_know_all)
                    if id is not None:
                        score = self._gang(self.now, id=id, type=input["type"], target=None)
                        reward = score
                        if input["type"] == "angang":
                            if len(self.card_indeck_perm) == 0:
                                self.logger.info("平局")
                                terminal = True
                            self.target = "draw"
                        else:
                            self.tmp_id = id
                            self.target = "qianggang"
                    else:
                        reward = 0
                        self.target = "richi"
                    # TODO 补全
                    self.players[self.now].reward_memory.append(reward)
                    experience = {"target": "gang", "experience": self.get_experience(), "act": input, "reward": reward,
                                  "uid": self.now,
                                  "terminal": terminal, 'global_feature': global_feature}
                    return experience, global_feature

            if self.target == 'qianggang':
                for _ in range(self.tmp_uid, 3):
                    now = (self.now + self.tmp_uid + 1) % 4
                    if input is None:
                        if self._win_vaild_check(now, self.tmp_id):
                            feature = self.players[now].get_feature(self.history, know_all=know_all)
                            object = {"target": "qianggang", "id": self.tmp_id, "uid": now, "feature": feature}
                            return object
                    else:
                        flag = input["flag"]
                        global_feature = self.players[self.now].get_feature(self.history, know_all=critic_know_all)
                        if flag:
                            score = self._win(now, target=self.now)
                            self.history.score[self.now] -= 3
                            for i in range(3):
                                self.history.score[(self.now + 1 + i) % 4] += 1
                            self.history.bugang[self.now, self.tmp_id] = 0
                            # self.reset_match()
                            reward = score
                        # TODO 补全
                        else:
                            reward = 0
                            self.target = "draw"
                        self.players[now].reward_memory.append(reward)
                        experience = {"target": "others_win", "experience": self.get_experience(), "act": input,
                                      "reward": reward, "uid": now, "terminal": flag, 'global_feature': global_feature}
                        return experience, global_feature
                    self.tmp_uid += 1
                self.target = "draw"
                self.tmp_uid = 0

            if self.target == "richi":
                if input is None:
                    mask = self._richi_vaild_check(self.now)
                    if mask is not None:
                        feature = self.players[self.now].get_feature(self.history, play=True, know_all=know_all)
                        object = {"target": "richi", "mask": mask, "uid": self.now, "feature": feature}
                        return object
                    else:
                        self.target = "play"
                else:
                    flag = input["flag"]
                    global_feature = self.players[self.now].get_feature(self.history, know_all=critic_know_all)
                    if flag:
                        id = self._richi(self.now, play=input["play"])
                        self.tmp_id = id
                        reward = 3
                        self.target = "others_win"
                    else:
                        reward = 0
                        self.target = "play"
                    # TODO 补全
                    self.players[self.now].reward_memory.append(reward)
                    experience = {"target": "richi", "experience": self.get_experience(), "act": input,
                                  "reward": reward, "uid": self.now, "terminal": False,
                                  'global_feature': global_feature}
                    return experience, global_feature

            if self.target == 'play':
                if self.richi[self.now]:
                    id = self._play(uid=self.now, p=None, play=self.players[self.now].last_draw)
                    self.tmp_id = id
                    self.target = "others_win"
                elif input is None:
                    feature = self.players[self.now].get_feature(self.history, play=True, know_all=know_all)
                    if self.history.discard[self.now, :].sum() == 0 or (
                            keep_xiangting and not any(self.history.richi)):  # 这么写是因为无法训练第一步弃牌
                        mask = torch.zeros(27, )
                        ids = torch.arange(27)[self.players[self.now].inhand > 0].tolist()
                        minus = len(self.players[self.now].chi) * 2 + len(self.players[self.now].peng) * 2 + len(
                            self.players[self.now].gang) * 2
                        min_xt = self.players[self.now].xt
                        for id in ids:
                            tmp = copy.deepcopy(self.players[self.now].inhand)
                            tmp[id] -= 1
                            xt = clc_xiangting(tmp, minus)
                            if xt == min_xt:
                                mask[id] = 1.
                        assert mask.sum() > 0
                    else:
                        mask = (self.players[self.now].inhand > 0).int()
                    object = {"target": "play", "mask": mask, "uid": self.now, "feature": feature,
                              "inhand": self.players[self.now].inhand, "richi": self.history.richi}
                    return object
                else:
                    play = input["play"]
                    global_feature = self.players[self.now].get_feature(self.history, know_all=critic_know_all)
                    self.history.cr = self.history.cr + 1
                    # for i in range(4):
                    #     self.players[i].update_priv_feature()
                    # self.update_env_features()

                    xt = self.players[self.now].xt
                    xt_feature = torch.zeros(4, 27)
                    for id, nums in enumerate(self.players[self.now].inhand):
                        for i in range(nums.int()):
                            xt_feature[i, id] += 1

                    id = self._play(uid=self.now, play=play)

                    _xt = self.players[self.now].xt
                    _xt_feature = torch.zeros((1, 27))
                    _xt_feature[0, play] = 1
                    xt_feature = torch.cat((xt_feature, _xt_feature), dim=0)
                    xt_ps = _xt - xt

                    self.tmp_id = id
                    # TODO 补全
                    self.target = "others_win"
                    reward = 0
                    self.players[self.now].reward_memory.append(reward)
                    experience = {"target": "play", "experience": self.get_experience(), "act": input, "reward": reward,
                                  "uid": self.now,
                                  "xt_feature": {"feature": xt_feature, "target": xt_ps}, "terminal": False,
                                  'global_feature': global_feature}
                    return experience, global_feature

            if self.target == 'others_win':
                for _ in range(self.tmp_uid, 3):
                    now = (self.now + self.tmp_uid + 1) % 4
                    if input is None:
                        if self._win_vaild_check(now, self.tmp_id):
                            feature = self.players[now].get_feature(self.history, know_all=know_all)
                            object = {"target": "others_win", "id": self.tmp_id, "uid": now, "feature": feature}
                            return object
                    else:
                        global_feature = self.players[self.now].get_feature(self.history, know_all=critic_know_all)
                        flag = input["flag"]
                        if flag:
                            score = self._win(now, target=self.now, type="dianpao")
                            # self.reset_match()
                            reward = score
                            self.tmp_uid = 0
                        # TODO 补全
                        else:
                            reward = 0
                            self.tmp_uid += 1
                        self.players[now].reward_memory.append(reward)
                        experience = {"target": "others_win", "experience": self.get_experience(), "act": input,
                                      "reward": reward, "uid": now,
                                      "terminal": flag, 'global_feature': global_feature}
                        return experience, global_feature
                    self.tmp_uid += 1
                self.target = "others_gang"
                self.tmp_uid = 0

            if self.target == 'others_gang':
                terminal = False
                for _ in range(self.tmp_uid, 3):
                    now = (self.now + self.tmp_uid + 1) % 4
                    if input is None:
                        flag = self._gang_vaild_check(now, self.tmp_id)
                        feature = self.players[now].get_feature(self.history, know_all=know_all)
                        if flag is not None:
                            object = {"target": "others_gang", "id": self.tmp_id, "uid": now, "feature": feature}
                            return object
                    else:
                        flag = input["flag"]
                        global_feature = self.players[self.now].get_feature(self.history, know_all=critic_know_all)
                        if flag:
                            xt = self.players[now].xt
                            score = self._gang(now, id=self.tmp_id, type="zhigang", target=self.now)
                            self.target = "draw"
                            self.now = now
                            self.history.now = self.now
                            _xt = self.players[now].xt
                            reward = score
                            # reward = score + (xt - _xt) * 1.5
                            if len(self.card_indeck_perm) == 0:
                                self.logger.info("平局")
                                terminal = True
                            self.tmp_uid = 0
                        else:
                            reward = 0
                            self.tmp_uid += 1
                        self.players[now].reward_memory.append(reward)
                        experience = {"target": "others_gang", "experience": self.get_experience(), "act": input,
                                      "reward": reward,
                                      "uid": now, "terminal": terminal, 'global_feature': global_feature}
                        return experience, global_feature
                    self.tmp_uid += 1
                self.tmp_uid = 0
                self.target = "others_peng"

            if self.target == 'others_peng':
                terminal = False
                for _ in range(self.tmp_uid, 3):
                    now = (self.now + self.tmp_uid + 1) % 4
                    if input is None:
                        if self._peng_vaild_check(now, self.tmp_id):
                            feature = self.players[now].get_feature(self.history, know_all=know_all)
                            object = {"target": "others_peng", "id": self.tmp_id, "uid": now, "feature": feature}
                            return object
                        else:
                            self.tmp_uid += 1
                    else:
                        flag = input["flag"]
                        global_feature = self.players[self.now].get_feature(self.history, know_all=critic_know_all)
                        if flag:
                            xt = self.players[now].xt
                            self._peng(now, id=self.tmp_id, target=self.now)
                            self.target = "play"
                            self.now = now
                            self.history.now = self.now
                            _xt = self.players[now].xt
                            # reward = 2 + (xt - _xt) * 1.5
                            reward = 2
                            if len(self.card_indeck_perm) == 0:
                                self.logger.info("平局")
                                terminal = True
                            self.tmp_uid = 0
                        else:
                            self.tmp_uid += 1
                            reward = 0
                        self.players[now].reward_memory.append(reward)
                        experience = {"target": "others_peng", "experience": self.get_experience(), "act": input,
                                      "reward": reward, "uid": now,
                                      "terminal": terminal, 'global_feature': global_feature}
                        return experience, global_feature
                self.tmp_uid = 0
                self.target = "others_chi"

            if self.target == "others_chi":
                now = (self.now + 1) % 4
                terminal = False
                if input is None:
                    idlist = self._chi_vaild_check(now, self.tmp_id)
                    if len(idlist) != 0:
                        feature = self.players[now].get_feature(self.history, know_all=know_all)
                        object = {"target": "others_chi", "idlist": idlist, "uid": now, "feature": feature}
                        return object
                    else:
                        self.target = "draw"
                        self.now = (self.now + 1) % 4
                        self.history.now = self.now
                else:
                    ids = input["ids"]
                    global_feature = self.players[self.now].get_feature(self.history, know_all=critic_know_all)
                    if ids is not None:
                        xt = self.players[now].xt
                        self._chi(uid=now, id=self.tmp_id, target=self.now, ids=ids)
                        self.now = now
                        self.history.now = self.now
                        self.target = "play"
                        _xt = self.players[now].xt
                        # reward = 1 + (xt - _xt) * 1.5
                        reward = 1
                    else:
                        self.target = "draw"
                        self.now = (self.now + 1) % 4
                        self.history.now = self.now
                        reward = 0
                    self.players[now].reward_memory.append(reward)
                    if len(self.card_indeck_perm) == 0:
                        self.logger.info("平局")
                        terminal = True
                    experience = {"target": "others_chi", "experience": self.get_experience(), "act": input,
                                  "reward": reward, "uid": now, "terminal": terminal, 'global_feature': global_feature}
                    return experience, global_feature
