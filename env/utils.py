from torch import nn
import torch
from torch import Tensor
import numpy as np
import copy


def idx_to_list(idx):
    return torch.IntTensor(list(range(len(idx))))[idx].tolist()


# NOTE 适用于1维手牌或2维手牌 手牌改为了(27)而不是(27,4)了 val为0或1, 0表示弃牌,1表示摸牌
def _set(ids, val, vec):
    if not isinstance(ids, Tensor):
        ids = torch.squeeze(torch.tensor(ids).int())
    else:
        ids = torch.squeeze(ids.int())

    if len(vec.shape) == 2:
        if len(ids.shape) == 0:
            for i in range(4):
                if vec[ids, i] != val:
                    vec[ids, i] = val
                    assert vec[ids, i] in [0, 1]
                    break
        else:
            for id in ids:
                for i in range(4):
                    if vec[id, i] != val:
                        vec[id, i] = val
                        assert vec[id, i] in [0, 1]
                        break
    else:
        if len(ids.shape) == 0:
            if val == 0:
                vec[ids] -= 1
                assert vec[ids] >= 0
            else:
                vec[ids] += 1
                assert vec[ids] <= 4
        else:
            for id in ids:
                if val == 0:
                    vec[id] -= 1
                    assert vec[id] >= 0
                else:
                    vec[id] += 1
                    assert vec[id] <= 4


# TODO cards表示牌库 play表示抓到的牌
def random_draw(cards: Tensor, num_cards, perm=None):
    shape = cards.shape
    cards = cards.view(-1, 1)
    indicate = (cards.view(-1) == 1)
    id = torch.tensor(list(range(len(cards)))).view(-1, 1)
    cards_with_id = torch.cat((cards, id), dim=1)
    left_cards = cards_with_id[indicate]
    if perm is None:
        select = np.random.choice(list(range(len(left_cards))), size=num_cards, replace=False)
        id = left_cards[select, 1].numpy()
    else:
        select = []
        for _ in range(num_cards):
            select.append(perm[0])
            perm.pop(0)
        id = np.array(select)
    cards[id, 0] = 0
    play = (id // 4).astype(int)
    return cards.view(*shape), play


# TODO 下面的都是计算相听/有效进牌与最优出牌
# TODO 可能的话改善下计算相听的效率 因为立直与胡牌的判断都需要计算相听
def get_kezi(cards):
    idx = (cards >= 3)
    ids = torch.tensor(range(len(cards)))[idx].numpy() if idx.sum() else []
    return ids


def get_shunzi(input):
    ids = []
    cards = copy.deepcopy(input)
    for i in range(1, 8):
        for n in range(4):
            if cards[i - 1] > n and cards[i] > n and cards[i + 1] > n:
                ids.append(i)
            if cards[i - 1 + 9] > n and cards[i + 9] > n and cards[i + 1 + 9] > n:
                ids.append(i + 9)
            if cards[i - 1 + 18] > n and cards[i + 18] > n and cards[i + 1 + 18] > n:
                ids.append(i + 18)
    return ids


def get_dazi(cards):
    idx = (cards == 2)
    dui = torch.tensor(range(len(cards)))[idx].numpy() if idx.sum() else []
    da = []
    for i in range(8):
        if cards[i] > 0:
            if cards[i + 1] > 0:
                da.append((i, i + 1))
            elif i != 7 and cards[i + 2] > 0:
                da.append((i, i + 2))
        if cards[i + 9] > 0:
            if cards[i + 9 + 1] > 0:
                da.append((i + 9, i + 9 + 1))
            elif i != 7 and cards[i + 9 + 2] > 0:
                da.append((i + 9, i + 9 + 2))

        if cards[i + 18] > 0:
            if cards[i + 18 + 1] > 0:
                da.append((i + 18, i + 18 + 1))
            elif i != 7 and cards[i + 18 + 2] > 0:
                da.append((i + 18, i + 18 + 2))
    return dui, da


'''
计算相听并返回最佳相听对应牌型
minus 为已经吃碰杠的数量*2
'''


def get_optim_xiangting(cards, minus=0, nmianzi=0, nduizi=0, ndazi=0,
                        nkezi=1, nshunzi=1, xiangting=9,
                        rec=None, dict=None):
    if rec is None:
        rec = []
        dict = [[[], xiangting]]
    if nkezi != 0:
        kezi = get_kezi(cards)
        nkezi = len(kezi)
    if nshunzi != 0:
        shunzi = get_shunzi(cards)
        nshunzi = len(shunzi)
    if nkezi != 0 or nshunzi != 0:
        if nkezi:
            for id in kezi:
                tmp = copy.deepcopy(cards)
                tmp[id] -= 3
                _rec = copy.deepcopy(rec)
                _rec.append([id, ] * 3)
                xt, dict = get_optim_xiangting(tmp, minus, nmianzi + 1, nkezi=nkezi - 1, nshunzi=nshunzi,
                                               xiangting=xiangting,
                                               rec=_rec, dict=dict)
                xiangting = min(xiangting, xt)
        if nshunzi:
            for id in shunzi:
                tmp = copy.deepcopy(cards)
                tmp[id - 1] -= 1
                tmp[id] -= 1
                tmp[id + 1] -= 1
                _rec = copy.deepcopy(rec)
                _rec.append([id - 1, id, id + 1])
                xt, dict = get_optim_xiangting(tmp, minus, nmianzi + 1, nkezi=nkezi, nshunzi=nshunzi - 1,
                                               xiangting=xiangting,
                                               rec=_rec, dict=dict)
                xiangting = min(xiangting, xt)

    if nkezi + nshunzi < 2:
        dui, da = get_dazi(cards)
        if len(dui) > 0:
            for id in dui:
                tmp = copy.deepcopy(cards)
                tmp[id] = tmp[id] - 2
                _rec = copy.deepcopy(rec)
                _rec.append([id, id])
                xt, dict = get_optim_xiangting(tmp, minus, nmianzi, nduizi + 1, ndazi, xiangting=xiangting, rec=_rec,
                                               dict=dict)
                xiangting = min(xiangting, xt)

        if len(da) > 0:
            for (a, b) in da:
                tmp = copy.deepcopy(cards)
                tmp[a] = tmp[a] - 1
                tmp[b] = tmp[b] - 1
                _rec = copy.deepcopy(rec)
                _rec.append([a, b])
                xt, dict = get_optim_xiangting(tmp, minus, nmianzi, nduizi, ndazi + 1, xiangting=xiangting, rec=_rec,
                                               dict=dict)
                xiangting = min(xiangting, xt)

        if len(dui) == 0 and len(da) == 0 and nkezi == 0 and nshunzi == 0:
            xt = 8
            nmianzi = nmianzi + minus // 2
            if nmianzi + ndazi + nduizi > 5:
                xt = 4
                if nduizi > 0:
                    xt -= 1
                xt = xt - nmianzi
            else:
                if nmianzi + ndazi + nduizi == 5:
                    if nduizi > 0:
                        xt = 8
                    else:
                        xt = 9
                xt = xt - 2 * nmianzi - ndazi - nduizi
            if xt > xiangting:
                return xiangting, dict
            rec.sort()
            result = [rec, xt]
            flag = False
            if dict[0][1] > xt:
                return xt, [result]
            elif dict[0][1] == xt:
                for i in range(len(dict)):
                    flag = (dict[i] == result) if not flag else flag
                if not flag:
                    dict.append(result)
            return xt, dict
    return xt, dict


'''
计算相听数,getoptim为True则返回最佳组合,否则只返回相听数
返回的rec的格式为[[[面子与对子的组合],相听]*n]
'''


def get_xiangting(input_cards, minus=0):
    xiangting = 6
    max_mianzi = 0
    rec = [[], [], [], [], []]
    terminal = False

    def dfs(cards, nmianzi=0, nduizi=0, ndazi=0, nkezi=1, nshunzi=1):
        # global cards, minus, rec, terminal, max_mianzi, xiangting
        nonlocal max_mianzi, xiangting, rec, terminal
        if nkezi != 0:
            kezi = get_kezi(cards)
            nkezi = len(kezi)
        if nshunzi != 0:
            shunzi = get_shunzi(cards)
            nshunzi = len(shunzi)
        if nkezi != 0 or nshunzi != 0:
            if nkezi:
                for id in kezi:
                    tmp = copy.deepcopy(cards)
                    tmp[id] -= 3
                    flag = True
                    layer = nmianzi + nduizi + ndazi
                    for r in rec[layer]:
                        if all(r[0] == tmp) and nmianzi + 1 == r[1] and nduizi == r[2]:
                            flag = False
                            break
                    if flag:
                        rec[layer].append([tmp, nmianzi + 1, nduizi])  # 保存状态 避免重复
                        dfs(cards=tmp, nmianzi=nmianzi + 1, nkezi=nkezi - 1, nshunzi=nshunzi)
                        if terminal:
                            return
            if nshunzi:
                for id in shunzi:
                    tmp = copy.deepcopy(cards)
                    tmp[id - 1] -= 1
                    tmp[id] -= 1
                    tmp[id + 1] -= 1

                    flag = True
                    layer = nmianzi + nduizi + ndazi
                    for r in rec[layer]:
                        if all(r[0] == tmp) and nmianzi + 1 == r[1] and nduizi == r[2]:
                            flag = False
                            break
                    if flag:
                        rec[layer].append([tmp, nmianzi + 1, nduizi])
                        dfs(cards=tmp, nmianzi=nmianzi + 1, nkezi=nkezi, nshunzi=nshunzi - 1)
                        if terminal:
                            return
        else:
            if max_mianzi > nmianzi:
                return
            if nmianzi + ndazi + nduizi >= 5:
                dui, da = [], []
                if nduizi > 0:
                    terminal = True
            else:
                dui, da = get_dazi(cards)
                if len(dui) > 0:
                    for id in dui:
                        tmp = copy.deepcopy(cards)
                        tmp[id] = tmp[id] - 2

                        flag = True
                        layer = nmianzi + nduizi + ndazi
                        for r in rec[layer]:
                            if all(r[0] == tmp) and nmianzi == r[1] and nduizi + 1 == r[2]:
                                flag = False
                                break
                        if flag:
                            rec[layer].append([tmp, nmianzi, nduizi + 1])
                            dfs(cards=tmp, nmianzi=nmianzi, nduizi=nduizi + 1, ndazi=ndazi, nkezi=0, nshunzi=0)
                            if terminal:
                                return
                    if len(da) > 0:
                        for (a, b) in da:
                            tmp = copy.deepcopy(cards)
                            tmp[a] = tmp[a] - 1
                            tmp[b] = tmp[b] - 1
                            flag = True
                            layer = nmianzi + nduizi + ndazi
                            for r in rec[layer]:
                                if all(r[0] == tmp) and nmianzi == r[1] and nduizi + 1 == r[2]:
                                    flag = False
                                    break
                            if flag:
                                rec[layer].append([tmp, nmianzi, nduizi + 1])
                                dfs(cards=tmp, nmianzi=nmianzi, nduizi=nduizi, ndazi=ndazi + 1, nkezi=0, nshunzi=0)
                                if terminal:
                                    return
            if len(dui) == 0 and len(da) == 0:
                xt = 8
                max_mianzi = max(nmianzi, max_mianzi)
                nmianzi = nmianzi + minus // 2
                if nmianzi + ndazi + nduizi > 5:
                    xt = 4
                    if nduizi > 0:
                        xt -= 1
                    xt = xt - nmianzi
                else:
                    if nmianzi + ndazi + nduizi == 5:
                        if nduizi > 0:
                            xt = 8
                        else:
                            xt = 9
                    xt = xt - 2 * nmianzi - ndazi - nduizi
                xiangting = min(xt, xiangting)

        return
    cards = copy.deepcopy(input_cards)
    dfs(cards)
    return xiangting



def clc_xiangting(cards=None, minus=0, getoptim=False, clc_qidui=True, limit_qidui=1):
    if isinstance(cards, Tensor):
        if len(cards.shape) == 2:
            cards = cards.sum(1).int()
    else:
        cards = cards
    dui = (cards == 2).sum()
    ke = (cards == 3).sum()
    gang = (cards >= 4).sum()
    xiangting = 6
    # 七对
    if clc_qidui and minus == 0:
        xiangting = xiangting - dui - ke - gang * 2
        dui = (cards == 2)
        ke = (cards == 3)
        gang = (cards == 4)
        ids = idx_to_list(dui)
        ids.extend(idx_to_list(ke))
        tmp = idx_to_list(gang)
        ids.extend(tmp)
        ids.extend(tmp)
        if len(ids) != 0:
            qidui = [[[id, ] * 2 for id in ids], xiangting]
        else:
            qidui = [[], xiangting]

    if getoptim:
        xt, record = get_optim_xiangting(cards, minus=minus)
        if clc_qidui and minus == 0:
            if xt > qidui[1]:
                return qidui[1], [qidui], True
            # elif record[0][1] == qidui[1]:
            #     record.append(qidui)
        return xt, record, False
    if xiangting > limit_qidui:
        xiangting = 6
    _xiangting = get_xiangting(cards, minus=minus)
    xiangting = min(_xiangting, xiangting)
    return xiangting


'''
计算有效进张数 配合计算相听使用 有效进张指哪些牌可以进相听
cards表示由最优相听计算出的组合(上个函数返回的rec), left为除了组合外剩余的单张, cards_available可以由任何手段获得的牌
'''


def clc_yxjz(cards, minus, hand, cards_available):
    if isinstance(hand, Tensor):
        if len(hand.shape) == 2:
            hand = hand.sum(1).int()
    else:
        hand = hand
    cards = [x[0] for x in cards]
    nduizi, ndazi, nmianzi = 0, 0, 0
    nmianzi += minus // 2
    idx = [False, ] * 27
    for i in range(len(cards)):
        tmp = [0, ] * 27
        for ids in cards[i]:
            if len(ids) == 2:
                tmp[ids[0]] += 1
                tmp[ids[1]] += 1
                if ids[0] != ids[1]:
                    ndazi += 1
                    if ids[1] - ids[0] > 1:
                        idx[ids[0] + 1] = True
                    else:
                        if ids[0] - 1 > 0 and ids[0] % 9 != 0:
                            idx[ids[0] - 1] = True
                        if ids[1] + 1 < 27 and ids[1] % 9 != 8:
                            idx[ids[1] + 1] = True
                else:
                    nduizi += 1
                    idx[ids[0]] = True
            elif len(ids) == 3:
                tmp[ids[0]] += 1
                tmp[ids[1]] += 1
                tmp[ids[2]] += 1
                nmianzi += 1

        left = torch.arange(0, 27)[(hand - Tensor(tmp)).bool()].tolist()
        if nduizi == 0:
            for id in left:
                idx[id] = True
            if nmianzi + nduizi + ndazi > 4:
                for ids in cards[i]:
                    if len(ids) == 2:
                        idx[ids[0]] = True
                        idx[ids[1]] = True
        if nmianzi + nduizi + ndazi < 4:
            for id in left:
                idx[id] = True
                if id - 1 > 0 and id % 9 != 0:
                    idx[id - 1] = True
                if id + 1 < 27 and id % 9 != 8:
                    idx[id + 1] = True
        elif nmianzi + nduizi + ndazi == 4 and nduizi > 0:
            for id in left:
                idx[id] = True
                if id - 1 > 0 and id % 9 != 0:
                    idx[id - 1] = True
                if id + 1 < 27 and id % 9 != 8:
                    idx[id + 1] = True

    num_ef_cards = cards_available[idx].sum()
    ef_draw = []
    ef_draw.extend(torch.arange(0, 27)[idx].tolist())
    return ef_draw, num_ef_cards


'''
把rec转换为cards的形式,cards为[27]每一个位置表示每张牌的数量
输出是一个rec长度对应的cards的列表
'''


def rec2cards(rec):
    out = []
    for list in rec:
        zero = [0, ] * 27
        for ids in list[0]:
            for id in ids:
                zero[id] += 1
        out.append(zero)
    return out


'''
把id转换为文字的形式,0-8为1-9筒,9-17为1-9条,18-26为1-9万
输入可以为[27,4]的tensor或者[27]的tensor或者[27]的列表,或者int
zhangshu表示是否要计算张数
'''


def id2card(cards, zhangshu=False):
    message = ""
    if isinstance(cards, torch.Tensor):
        if len(cards.shape) == 2:
            cards = cards.sum(1).int()
        elif len(cards.shape) == 1:
            tmp = torch.zeros(27).int()
            for id in cards:
                tmp[id] += 1
            cards = tmp
        else:
            n = cards % 9
            m = cards // 9
            return "{}{}".format(n + 1, ["筒", "条", "万"][m])
    elif isinstance(cards, list) or isinstance(cards, np.ndarray):
        tmp = torch.zeros(27).int()
        for id in cards:
            tmp[id] += 1
        cards = tmp
    elif isinstance(cards, int):
        n = cards % 9
        m = cards // 9

        return "{}{}".format(n + 1, ["筒", "条", "万"][m])

    else:
        raise TypeError("only int list ndarray tensor support")
    for i in range(9):
        if zhangshu:
            message += "{}筒{}张,".format(i % 9 + 1, cards[i]) if cards[i] != 0 else ""
        else:
            message += "{}筒,".format(i % 9 + 1) if cards[i] != 0 else ""
    for i in range(9, 18):
        if zhangshu:
            message += "{}条{}张,".format(i % 9 + 1, cards[i]) if cards[i] != 0 else ""
        else:
            message += "{}条,".format(i % 9 + 1) if cards[i] != 0 else ""
    for i in range(18, 27):
        if zhangshu:
            message += "{}万{}张,".format(i % 9 + 1, cards[i]) if cards[i] != 0 else ""
        else:
            message += "{}万,".format(i % 9 + 1) if cards[i] != 0 else ""
    return message


'''
计算最佳出牌
cards表示手牌,可以是[27,4]也可以是[27]
cards_available表示可以从任何方式获得的牌
minus 为已经吃碰杠的数量*2
输出为 最佳牌id(int) 出牌后的有效进张(id的列表) 有效进张总数(int)
出牌策略:
七对:
判断当前单张对应的牌,在场上的剩余有效数量,选择数量最低的
多个数量相同时优先扔3-7,再是2,8,最后1,9
一般情况:
去掉cards一种的一张,重新计算相听与有效进张数
取有效进张数最多的一种组合
多个有效进张数相同时优先扔1,9,再是2,8,最后3-7
'''


# TODO 改善效率 手牌组合很多的情况下得算个10s+实在不能接受
def clc_optim_play(cards, cards_available, minus=0):
    if isinstance(cards, Tensor):
        if len(cards.shape) == 2:
            cards = cards.sum(1).int()
    else:
        cards = cards
    cards_type = torch.arange(0, 27)[cards > 0].int()
    if cards.sum() + minus // 2 * 3 != 14:
        raise AssertionError
    xt, rec, qidui = clc_xiangting(cards, minus, getoptim=True, clc_qidui=True)
    if xt < 0:
        raise AssertionError
    if qidui:
        priority = [0, 1, 2, 2, 2, 2, 2, 1, 0]
        hands = rec2cards(rec)[0]
        single = cards - torch.Tensor(hands)
        ids = torch.arange(27)[single == 1].tolist()
        min = 3
        play_list = []
        for id in ids:
            if cards_available[id] < min:
                play_list = [id]
                min = cards_available[id]
            elif cards_available[id] == min:
                play_list.append(id)
        for id in play_list:
            card = id % 9
            prior = priority[card]
            drop_id = -1
            max_prior = -1
            if prior == 2:
                ids.remove(id)
                return id, ids, int(cards_available[ids].sum()), int(xt)
            else:
                if prior > max_prior:
                    drop_id = id
                    ids.remove(drop_id)
            return drop_id, ids, int(cards_available[ids].sum()), int(xt)
    else:
        # 首先排除完全不相邻的孤张 为了尽可能减少计算
        priority = [2, 1, 0, 0, 0, 0, 0, 1, 2]
        is_guzhang = torch.stack([torch.arange(27), torch.zeros(27)], dim=1).int()
        gu_zhang = []
        record = [x[0] for x in rec]
        for ids in record:
            for id in ids:
                is_guzhang[id, 1] += 1
        gu_zhang = is_guzhang[cards_type.tolist()][is_guzhang[cards_type.tolist()][:, 1] == 0][:, 0].tolist()
        if len(gu_zhang) == 1:
            tmp = copy.deepcopy(cards)
            tmp[gu_zhang[0]] -= 1
            ef_draw, num_ef_cards = clc_yxjz(rec, minus, tmp, cards_available)
            return gu_zhang[0], ef_draw, int(num_ef_cards), int(xt)
        elif len(gu_zhang) > 1:
            max_ef_cards = 0
            best_ef_draw = []
            optim_play = []
            for id in gu_zhang:
                tmp = copy.deepcopy(cards)
                tmp[gu_zhang[0]] -= 1
                ef_draw, num_ef_cards = clc_yxjz(rec, minus, tmp, cards_available)
                if num_ef_cards > max_ef_cards:
                    best_ef_draw = [ef_draw]
                    optim_play = [id]
                    max_ef_cards = num_ef_cards
                elif num_ef_cards == max_ef_cards:
                    best_ef_draw.append(ef_draw)
                    optim_play.append(id)
            if len(optim_play) > 1:
                drop = -1
                max_prior = -1
                for i, id in enumerate(optim_play):
                    card = id % 9
                    prior = priority[card]
                    if prior == 2:
                        return id, best_ef_draw[i], int(max_ef_cards), int(xt)
                    else:
                        if prior > max_prior:
                            max_prior = prior
                            drop = i
                return optim_play[drop], best_ef_draw[drop], int(max_ef_cards), int(xt)
            else:
                return optim_play[0], best_ef_draw[0], int(max_ef_cards), int(xt)

        max_ef_cards = 0
        best_ef_draw = []
        optim_play = []
        for id in cards_type.tolist():
            tmp = copy.deepcopy(cards)
            tmp[id] -= 1
            _xt, rec, qidui = clc_xiangting(tmp, minus, getoptim=True, clc_qidui=True)
            if _xt == xt:
                ef_draw, num_ef_cards = clc_yxjz(rec, minus, tmp, cards_available)
                if num_ef_cards > max_ef_cards:
                    best_ef_draw = ef_draw
                    optim_play = [id]
                    max_ef_cards = num_ef_cards
                elif num_ef_cards == max_ef_cards:
                    best_ef_draw.append(ef_draw)
                    optim_play.append(id)
        if len(optim_play) == 0:
            raise AssertionError
        if len(optim_play) > 1:
            drop = -1
            max_prior = -1
            for i, id in enumerate(optim_play):
                card = id % 9
                prior = priority[card]
                if prior == 2:
                    return id, best_ef_draw[i], int(max_ef_cards), int(xt)
                else:
                    if prior > max_prior:
                        max_prior = prior
                        drop = i
            return optim_play[drop], best_ef_draw[drop], int(max_ef_cards), int(xt)
        else:
            return optim_play[0], best_ef_draw, int(max_ef_cards), int(xt)


cards = Tensor(
    [3, 1, 1, 1, 1, 1, 1, 1, 3,
     1, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0
     ]
)
# xt, rec, qidui = clc_optim_play(cards=cards, cards_available=)
