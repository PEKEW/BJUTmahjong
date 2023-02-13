from torch import nn
import torch
from torch import Tensor
import numpy as np
import copy


def idx_to_list(idx):
    return torch.IntTensor(list(range(len(idx))))[idx].tolist()


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


def id2msg(cards):
    message = ""
    if isinstance(cards, torch.Tensor):
        if len(cards.shape) == 2:
            card = cards.sum(1).int()
            for i in range(27):
                m = cards // 9
                n = cards % 9
                while card[i] > 0:
                    message += "{}{}".format(["D", "B", "C"][m], n + 1)
                    card[i] -= 1
            return message
        elif len(cards.shape) == 1:
            for id in cards:
                n = id % 9
                m = id // 9
                message += "{}{}".format(["D", "B", "C"][m], n + 1)
            return message
        else:
            n = cards % 9
            m = cards // 9
            return "{}{}".format(["D", "B", "C"][m], n + 1)
    elif isinstance(cards, list) or isinstance(cards, np.ndarray):
        for id in cards:
            n = id % 9
            m = id // 9
            message += "{}{}".format(["D", "B", "C"][m], n + 1)
        return message
    elif isinstance(cards, int):
        n = cards % 9
        m = cards // 9
        return "{}{}".format(["D", "B", "C"][m], n + 1)
    else:
        raise TypeError("only int list ndarray tensor support")


def msg2id(msg, tensor):
    type = {"D": 0, "B": 9, "C": 18}
    id = 0
    if tensor:
        output = torch.zeros(27)
        for str in msg:
            if str in ["D", "B", "C"]:
                id = type[str]
            else:
                id = id + int(str) - 1
                output[id] += 1
    else:
        output = []
        for str in msg:
            if str in ["D", "B", "C"]:
                id = type[str]
            else:
                id = id + int(str) - 1
                output.append(id)
    return output


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


# minus 为已经吃碰杠的数量*2
def get_xiangting(cards, minus=0, nmianzi=0, nduizi=0, ndazi=0, nkezi=1, nshunzi=1, rec=None):
    if rec is None:
        rec = [[], [], [], [], []]
    if nkezi != 0:
        kezi = get_kezi(cards)
        nkezi = len(kezi)
    if nshunzi != 0:
        shunzi = get_shunzi(cards)
        nshunzi = len(shunzi)
    minus = minus
    xiangting = 9
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
                    _xiangting, rec = get_xiangting(tmp, minus, nmianzi + 1, nkezi=nkezi - 1, nshunzi=nshunzi, rec=rec)
                    xiangting = min(_xiangting, xiangting)
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
                    _xiangting, rec = get_xiangting(tmp, minus, nmianzi + 1, nkezi=nkezi, nshunzi=nshunzi - 1, rec=rec)
                    xiangting = min(_xiangting, xiangting)
    else:
        dui, da = get_dazi(cards)
        if nmianzi + ndazi + nduizi < 5:
            if len(dui) > 0:
                for id in dui:
                    tmp = copy.deepcopy(cards)
                    tmp[id] = tmp[id] - 2

                    flag = True
                    layer = nmianzi + nduizi + ndazi
                    for r in rec[layer]:
                        if all(r[0] == tmp) and nmianzi == r[1] and nduizi+1 == r[2]:
                            flag = False
                            break
                    if flag:
                        rec[layer].append([tmp, nmianzi, nduizi + 1])
                        _xiangting, rec = get_xiangting(tmp, minus, nmianzi, nduizi + 1, ndazi, 0, 0, rec=rec)
                        xiangting = min(_xiangting, xiangting)
            if len(da) > 0:
                for (a, b) in da:
                    tmp = copy.deepcopy(cards)
                    tmp[a] = tmp[a] - 1
                    tmp[b] = tmp[b] - 1

                    flag = True
                    layer = nmianzi + nduizi + ndazi
                    for r in rec[layer]:
                        if all(r[0] == tmp) and nmianzi == r[1] and nduizi+1 == r[2]:
                            flag = False
                            break
                    if flag:
                        rec[layer].append([tmp, nmianzi, nduizi + 1])
                        _xiangting, rec = get_xiangting(tmp, minus, nmianzi, nduizi, ndazi + 1, 0, 0, rec=rec)
                        xiangting = min(_xiangting, xiangting)
        else:
            dui, da = [], []
        if len(dui) == 0 and len(da) == 0:
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
            return min(xt, xiangting), rec

    return xiangting, rec


def clc_xiangting(cards=None, minus=0, clc_qidui=True):
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
    _xiangting, _ = get_xiangting(cards, minus=minus)
    xiangting = min(_xiangting, xiangting)
    return xiangting