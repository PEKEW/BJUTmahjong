import logging
import time
from Interface.agent_critic.model import MAModel
from Interface.env_port import game
from Interface.agent_critic.agent import Agent
import torch

logging.basicConfig(filename='test_log.txt',
                    filemode='w')
log = logging.getLogger()
log.setLevel(logging.INFO)

if __name__ == "__main__":
    uid = 0
    env = game()
    model = MAModel(obs_dim=401, uid=0, num_blocks=30)
    agent = Agent(model=model, id=0, batch_size=1024, critic_batch_szie=128, save_path="\save", lr=0.001,
                  warmup_epoch=200)
    agent.resume("checkpt_930.pth", None, False)
    with torch.no_grad():
        while True:
            T = time.time()
            meta_data = {"pon_hand": "B4B4B4,C2C2C2", "kon_hand": "", "chow_hand": "D3D4D5", "hand": "D8D8D8B3C5",
                         "history": ["0,Discard,C5", "3,Discard,C7", "2,Discard,D1", "1,Chow,D1D2D3", "1,Discard,D1",
                                     "0,Discard,B1", "3,Discard,B4", "2,Pon,B4B4B4", "2,Discard,D9", "3,Pon,D9D9D9",
                                     "3,Discard,D4", "2,Chow,D3D4D5", "2,Discard,D4", "1,Chow,D4D5D6", "1,Discard,C4",
                                     "0,Discard,C2", "2,Pon,C2C2C2", "2,Discard,B2", "1,Chow,B1B2B3", "1,Discard,B5",
                                     "0,Discard,B7", "3,Discard,C6", "2,Discard,B9", "1,Discard,B5", "0,Discard,D7",
                                     "3,Discard,D2", "2,Discard,B5", "1,Discard,C5", "0,Discard,C1", "3,Discard,D2",
                                     "2,Discard,B3", "1,Discard,C7", "0,Chow,C7C8C9", "0,Discard,C9", "3,Discard,C5",
                                     "2,Discard,C1", "1,Discard,C7", "0,Listen,D1", "3,Discard,B6", "2,Discard,B6",
                                     "1,Discard,D9", "0,Discard,B9", "3,Discard,B6", "2,Discard,D8", "1,Discard,C4",
                                     "3,Pon,C4C4C4", "3,Listen,D7", "2,Discard,C3", "1,Discard,C6", "0,Discard,B7",
                                     "3,Discard,C9", "2,Discard,D5", "1,Discard,D1", "0,Discard,B7", "3,Discard,B5"],
                         "dealer": 0, "seat": 2, "special": ""}
            request = env.step(meta_data, agent)
            log.info(
                "{:},{:},".format(uid, request["action_type"]) + "{:}".format(request["action_content"])
                if request["action_content"] is not None else "")
            for i in range(4):
                log.info("玩家{:}得分为:{:}".format(i, env.history.score[i]))
            log.info("用时{:.3f}s".format(T - time.time()))
