import csv
import logging
import os


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


# traincolumns should follow the format {"object":"format"}
class logger(object):
    def __init__(self, cfg):
        if not os.path.exists(cfg.log.save_path):
            os.makedirs(cfg.log.save_path)
        self.log = get_logger(logpath=os.path.join(cfg.log.save_path, 'logs'))
        self.trainlogpath = os.path.join(cfg.log.save_path, 'training.csv')
        self.testlogpath = os.path.join(cfg.log.save_path, 'test.csv')
        assert len(cfg.log.trainformat) == len(cfg.log.traincolumns)
        assert len(cfg.log.testformat) == len(cfg.log.testcolumns)
        self.traincolumns = cfg.log.traincolumns
        self.trainformat = cfg.log.trainformat
        self.testcolumns = cfg.log.testcolumns
        self.testformat = cfg.log.testformat
        if not os.path.exists(self.trainlogpath):
            with open(self.trainlogpath, 'w') as f:
                csvlogger = csv.DictWriter(f, cfg.log.traincolumns)
                csvlogger.writeheader()
        if not os.path.exists(self.testlogpath):
            with open(self.testlogpath, 'w') as f:
                csvlogger = csv.DictWriter(f, cfg.log.testcolumns)
                csvlogger.writeheader()

    def info(self, message):
        self.log.info(message)

    def write(self, message, mode="train"):
        assert mode in ["train", "test", "eval"]
        if mode == "train":
            assert len(message) == len(self.traincolumns)
            logpath = self.trainlogpath
            columns = self.traincolumns
            form = self.trainformat
        else:
            assert len(message) == len(self.testcolumns)
            logpath = self.testlogpath
            columns = self.testcolumns
            form = self.testformat
        # try:
        logdict = {
            columns[i]: (message[i] if form[i] is None else
                                   form[i].format(message[i]))
            for i in range(len(message))
        }
        with open(logpath, 'a') as f:
            csvlogger = csv.DictWriter(f, columns)
            csvlogger.writerow(logdict)
        # except:
        #     self.info("fail to write log")
