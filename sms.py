import requests
import sys, os
import logging

server = 133
code = None
# for telegram messaging. if you don't need, deactivate all mes functions
# if you want to use this, you have to make telegram_bot 
# and get the valid code and chat_id of your own.

def mes(messeage='done', code=None, chat_id=None):
    messeage = '{} : '.format(server) + messeage
    code = code
    if code != None and chat_id != None:
        teleurl = "https://api.telegram.org/bot" + code + "/sendMessage"
        params = {'chat_id': chat_id, 'text': messeage}
        res = requests.get(teleurl, params=params)
        # sys.exit()


def mes2(messeage='done', code=None, chat_id=None):
    messeage = '{} : '.format(server) + messeage
    code = code
    if code != None and chat_id != None:
        teleurl = "https://api.telegram.org/bot" + code + "/sendMessage"
        params = {'chat_id': chat_id, 'text': messeage}
        res = requests.get(teleurl, params=params)
        # sys.exit()


def mes_done_device(messeage='done', code=None, chat_id=None):
    messeage = '{} : '.format(server) + messeage
    code = code
    if code != None and chat_id != None:
        teleurl = "https://api.telegram.org/bot" + code + "/sendMessage"
        params = {'chat_id': chat_id, 'text': messeage}
        res = requests.get(teleurl, params=params)
        # sys.exit()


def make_logger(name=None, logfilename='all'):
    # make 1 logger instance
    logging.basicConfig(
        format="[%(asctime)s] | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] | %(levelname)s | %(name)s | %(message)s")
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logfile = "./logs/{}.log".format(logfilename)
    file_handler = logging.FileHandler(filename=logfile)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # logger.info('logs will be written to [{}]'.format(logfile))
    return logger


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        mes_done_device('Done device {}'.format(sys.argv[1]))
    else:
        mes_done_device('Done..')
