import json
import logging
import pickle
import os


def get_log_path(config):
    exp_config = 'model=' + config.model +\
                 '_optimizer=' + config.optimizer +\
                 '_lr=' + str(config.lr) +\
                 '_reg=' + str(config.reg) +\
                 '_batch_size=' + str(config.batch_size) +\
                 '_epochs=' + str(config.epochs) +\
                 '_dropout=' + str(config.dropout)

    return exp_config

def count_dirs(path):
    dirs = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    return len(dirs)

def set_logger(config):
    # Based on used parameters
    log_dir = os.path.join(config.logs_dir, get_log_path(config))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        log_dir = os.path.join(log_dir, '1')
        os.makedirs(log_dir)
    else:
        log_dir = os.path.join(log_dir, str(count_dirs(log_dir) + 1))
        os.makedirs(log_dir)


    log_file_path = os.path.join(log_dir, 'console.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def set_test_logger(config):
    log_file_path = os.path.join(config.model_dir, 'test.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(dictionary, json_path):
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        dictionary = {k: float(v) for k, v in dictionary.items()}
        json.dump(dictionary, f, indent=4)


def writer(path, obj):
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()


def reader(path):
    f = open(path, 'rb')
    obj = pickle.load(f)
    return obj


def write_object(obj, name, config):
    path = os.path.join(config.logs_dir, get_log_path(config), name)
    i = 0
    while os.path.exists(path):
        i += 1
        path = os.path.join(config.logs_dir, get_log_path(config), name + str(i))

    writer(path, obj)

def test():
    print(count_dirs('logs'))

test()
