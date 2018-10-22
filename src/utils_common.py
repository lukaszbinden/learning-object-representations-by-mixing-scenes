"""General utility functions"""
""" cf. https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision/model/utils.py"""

import os
import sys
import shutil
import json
import logging
import pprint
import logging as log

class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def __repr__(self):
        return "Params(\n" + get_pp().pformat(self.__dict__) + "\n)"

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())

   def flush(self):
      for handler in self.logger.handlers:
         handler.flush()


def init_logging(log_dir, log_file_name):
    log_path = os.path.join(log_dir, log_file_name)
    set_logger(log_path)
    # Redirect stdout and stderr
    stdout_logger = log.getLogger('STDOUT')
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    stderr_logger = logging.getLogger('STDERR')
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
    global pp
    pp = pprint.PrettyPrinter()


def copy_src(params):
    main_name = os.path.basename(sys.argv[0])
    model_name = main_name.replace('main', 'model')
    print('main_name: ', main_name)
    print('model_name: ', model_name)
    base_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    src_files = os.listdir(base_dir)
    for file_name in src_files:
        full_file_name = os.path.join(base_dir, file_name)
        print('full_file_name: ', full_file_name)
        if os.path.isfile(full_file_name) and (file_name == main_name or file_name == model_name):
            shutil.copy(full_file_name, params.src_dir)


def get_pp():
    return pp


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        # default: '%(asctime)s:%(levelname)s: %(message)s'
        # "%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


