# author: LZ, 16.01.19
#
# Ex.:
# nohup python -u metrics_main.py exp53 20190107_114442 > nohup_metrics_exp53_20190107_114442.out &
#

import time
import subprocess
import signal
from utils_common import *
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

do_exit = False

class CheckpointCreatedEventHandler(FileSystemEventHandler):

    def __init__(self, file, params_base_dir, params):
        self.file = file
        self.params_base_dir = params_base_dir
        self.params = params

    def on_created(self, event):
        if event.src_path.endswith(".index"):
            print('checkpoint created:', event.src_path)
            key1 = 'model-'
            key2 = '.index'
            start = event.src_path.find('model-') + len(key1)
            end = len(event.src_path) - len(key2)
            iteration = int(event.src_path[start:end])
            self.params.metric_model_iteration = iteration
            print('with iteration: %d' % iteration)
            file_dir = os.path.join(self.params_base_dir, "params_" + str(iteration) + ".json")
            print('save to %s...' % file_dir)
            self.params.save(file_dir)

            # python - u lorbms_main.py - c = "calc metrics FID/IS for exp56 20190108_194739 (gen. images te_v4)"
            params_file = "-p=" + file_dir
            comment = "-c=\"calc metrics FID/IS for %s and iter %s (gen. images te_v4)\"" % (self.params.test_from, str(iteration))
            cmd = ['python', '-u', 'lorbms_main.py', params_file, comment]
            print("spawn lorbms_main [%s, %s] -->"  % (self.params.test_from, str(iteration)))
            subprocess.Popen(cmd)
            print("spawn lorbms_main [%s, %s] <--"  % (self.params.test_from, str(iteration)))


def init(argv):
    file = [p[len(JSON_FILE_PARAM):] for p in argv if
            p.startswith(JSON_FILE_PARAM) and len(p[len(JSON_FILE_PARAM):]) > 0]
    assert len(file) <= 1, 'only one params.json allowed'
    if not file:
        file.append(JSON_FILE_DEFAULT)
    file = file[0]
    params = Params(file)
    return file, params


def handle_exit(signum, frame):
    global do_exit
    do_exit = True
    print('received signal, going to exit...')


if __name__ == "__main__":
    file, params = init(sys.argv)
    metric_model_dir = os.path.join(params.log_dir, params.test_from, params.metric_model_folder)
    print('listening in folder \'%s\'...' % metric_model_dir)

    params_base_dir = os.path.join(params.log_dir, params.test_from, params.metric_results_folder)
    if not os.path.exists(params_base_dir):
        os.makedirs(params_base_dir)
        print('created params_base_dir: %s' % params_base_dir)

    # settings because of test mode
    params.is_train = "False" # here always test mode
    params.batch_size = 4 # be on the save side memorywise

    signal.signal(signal.SIGTERM, handle_exit)
    event_handler = CheckpointCreatedEventHandler(file, params_base_dir, params)
    observer = Observer()
    observer.schedule(event_handler, metric_model_dir, recursive=False)
    observer.start()

    try:
        while not do_exit:
            time.sleep(20)
    finally:
        print('stop and join...')
        observer.stop()
        observer.join()
        print('done.')