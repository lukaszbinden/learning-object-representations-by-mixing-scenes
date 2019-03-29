# author: LZ, 16.01.19
#
# Ex.:
# run_metrics_exp59.sh
#
# nohup python -u metrics_main.py -exp=exp59 -test_from=20190118_172400 -main=lorbms_main_exp59.py -p=params_exp59.json > nohup_metrics_exp59_20190118_172400.out &
#

import time
from datetime import datetime
import subprocess
import signal
from utils_common import *
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

do_exit = False
PREFIX = 'DCGAN.model-'
POSTFIX = '.index'

class CheckpointCreatedEventHandler(FileSystemEventHandler):

    def __init__(self, file, main, metric_results_folder, params):
        self.file = file
        self.main = main # e.g. 'lorbms_main.py'
        self.metric_results_folder = metric_results_folder
        self.params = params

    def on_created(self, event):
        print("[%s] on_created: %s" % (datetime.now().strftime('%Y%m%d_%H%M%S'), event))

        iprefix = event.src_path.find(PREFIX)
        ipostfix = event.src_path.find(POSTFIX)

        if iprefix != -1 and ipostfix != -1 and iprefix < ipostfix:
            # ex: on_created:  <FileCreatedEvent: src_path='logs/20190116_222243/metrics/model/DCGAN.model-6.index.tempstate17813430530705777942'>
            print('checkpoint created:', event.src_path)

            start = iprefix + len(PREFIX)
            end = ipostfix
            iteration = int(event.src_path[start:end])

            self.params.metric_model_iteration = iteration

            for stats_type in ["training", "test"]: # cf. params.stats_type
                self.params.stats_type = stats_type
                print('run LORBMS test with checkpoint (iteration): %d and stats_type: %s...' % (iteration, stats_type))
                metric_results_dir = os.path.join(self.metric_results_folder, stats_type)
                if not os.path.exists(metric_results_dir):
                    os.makedirs(metric_results_dir)
                    print('created metric_results_dir: %s' % metric_results_dir)
                file_dir = os.path.join(metric_results_dir, "params_" + str(iteration) + ".json")
                print('save to %s...' % file_dir)
                self.params.save(file_dir)

                # wait for 3s because of weird file name (prob due to file system..)
                try:
                    time.sleep(3)
                finally:
                    # python - u lorbms_main.py - c = "calc metrics FID/IS for exp56 20190108_194739 (gen. images te_v4)"
                    params_file = "-p=" + file_dir
                    comment = "-c=\"calc metrics FID/IS for %s and iter %s\"" % (self.params.test_from, str(iteration))
                    cmd = ['python', '-u', self.main, params_file, comment]
                    print("spawn lorbms_main [%s, %s] -->"  % (self.params.test_from, str(iteration)))
                    subprocess.Popen(cmd)
                    print("spawn lorbms_main [%s, %s] <--"  % (self.params.test_from, str(iteration)))

                if stats_type == "training":
                    # wait some seconds to let training process take off, then start test in parallel (CPU)
                    time.sleep(10)


def init(argv):
    file = [p[len(JSON_FILE_PARAM):] for p in argv if
            p.startswith(JSON_FILE_PARAM) and len(p[len(JSON_FILE_PARAM):]) > 0]
    assert len(file) <= 1, 'only one params.json allowed'
    if not file:
        file.append(JSON_FILE_DEFAULT)
    file = file[0]
    print('params.json..: ', file)

    test_from = [p[len(TEST_FROM_PARAM):] for p in argv if
            p.startswith(TEST_FROM_PARAM) and len(p[len(TEST_FROM_PARAM):]) > 0]
    assert len(test_from) == 1 and test_from[0], '-test_from= param is missing'
    test_from = test_from[0]

    print('test_from....: ', test_from)

    exp = [p[len(EXP_PARAM):] for p in argv if
                 p.startswith(EXP_PARAM) and len(p[len(EXP_PARAM):]) > 0]
    assert len(exp) == 1 and exp[0], '-exp= param is missing'
    exp = exp[0]
    print('exp..........: ', exp)

    main = [p[len(MAIN_PARAM):] for p in argv if
                 p.startswith(MAIN_PARAM) and len(p[len(MAIN_PARAM):]) > 0]
    assert len(main) == 1 and main[0], '-main= param is missing (e.g. lorbms_main.py)'
    main = main[0]
    print('main.........: ', main)

    params = Params(file)
    params.test_from = test_from
    params.exp = exp
    params.main = main
    return file, main, params


def handle_exit(signum, frame):
    global do_exit
    do_exit = True
    print('received signal, going to exit...')


if __name__ == "__main__":
    file, main, params = init(sys.argv)
    metric_model_dir = os.path.join(params.log_dir, params.test_from, params.metric_model_folder)
    print('listening in folder \'%s\'...' % metric_model_dir)

    metric_results_folder = os.path.join(params.log_dir, params.test_from, params.metric_results_folder)
    if not os.path.exists(metric_results_folder):
        os.makedirs(metric_results_folder)
        print('created metric_results_folder: %s' % metric_results_folder)

    # settings because of test mode
    params.is_train = "False" # here always test mode
    params.batch_size = 4 # be on the save side memorywise
    params.gpu = -1 # always use CPU

    signal.signal(signal.SIGTERM, handle_exit)
    event_handler = CheckpointCreatedEventHandler(file, main, metric_results_folder, params)
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