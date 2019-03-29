# author: LZ, 06.02.19
#
# Calcuates metrics (FID/IS) for all checkpoints in a given directory, subfolder params.metric_model_folder.
#
# Ex.:
# run_metrics_all_ckpts_exp53.sh
#
# nohup python -u metrics_all_ckpts.py -exp=exp53 -test_from=20190107_114442 -main=lorbms_main_exp53.py -p=params_exp53.json >  nohup_metrics_list_exp53_20190107_114442.out &
#

import time
from datetime import datetime
import subprocess
from utils_common import *
import glob

do_exit = False
PREFIX = 'DCGAN.model-'
POSTFIX = '.index'

class DirectoryCheckpointHandler:

    def __init__(self, file, main, metric_model_dir, metric_results_folder, params):
        self.file = file
        self.main = main # e.g. 'lorbms_main.py'
        self.metric_model_dir = metric_model_dir
        self.metric_results_folder = metric_results_folder
        self.params = params

    def execute(self):
        print("execute --> [%s]" % (datetime.now().strftime('%Y%m%d_%H%M%S')))

        print('checkpoint dir: %s' % self.metric_model_dir)

        cwd = os.getcwd()
        os.chdir(self.metric_model_dir)
        iterations = []
        for ckpt in glob.glob("DCGAN.model-*.index"):
            iprefix = ckpt.find(PREFIX)
            ipostfix = ckpt.find(POSTFIX)
            start = iprefix + len(PREFIX)
            end = ipostfix
            iteration = int(ckpt[start:end])
            iterations.append(iteration)

        iterations.sort()
        print(iterations)

        os.chdir(cwd)

        for iteration in iterations:
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

                # python - u lorbms_main.py - c = "calc metrics FID/IS for exp56 20190108_194739 (gen. images te_v4)"
                params_file = "-p=" + file_dir
                comment = "-c=\"calc metrics FID/IS for %s and iter %s and type %s\"" % (self.params.test_from, str(iteration), stats_type)
                cmd = ['python', '-u', self.main, params_file, comment]
                # cmd = ['echo', "hello from subprocess " + str(iteration)]
                print("subprocess lorbms_main [%s, %s] -->"  % (self.params.test_from, str(iteration)))
                process = subprocess.Popen(cmd)
                print("wait for subprocess...")
                process.wait()
                print("subprocess lorbms_main [%s, %s] <--"  % (self.params.test_from, str(iteration)))

        print("execute <-- [%s]" % (datetime.now().strftime('%Y%m%d_%H%M%S')))



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


if __name__ == "__main__":
    print('main -->')

    file, main, params = init(sys.argv)
    metric_model_dir = os.path.join(params.log_dir, params.test_from, params.metric_model_folder)
    print('processing checkpoints in folder \'%s\'...' % metric_model_dir)

    metric_results_folder = os.path.join(params.log_dir, params.test_from, params.metric_results_folder)
    if not os.path.exists(metric_results_folder):
        os.makedirs(metric_results_folder)
        print('created metric_results_folder: %s' % metric_results_folder)

    # settings because of test mode
    params.is_train = False # here always test mode
    params.batch_size = 4 # be on the save side memorywise
    params.gpu = -1 # always use CPU

    handler = DirectoryCheckpointHandler(file, main, metric_model_dir, metric_results_folder, params)
    handler.execute()

    print('main <--')
