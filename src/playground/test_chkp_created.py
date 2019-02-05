import time
from datetime import datetime

PREFIX = 'DCGAN.model-'
POSTFIX = '.index'

def on_created(src_path):
    print("[%s] on_created: %s" % (datetime.now().strftime('%Y%m%d_%H%M%S'), src_path))

    iprefix = src_path.find(PREFIX)
    ipostfix = src_path.find(POSTFIX)

    if iprefix != -1 and ipostfix != -1 and iprefix < ipostfix:
        # ex: on_created:  <FileCreatedEvent: src_path='logs/20190116_222243/metrics/model/DCGAN.model-6.index.tempstate17813430530705777942'>
        print('checkpoint created:', src_path)

        start = iprefix + len(PREFIX)
        end = ipostfix
        iteration = int(src_path[start:end])
        print('with iteration: %d' % iteration)




if __name__ == "__main__":

    src_path = 'logs/20190116_222243/metrics/model/DCGAN.model-6987.index.tempstate17813430530705777942'
    on_created(src_path)

    src_path = 'logs/20190116_222243/metrics/model/DCGAN.model-5.index'
    on_created(src_path)

    src_path = 'is_index_dotprice.txt'
    on_created(src_path)
