import sys
import time
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

do_exit = False

class FileCreatedEventHandler(FileSystemEventHandler):

    def __init__(self, params):
        self.params = params

    def on_created(self, event):
        print('file created: ', event)
        if event.src_path.endswith(".index"):
            print('new model arrived!!')
            key1 = 'model-'
            key2 = '.index'
            start = event.src_path.find('model-') + len(key1)
            end = len(event.src_path) - len(key2)
            iteration = int(event.src_path[start:end])
            print(iteration)

def handle_exit(signum, frame):
    global do_exit
    do_exit = True
    print('received signal, going to exit...')

if __name__ == "__main__":

    path = sys.argv[1] if len(sys.argv) > 1 else '.'

    signal.signal(signal.SIGTERM, handle_exit)
    event_handler = FileCreatedEventHandler("hello")
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while not do_exit:
            time.sleep(5)
    finally:
        print('stop and join...')
        observer.stop()
        observer.join()
    print('done.')

