import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileCreatedEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        print('file created: ', event)
        if event.src_path.endswith(".index"):
            print('new model arrived!!')

if __name__ == "__main__":

    path = sys.argv[1] if len(sys.argv) > 1 else '.'

    event_handler = FileCreatedEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()