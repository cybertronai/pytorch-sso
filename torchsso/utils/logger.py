import os
import time
import json
import shutil


# Select the best-resolution timer function
try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time


class Logger(object):

    def __init__(self, out, logname):
        self.out = out
        self.logname = logname
        self._log = []
        self._start_at = None

        if not os.path.isdir(self.out):
            os.makedirs(self.out)

    def start(self):
        self._start_at = _get_time()

    @property
    def elapsed_time(self):
        if self._start_at is None:
            raise RuntimeError('training has not been started yet')
        return _get_time() - self._start_at

    def write(self, log):
        self._log.append(log)
        tmp_path = os.path.join(self.out, 'tmp')
        with open(tmp_path, 'w') as f:
            json.dump(self._log, f, indent=4)

        path = os.path.join(self.out, self.logname)
        shutil.move(tmp_path, path)

