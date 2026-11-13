"""Observe original sklearn fits without changing models, data, or hyperparameters."""
import csv
import sys
import time

from sklearn.linear_model import LogisticRegression

from research.common import now, write_json


class FitAudit:
    def __init__(self, output):
        self.output = output
        self.code = LogisticRegression.fit.__wrapped__.__code__
        self.started = 0
        self.completed = 0
        self.pending = {}
        self.stream = (output / 'fit_audit.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=['fit_index', 'partition', 'arm',
                                      'stage', 'images', 'width', 'C', 'fit_sec', 'n_iter', 'status'])
        self.writer.writeheader()

    def observe(self, frame, event, value):
        if frame.f_code is not self.code:
            return
        if event == 'call':
            self.started += 1
            index = self.started
            partition = (index - 1) // 195
            arm = ['A', 'B', 'A+B'][((index - 1) // 65) % 3]
            row = {'fit_index': index, 'partition': f'outer{partition}' if partition < 3 else 'full_train',
                   'arm': arm, 'stage': 'candidate' if (index - 1) % 65 < 52 else 'refit',
                   'images': frame.f_locals['X'].shape[0], 'width': frame.f_locals['X'].shape[1],
                   'C': frame.f_locals['self'].C}
            self.pending[id(frame)] = (time.perf_counter(), row)
        elif event == 'return':
            start, row = self.pending.pop(id(frame))
            ok = value is frame.f_locals['self']
            self.completed += int(ok)
            row.update(fit_sec=time.perf_counter() - start,
                       n_iter=int(value.n_iter_.max()) if ok else '', status='COMPLETE' if ok else 'FAILED')
            self.writer.writerow(row); self.stream.flush()
            write_json(self.output / 'fit_progress.json', {'fits_started': self.started,
                       'fits_completed': self.completed, 'expected_fits': 780, 'last_fit': row, 'timestamp': now()})

    def __enter__(self):
        assert sys.getprofile() is None
        sys.setprofile(self.observe)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)
        self.stream.close()
