import os, json, time
from collections import defaultdict

class MetricLogger:
    def __init__(self, out_file=None):
        self.data = defaultdict(list)
        self.out_file = out_file
    def log(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(float(v))
    def dump(self):
        if self.out_file:
            with open(self.out_file, 'w') as f:
                json.dump(self.data, f, indent=2)
    def summary(self):
        return {k: sum(v)/len(v) for k, v in self.data.items() if v}
