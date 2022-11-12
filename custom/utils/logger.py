import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TKAgg")
import json
import os
import atexit
import time


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

class MyLogger:
    def __init__(self, output_dir=None, output_fname="progress.txt", exp_name=None):
        self.epoch_dict = dict()

        self.output_dir = output_dir or "/tmp/experiments/%i"%int(time.time())
        if os.path.exists(self.output_dir):
            print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
        else:
            os.makedirs(self.output_dir)
        self.output_file = open(os.path.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print(self.colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))

        self.first_row=True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def store(self, **kwargs):
        for key, value in kwargs.items():
            if not(key in self.epoch_dict.keys()):
                self.epoch_dict[key] = []
            self.epoch_dict[key].append(value)

    def plot(self, signals):
        for i, key in zip(range(len(signals)), signals):
            plt.figure(num=i)
            plt.plot(list(map(abs, self.epoch_dict[key])))
            plt.title(label=key)
            plt.yscale('log')

    def convert_json(self, obj):
        """ Convert obj to a version which can be serialized with JSON. """
        if self.is_json_serializable(obj):
            return obj
        else:
            if isinstance(obj, dict):
                return {self.convert_json(k): self.convert_json(v) 
                        for k,v in obj.items()}

            elif isinstance(obj, tuple):
                return (self.convert_json(x) for x in obj)

            elif isinstance(obj, list):
                return [self.convert_json(x) for x in obj]

            elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
                return self.convert_json(obj.__name__)

            elif hasattr(obj,'__dict__') and obj.__dict__:
                obj_dict = {self.convert_json(k): self.convert_json(v) 
                            for k,v in obj.__dict__.items()}
                return {str(obj): obj_dict}

            return str(obj)

    def is_json_serializable(self, v):
        try:
            json.dumps(v)
            return True
        except:
            return False

    def save_config(self, config):
        """
        Log an experiment configuration.
        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 
        Example use:
        .. code-block:: python
            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = self.convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
        print(self.colorize('Saving config:\n', color='cyan', bold=True))
        print(output)
        with open(os.path.join(self.output_dir, "config.json"), 'w') as out:
            out.write(output)

    def colorize(self, string, color, bold=False, highlight=False):
        """
        Colorize a string.
        This function was originally written by John Schulman.
        """
        attr = []
        num = color2num[color]
        if highlight: num += 10
        attr.append(str(num))
        if bold: attr.append('1')
        return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

    def log(self, msg, color='green', write_to_file=False):
        """Print a colorized message to stdout."""
        print(self.colorize(msg, color, bold=True))

        if write_to_file and (self.output_file is not None):
            self.output_file.write(msg + "\n")
            self.output_file.flush()