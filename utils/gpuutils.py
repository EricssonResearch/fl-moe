"""
Simple helper functions for GPU use in AI lab
"""

import subprocess
import io
import pandas as pd


def get_gpu_info(fields=["index", "utilization.gpu", "memory.free"]):
    """
    Return a pandas data frame with GPU info, indexed by GPU index.

    See `nvidia-smi --help-query-gpu` for a list of available fields.
    """

    sp = subprocess.Popen(
        ['nvidia-smi', '--query-gpu=' + ",".join(fields),
         '--format=csv'
         ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    csv = io.StringIO(sp.stdout.read().decode())
    data = pd.read_csv(csv, index_col=0, names=fields, skiprows=1)
    csv.close()

    sp.terminate()

    return data


def get_available_gpus(threshold=1400):
    """
    Return a list of GPU indices where `threshold` MB of memory is free.
    """

    data = get_gpu_info()
    data['memory.free'] = data['memory.free'].str.extract('(\d+)').astype(int)
    data = data.sort_values("memory.free", ascending=False)
    return data[data["memory.free"] > threshold].index.values.tolist()
