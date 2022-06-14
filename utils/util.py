from tabulate import tabulate, TableFormat, Line, DataRow
import colorlog
import errno
import logging
import os
import pandas as pd
import json

# Riiight
format = '%(log_color)s%(asctime)15s %(name)15s %(levelname)10s - %(message)s'
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    format, datefmt="%y%m%d %H:%M:%S"))

# This shouldn't be here.

root = logging.getLogger()
root.setLevel(logging.WARNING)

myfmt = TableFormat(lineabove=Line("", "-", "  ", ""),
                    linebelowheader=Line("| ", "-", " | ", " |"),
                    linebetweenrows=None,
                    linebelow=None,
                    headerrow=DataRow("| ", " | ", " |"),
                    datarow=DataRow("| ", " | ", " |"),
                    padding=0,
                    with_header_hide=["lineabove", "linebelow"])


def read_config(config_file):
    """
    Read configuration and return it in a dict
    """

    # Check if the configuration file exists.
    if os.path.isfile(config_file):
        with open(config_file) as config_f:
            config = json.load(config_f)
    else:
        raise IOError(
            errno.ENOENT, os.strerror(errno.ENOENT), config_file)

    return config


def get_logger(name, level=logging.DEBUG):
    """
    This function returns a logger object. It removes existing handles,
    and sets the propagation flag to False to avoid double outputs.
    """

    logger = colorlog.getLogger(name)
    while logger.handlers:
        logger.handlers.pop()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def ppjson(d):
    """
    This function prints JSON in a nicer way
    """
    return json.dumps(d, sort_keys=True, indent=4)


def table(logger, table, message="", level=logging.DEBUG):
    """
    Print a table (pandas for example) in Markdown format
    """

    if isinstance(table, pd.DataFrame):
        table.reset_index(inplace=True)

    tbl = tabulate(
        table,
        headers='keys',
        tablefmt=myfmt,
        showindex=False)

    logger.log(level, message + "\n" + tbl)
