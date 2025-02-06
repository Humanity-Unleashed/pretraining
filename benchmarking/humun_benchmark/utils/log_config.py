import logging
import logging.config
import logging.handlers
import atexit


class StdoutFilter(logging.Filter):
    """Allow only INFO and lower logs to stdout"""

    def filter(self, record):
        return record.levelno <= logging.INFO


class StderrFilter(logging.Filter):
    """Allow only WARNING and higher logs to stderr"""

    def filter(self, record):
        return record.levelno > logging.INFO


def setup_logging(log_filepath: str = None):
    """
    Sets up simple plaintext logging for files and console output.

    log_filepath (str): add the file-writing log handler to config if provided,
        otherwise just prints to stdout/stdin.
    """

    # clear any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # `basic_handlers`
    basic_handlers = ["console_stdout", "console_stderr"]

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "[%(asctime)s] %(levelname)s - %(name)s:\n\t %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console_stdout": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "detailed",
                "filters": ["stdout_filter"],
            },
            "console_stderr": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "formatter": "detailed",
                "filters": ["stderr_filter"],
            },
            "output_log": {
                "class": "logging.FileHandler",
                "filename": log_filepath,
                "mode": "a",
                "formatter": "detailed",
            },
        },
        "filters": {
            "stdout_filter": {"()": StdoutFilter},
            "stderr_filter": {"()": StderrFilter},
        },
        "loggers": {
            "humun_benchmark": {
                "level": "DEBUG",
                "handlers": basic_handlers + ["output_log"]
                if log_filepath
                else basic_handlers,
                "propagate": True,
            }
        },
    }

    logging.config.dictConfig(log_config)

    if log_filepath:
        log = logging.getLogger(__name__)
        log.info(f"Writing logs to {log_filepath}")

    # ensure logging is shutdown automatically on exit
    atexit.register(logging.shutdown)
