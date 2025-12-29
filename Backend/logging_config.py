import logging
from flask import g, has_request_context

class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        if has_request_context():
            record.correlation_id = getattr(g, "correlation_id", "N/A")
        else:
            record.correlation_id = "N/A"
        return True

class SafeFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt, style="{")

    def format(self, record):
        if not hasattr(record, "correlation_id"):
            record.correlation_id = "N/A"
        return super().format(record)

def configure_logger():
    LOG_FORMAT = "{asctime} - [{levelname}] - [CID={correlation_id}] - {message}"

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        style="{",
        handlers=[logging.StreamHandler()]
    )

    logger = logging.getLogger()
    safe_formatter = SafeFormatter(LOG_FORMAT)

    for handler in logger.handlers:
        handler.setFormatter(safe_formatter)
        handler.addFilter(CorrelationIdFilter())

    return logger