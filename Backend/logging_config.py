import logging
from flask import g, has_request_context

class CorrelationIdFilter(logging.Filter):
    """
    Logging filter that injects a correlation ID into every log record.

    This filter checks whether the current execution is inside a Flask
    request context. If so, it retrieves `g.correlation_id` (set earlier
    in the request lifecycle). If not available, or if no request context
    exists (e.g., during app startup), it defaults to "N/A".

    Adding this filter ensures that all logs—application logs, errors,
    and access logs—carry a consistent correlation ID, making it easier
    to trace a single request across multiple log entries.
    """

    def filter(self, record):
        if has_request_context():
            record.correlation_id = getattr(g, "correlation_id", "N/A")
        else:
            record.correlation_id = "N/A"
        return True


class SafeFormatter(logging.Formatter):
    """
    Custom log formatter that safely handles missing correlation IDs.

    This formatter extends Python's standard logging.Formatter but ensures
    that the `correlation_id` attribute always exists on the log record.
    This prevents formatting errors when logs are generated outside a
    request context or before the CorrelationIdFilter is applied.

    Args:
        fmt (str): Log message format string.
        datefmt (str): Optional date format string.
    """

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt=fmt, datefmt=datefmt, style="{")

    def format(self, record):
        if not hasattr(record, "correlation_id"):
            record.correlation_id = "N/A"
        return super().format(record)


def configure_logger():
    """
    Configure the global application logger with correlation ID support.

    This function:
    - Defines a structured log format including timestamp, level, and correlation ID.
    - Initializes Python's root logger with INFO level and a StreamHandler.
    - Applies SafeFormatter to ensure logs never break due to missing fields.
    - Attaches CorrelationIdFilter so every log record includes a correlation ID.

    Returns:
        logging.Logger: The configured root logger instance.

    Usage:
        logger = configure_logger()
        logger.info("Application started")
    """

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
