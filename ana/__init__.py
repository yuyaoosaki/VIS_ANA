import logging

logger = logging.getLogger('ana')
logger.setLevel(logging.DEBUG)

log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt="[{threadName}:{levelname}] {module} - {message}", style='{')
log_handler.setFormatter(log_formatter)
log_handler.setLevel(logging.DEBUG)
logger.addHandler(log_handler)
