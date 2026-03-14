import sys
import logging
from loguru import logger

class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentation.
    Intercepts standard logging messages and routes them to loguru.
    """
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging():
    # Remove standard loguru handler
    logger.remove()

    # Add stdout handler with structed JSON logging for production
    # and readable colorized logging for development.
    import os
    import time
    os.environ['TZ'] = 'Asia/Kolkata'
    time.tzset()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Optional JSON file output for log aggregation
    logger.add("logs/cronos.json", format="{message}", level="INFO", serialize=True, rotation="10 MB")

    # Intercept everything at the root logger
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Re-configure third-party loggers
    for name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False

    logger.info("Centralized logging system initialized.")
