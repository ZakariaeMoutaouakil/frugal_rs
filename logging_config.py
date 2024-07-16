from logging import Formatter, StreamHandler, getLogger, Logger, LogRecord, ERROR, INFO, DEBUG, WARNING, CRITICAL
from logging.handlers import RotatingFileHandler
from os import path, makedirs
from typing import Dict, Any

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class ColorFormatter(Formatter):
    """
    Custom formatter class to add colors to log messages based on their level.
    """

    COLORS: Dict[int, str] = {
        DEBUG: Fore.CYAN,
        INFO: Fore.GREEN,
        WARNING: Fore.YELLOW,
        ERROR: Fore.RED,
        CRITICAL: Fore.MAGENTA
    }

    def format(self, record: LogRecord) -> str:
        """
        Format the specified log record as text.

        Args:
            record (LogRecord): The log record to be formatted.

        Returns:
            str: The formatted log message with appropriate color.
        """
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)
        levelname = f"{record.levelname:<8}"
        record.msg = f"{log_color}{levelname} - {record.msg}{Style.RESET_ALL}"
        if record.exc_info:
            record.exc_text = self.formatException(record.exc_info)
        return super().format(record)


def setup_logger(name: str, config: Dict[str, Any]) -> Logger:
    """
    Set up and configure a logger with the given name and configuration.

    Args:
        name (str): The name of the logger.
        config (Dict[str, Any]): A dictionary containing logger configuration.

    Returns:
        Logger: The configured logger instance.
    """
    logger = getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(config['level'])

        # File handler (with rotation)
        makedirs(path.dirname(config['log_file']), exist_ok=True)
        file_handler = RotatingFileHandler(
            config['log_file'],
            maxBytes=config['max_file_size'],
            backupCount=config['backup_count']
        )
        file_handler.setLevel(config['level'])
        file_format = Formatter('%(levelname)-8s : %(message)s')
        file_handler.setFormatter(file_format)

        # Console handler
        console_handler = StreamHandler()
        console_handler.setLevel(config['level'])
        color_format = ColorFormatter('%(message)s')
        console_handler.setFormatter(color_format)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def change_log_level(logger: Logger, level: int) -> None:
    """
    Change the logging level for the specified logger and all its handlers.

    Args:
        logger (Logger): The logger whose level needs to be changed.
        level (int): The new logging level.
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def basic_logger(log_file: str) -> Logger:
    """
    Create a basic logger with default configuration.

    Args:
        log_file (str): The path to the log file.

    Returns:
        Logger: A configured logger instance.
    """
    basic_config = {
        'level': DEBUG,
        'log_file': log_file,
        'max_file_size': 50 * 1024 * 1024,  # 50 MB
        'backup_count': 3
    }
    return setup_logger('basic_logger', basic_config)


def main() -> None:
    # Default configuration
    default_config: Dict[str, str | int] = {
        'level': DEBUG,
        'log_file': 'logs/app.log',
        'max_file_size': 5 * 1024 * 1024,  # 5 MB
        'backup_count': 3
    }

    # Usage
    logger = setup_logger('my_app', default_config)

    # Example usage
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    try:
        1 / 0
    except ZeroDivisionError as e:
        logger.exception("An error occurred: %s", e)

    # Change log level
    change_log_level(logger, INFO)


if __name__ == '__main__':
    main()
