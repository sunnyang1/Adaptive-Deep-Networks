"""
Structured Logging for Experiments

JSON-formatted logs for machine parsing with human-readable console output.
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for machine-readable logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'extras'):
            log_data['extras'] = record.extras
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for console output."""
    
    def __init__(self, colored: bool = True):
        super().__init__()
        self.colored = colored and sys.stdout.isatty()
        
        # Color codes
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m',
        }
    
    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        message = record.getMessage()
        
        if self.colored and level in self.colors:
            level = f"{self.colors[level]}{level}{self.colors['RESET']}"
        
        return f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {message}"


def setup_logging(
    name: str,
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    use_json: bool = True,
    console: bool = True
) -> logging.Logger:
    """
    Setup structured logging for experiments.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (None for no file logging)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        use_json: Use JSON format for file logs
        console: Enable console output
    
    Returns:
        Configured logger
    
    Example:
        >>> logger = setup_logging("exp1", log_dir=Path("logs"))
        >>> logger.info("Starting experiment")
        >>> logger.info("Progress", extra={'extras': {'epoch': 1, 'loss': 0.5}})
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        if use_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(HumanReadableFormatter())
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger or create new one with default settings."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create default handler if none exists
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(HumanReadableFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


class ExperimentLogger:
    """
    High-level experiment logging interface.
    
    Provides structured logging for experiment phases, metrics, and results.
    """
    
    def __init__(self, name: str, log_dir: Optional[Path] = None):
        self.logger = setup_logging(name, log_dir)
        self.name = name
        self.metrics: Dict[str, Any] = {}
        self.start_time = datetime.now()
    
    def log_phase(self, phase: str, status: str = "started", **kwargs):
        """Log experiment phase."""
        self.logger.info(
            f"Phase '{phase}': {status}",
            extra={'extras': {'phase': phase, 'status': status, **kwargs}}
        )
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        self.metrics[name] = value
        extras = {'metric': name, 'value': value}
        if step is not None:
            extras['step'] = step
        self.logger.info(f"Metric {name}={value:.4f}", extra={'extras': extras})
    
    def log_result(self, key: str, value: Any):
        """Log a final result."""
        self.logger.info(
            f"Result {key}={value}",
            extra={'extras': {'result_key': key, 'result_value': value}}
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with context."""
        self.logger.error(
            f"{context}: {str(error)}" if context else str(error),
            exc_info=True,
            extra={'extras': {'error_type': type(error).__name__}}
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            'name': self.name,
            'duration_seconds': duration,
            'metrics': self.metrics,
        }
