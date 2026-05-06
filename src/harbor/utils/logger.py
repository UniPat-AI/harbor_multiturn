import logging
import os


class HarborConsoleFormatter(logging.Formatter):
    _RESET = "\033[0m"
    _STYLES = {
        "multiround_enabled": "\033[1;36m",
        "multiround_start": "\033[1;34m",
        "multiround_done": "\033[1;32m",
        "multiround_parent": "\033[1;35m",
        "multiround_stop": "\033[1;33m",
        "progress_ok": "\033[32m",
        "progress_failed": "\033[31m",
        "progress_error": "\033[1;31m",
        "trial_failure": "\033[1;31m",
    }

    def __init__(self, *, enable_color: bool) -> None:
        super().__init__("%(message)s")
        self._enable_color = enable_color

    @staticmethod
    def should_enable_color(stream: object) -> bool:
        if os.getenv("NO_COLOR") is not None:
            return False
        if os.getenv("TERM") == "dumb":
            return False
        return bool(getattr(stream, "isatty", lambda: False)())

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self._enable_color:
            return message
        return "".join(self._colorize_line(line) for line in message.splitlines(True))

    def _colorize_line(self, line: str) -> str:
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        if not body:
            return line
        style = self._style_for_line(body)
        if style is None:
            return line
        return f"{style}{body}{self._RESET}{newline}"

    def _style_for_line(self, line: str) -> str | None:
        if line.startswith("[multiround]"):
            if " stop " in line:
                return self._STYLES["multiround_stop"]
            if " parent trials:" in line:
                return self._STYLES["multiround_parent"]
            if " done " in line:
                return self._STYLES["multiround_done"]
            if " start " in line:
                return self._STYLES["multiround_start"]
            return self._STYLES["multiround_enabled"]

        if line.startswith("[progress]"):
            if "status=error" in line:
                return self._STYLES["progress_error"]
            if "status=failed" in line:
                return self._STYLES["progress_failed"]
            if "status=ok" in line:
                return self._STYLES["progress_ok"]
            return self._STYLES["multiround_enabled"]

        if line.startswith("[trial="):
            return self._STYLES["trial_failure"]

        return None


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


logger = setup_logger(__name__)
