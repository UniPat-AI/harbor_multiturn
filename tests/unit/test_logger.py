import logging

from harbor.utils.logger import HarborConsoleFormatter


def _format_message(message: str, *, enable_color: bool) -> str:
    formatter = HarborConsoleFormatter(enable_color=enable_color)
    record = logging.LogRecord(
        name="harbor.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    return formatter.format(record)


def test_console_formatter_colors_multiround_and_progress_lines():
    assert "\033[" in _format_message(
        "[multiround] round 2/3 start | trial_count=4",
        enable_color=True,
    )
    assert "\033[" in _format_message(
        "[progress] 1/4 done | status=failed | reward=0.000 | trial=demo",
        enable_color=True,
    )
    assert "\033[" in _format_message(
        "[trial=demo] Verification did not produce a usable reward at round 2",
        enable_color=True,
    )


def test_console_formatter_keeps_plain_text_when_color_disabled():
    message = "[multiround] round 1/3 start | trial_count=4"
    assert _format_message(message, enable_color=False) == message


def test_console_formatter_leaves_unmatched_lines_unchanged():
    message = "Running 4 trial(s) with quiet progress mode"
    assert _format_message(message, enable_color=True) == message
