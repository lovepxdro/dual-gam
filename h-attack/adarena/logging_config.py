from __future__ import annotations

import logging

from rich.logging import RichHandler


_HANDLER_MARKER = "_adarena_console_handler"


def configure_console_logging(
    verbose: bool = False,
) -> None:
    """
    Configura somente a saída do terminal.

    O nível global permanece em DEBUG para permitir que
    handlers de arquivo persistam o log completo. O handler
    do terminal decide quanto o usuário vê.
    """

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    for handler in list(root.handlers):
        if getattr(
            handler,
            _HANDLER_MARKER,
            False,
        ):
            root.removeHandler(handler)
            handler.close()

    console = RichHandler(
        level=(
            logging.DEBUG
            if verbose
            else logging.INFO
        ),
        show_time=verbose,
        show_level=verbose,
        show_path=False,
        rich_tracebacks=verbose,
        markup=False,
    )

    setattr(
        console,
        _HANDLER_MARKER,
        True,
    )

    console.setFormatter(
        logging.Formatter(
            "%(name)s — %(message)s"
            if verbose
            else "%(message)s"
        )
    )

    root.addHandler(console)
