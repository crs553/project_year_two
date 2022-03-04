import logging


# change to false here to disable debug mode
debug_mode: bool = True


def start_logging() -> None:
    """Configures the logging system. Only call this once during program
    execution!

    In each module, create a new Logger object using:

    log: Logger = logging.getLogger(__name__)

    You can then write to the log using:

    log.LEVEL(string)

    """

    if debug_mode:
        logging.basicConfig(
            level=logging.DEBUG,
            filename="log.txt",
            filemode="a",
            format="%(asctime)s %(levelname)s :: %(message)s",
            datefmt="%d-%b-%y %H-%M-%S"
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            filename="log.txt",
            filemode="a",
            format="%(asctime)s %(levelname)s :: %(message)s",
            datefmt="%d-%b-%y %H-%M-%S"
        )

    log: Logger = logging.getLogger(__name__)
    log.info("started logging system")
    print("[info] started logging system")

    if debug_mode:
        log.debug("debugging mode enabled")
        print("[debug] debugging mode enabled")
