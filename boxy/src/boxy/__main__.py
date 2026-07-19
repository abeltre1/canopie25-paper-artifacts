"""Enable `python -m boxy` as an alias for the `boxy` console script."""

import sys

from boxy.cli import main

if __name__ == "__main__":
    sys.exit(main())
