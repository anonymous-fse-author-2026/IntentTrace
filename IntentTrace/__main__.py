import asyncio
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
