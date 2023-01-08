import os
import sys
import logging
import numpy as np

sys.path.append('./python')

logger = logging.getLogger(__name__)

import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd

def main():
    print(f"{os.getpid()=}")

    device = ndl.cpu()
   
    print("hello, world!")

if __name__ == '__main__':
    main()