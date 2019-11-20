# 96 x 96 per color in column major order
# Three colors per image
# Column major order:

# | 1  2  3 |
# | 4  5  6 |
# | 7  8  9 |
# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
#  |   1   |   4   |   7   |   2   |   5   |   8   |   3   |   6   |   9   |
# -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

import numpy as np

n = 97 * 97

for column in n:
    