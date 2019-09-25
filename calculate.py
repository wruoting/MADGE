from classification_set_n import ClassificationSetN
from n_point import NPoint
import gzip
import numpy as np
import ast

a = NPoint((1, 2, 3), type=1)
b = NPoint((1, 2, 4), type=0)
d = NPoint((1, 2, 5), type=1)
c = ClassificationSetN()
c.add_point(a)
c.add_point(b)

print(c.calculate_madge_data_and_map_to_point(d))


path = './SampleData/train-images-idx3-ubyte.gz'
f = gzip.open(path, 'r')

image_size = 28
num_images = 5
#https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python
f.read(1)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
print(data.shape)
data = data.reshape(num_images, image_size, image_size, 1)

import matplotlib.pyplot as plt
image = np.asarray(data[2]).squeeze()
plt.imshow(image)
plt.show()