import numpy as np
myarray = np.empty(shape=67, dtype=object)

for i in range(len(myarray)):
    myarray[i] = np.random.rand(26)

print(myarray.shape, myarray[0].shape, myarray.dtype, type(myarray))

x = np.stack(myarray)
print(type(x), x.shape)