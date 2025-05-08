from burger_lib.guezzi import *
import numpy as np

a = Measurement([1,2,3,4,5,6],1,name='bob')
b = Measurement([1,2,4,4,5,6],1)
#print(a.shape, a.size, a.ndim, a.variance, len(a))
print(a.to_eng_string())
print(a)
print(Measurement(1, 3))
print(b==a)