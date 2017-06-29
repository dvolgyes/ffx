#!/usr/bin/env python

import numpy as np
import ffx

# This creates a dataset of 1 predictor
train_X = np.array([[-5,-4,-3,-2,-1, 0, 1, 2, 3,4,5,6,7,8,9]]).T
train_y = np.array([4,4,3,2,1, 0, 1, 2, 3,4,4,4,4,4,4])
test_X = np.array([[10,11,12]]).T
test_y = np.array([4, 4,4,])

models = ffx.run(train_X, train_y, test_X, test_y, ["x"],verbose = True)

print('True model: y = min(abs(x),4)')
print('Results:')
print('Num bases,Test error (%),Model\n')
for model in models:
    print('%10s, %13s, %s\n' %
          ('%d' % model.numBases(), '%.4f' % (model.test_nmse * 100.0), model))
