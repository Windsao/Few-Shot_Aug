import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

folder = './VPT_adv_un/'
y_pred = np.load(folder + 'oxford_pets-FS_2_8_y_pred.npy')
y_true = np.load(folder + 'oxford_pets-FS_2_8_y_true.npy')
y_true = (np.argmax(y_true, axis=1))

# print(y_true.shape)
plt.hist(y_pred, bins=37, rwidth=0.7)
plt.savefig(folder + 'train_pets_2_8_pred')
plt.close()
plt.hist(y_true, bins=37, rwidth=0.7)
plt.savefig(folder + 'train_pets_2_8_true')
plt.close()

cf_matrix = confusion_matrix(y_true, y_pred)
plt.imshow(cf_matrix)
plt.savefig(folder + 'cf_mat')
plt.close()
# print(cf_matrix)
# print(y_pred[0:200], y_true[0:200])