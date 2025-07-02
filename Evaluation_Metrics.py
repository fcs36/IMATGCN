import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def sigma_area(y_pred, score, n):
    y_mean = np.mean(score)
    y_std = np.std(score)
    up = y_mean + n*y_std
    down = y_mean - n*y_std
    up_area = y_pred+n*y_std
    down_area = y_pred-n*y_std
    return up_area, down_area

y1 = np.load('result/y_pred.npy')
y0 = np.load('result/y_real.npy')

y_pred = y1[0]
y_test = y0[0]



score = y_pred.flatten() - y_test.flatten()
up_area, down_area = sigma_area(y_pred, score, 1)
np.save('result/up.npy',up_area)
np.save('result/down.npy',down_area)

outlier_x = []
outlier_y = []


ano_label = np.load('data/ano_label.npy')


for i in range(y_pred.shape[0]):
    if y_test[i] < down_area[i] or y_test[i] > up_area[i]:
        outlier_x.append(i)
        outlier_y.append(y_test[i])


outlier_x = np.array(outlier_x)
outlier_y = np.array(outlier_y)
deteced = np.zeros(y_test.shape[0])
deteced[outlier_x.astype('int64')] = 1
true_label = ano_label

confusion = confusion_matrix(true_label, deteced, labels=[1, 0])
FPR, FTR, threshold = roc_curve(true_label, deteced)
aroc_auc = auc(FPR, FTR)
acc = (confusion[0, 0] + confusion[1, 1]) / (
        confusion[1, 0] + confusion[1, 1] + confusion[0, 0] + confusion[0, 1])
fp_rate = (confusion[1, 0]) / (confusion[1, 0] + confusion[1, 1])
miss_rate = (confusion[0, 1]) / (confusion[0, 1] + confusion[0, 0])
af1 = f1_score(true_label, deteced)
arec_rate = (confusion[0, 0]) / (confusion[0, 0] + confusion[1, 0])
apre_rate = (confusion[0, 0]) / (confusion[0, 0] + confusion[0, 1])


