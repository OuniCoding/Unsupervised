'''
傳統機器學習 LOF
'''
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.neighbors import LocalOutlierFactor
import glob
import os

# 參數設定
RADIUS = 2 # LBP 半徑
N_POINTS = 8 * RADIUS # LBP 鄰居點
THRESHOLD = -1.5 # LOF 閥值(低於此值為異常)

# 讀取所有正常影像來訓練 LOF
image_paths = glob.glob('F:/project/python/vgg16/image_normal/*.jpg')
feature_list = []

for img_path in image_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (416, 416)) # 調整大小
    lbp = local_binary_pattern(img, N_POINTS, RADIUS, method='uniform') # 萃取 LBP 特徵
    # local_binary_pattern(image, P, R, method='default')
    # P：每個像素周圍的 取樣點數（通常設為 8 * R)
    # R：半徑（決定取樣點距離）
    # method：計算 LBP 的方法，常見選擇：'default'：經典 LBP/'ror'：旋轉不變 LBP（Rotation Invariant）/'uniform'：均勻模式 LBP/'var'：變異數 LBP
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), density=True)
    feature_list.append(hist)

# 訓練 LOF 模型
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(feature_list)

# 測試異常影像
test_paths = glob.glob(r'E:\logo\trans\2025-01-10\1\resultNG/*.jpg')
for test_path in test_paths:
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(test_img, (416, 416)) # 調整大小
    lbp = local_binary_pattern(test_img, N_POINTS, RADIUS, method='uniform') # 萃取 LBP 特徵
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), density=True)

    # 預測異常程度
    score = lof.decision_function([hist])
    print(f'{test_path}的異常分數:{score}', end=" ")

    # 如果異常分數低於閥值,判定為異常
    if score < THRESHOLD:
        print('偵測道裂痕')
    else:
        print('影像正常')