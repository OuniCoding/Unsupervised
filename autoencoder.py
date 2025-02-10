import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
import cv2
import glob
import os

THRESHOLD = 0.02 # 低於此值為異常
# 建立 Autoencoder模型
def bulid_autoencoder(input_shape):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

# 讀取正常圖片並訓練

def load_training_data(data_path, img_size=(128, 128)):
    train_data = []

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size) # 調整大小
        img = img.astype('float32') / 255.0
        train_data.append(img) # 正規畫

    train_data = np.array(train_data).reshape(-1, img_size[0], img_size[1], 1)

    return train_data

# 測試異常影像
def detect_anomaly(test_path, img_size=(128, 128), threshold=0.05):
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(test_img, (img_size[0], img_size[1]))
    test_img = test_img.astype('float32') / 255.0
    # test_img = np.expand_dims(test_img, axis=(0, -1))
    test_img = test_img.reshape(1, img_size[0], img_size[1], 1)

    # 重建影像
    reconstructed = autoencoder.predict(test_img)

    # 計算重建誤差
    error_map = np.abs(test_img - reconstructed)
    # mse = np.mean((test_img - reconstructed) ** 2)
    mse = np.mean(error_map)
    print(f'{test_path}的重建誤差: {mse}', end=' ')

    # 如果分數低於閥值,判定為異常
    if mse > threshold:
        print('偵測道裂痕')
    else:
        print('影像正常')

    return error_map


image_paths = glob.glob('F:/project/python/vgg16/image_normal/*.jpg')
image_shape = (128, 128, 1)
if not os.path.exists('glass_bottle.h5'):
    train_datas = load_training_data(image_paths, (image_shape[0],image_shape[1]))
    # 訓練 Autoencoder
    autoencoder = bulid_autoencoder(image_shape)

    autoencoder.fit(train_datas, train_datas, epochs=50, batch_size=16, shuffle=True)
    # 模型儲存
    autoencoder.save('glass_bottle.h5')
    print('訓練完成,並儲存為glass_bottle.h5')

# 載入模型
autoencoder = load_model('glass_bottle.h5')

test_paths = glob.glob(r'E:\logo\trans\2025-01-10\1\resultG\*.jpg')
for test_path in test_paths:
    error_map = detect_anomaly(test_path, (image_shape[0],image_shape[1]), threshold=THRESHOLD)
    cv2.imshow('Error Map', cv2.resize(error_map[0], (512, 512)))
    key = cv2.waitKey(100)
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


