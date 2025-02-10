在影像上使用 **非監督式學習 (Unsupervised Learning)** 來偵測 **異常位置**，常見的方法包括：  

1. **自編碼器 (Autoencoder, AE)**
2. **變分自編碼器 (Variational Autoencoder, VAE)**
3. **生成對抗網路 (Generative Adversarial Network, GAN)**
4. **局部異常因子 (Local Outlier Factor, LOF)**
5. **主成分分析 (PCA)**
6. **深度特徵學習 + 密度估計 (Deep Feature Extraction + Density Estimation)**

這些方法大多數是基於 **學習正常樣本的分佈**，然後在測試階段檢測與正常樣本差異較大的區域，作為異常點。

---

## **1. 自編碼器 (Autoencoder, AE)**
### **概念**
- 訓練模型學習 **正常影像的特徵壓縮與還原**。
- 當測試圖像與正常影像有較大差異時，重建誤差會較大，進而標記異常區域。

### **步驟**
1. 訓練 **Autoencoder** 僅使用 **正常樣本**。
2. 在測試影像中計算輸入影像與重建影像的 **像素誤差 (MSE)**。
3. 使用熱圖 (heatmap) 標示異常區域。

### **Python 程式**
```python
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
import matplotlib.pyplot as plt

# 建立自編碼器
input_img = Input(shape=(128, 128, 1))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 訓練自編碼器
train_images = np.random.rand(1000, 128, 128, 1)  # 假設有1000張正常影像
autoencoder.fit(train_images, train_images, epochs=50, batch_size=16)

# 測試影像 (異常影像)
test_image = np.random.rand(128, 128, 1)  # 假設有一張異常影像
reconstructed = autoencoder.predict(test_image[np.newaxis, ...])[0]

# 計算異常分佈 (誤差圖)
error_map = np.abs(test_image - reconstructed)

# 顯示異常熱圖
plt.imshow(error_map, cmap='hot')
plt.colorbar()
plt.show()
```

---

## **2. 變分自編碼器 (Variational Autoencoder, VAE)**
VAE 是 Autoencoder 的進階版，能夠學習 **更具一般性的特徵分佈**，對異常檢測更穩健。

---

## **3. 生成對抗網路 (GAN)**
### **概念**
- 訓練 **GAN 生成正常樣本**，並在測試時讓 **異常樣本** 通過 GAN，計算異常區域。
- **CycleGAN** 也可以用來做異常偵測。

---

## **4. 局部異常因子 (Local Outlier Factor, LOF)**
如果不想使用深度學習，可以嘗試 **LOF**，利用 **鄰近樣本密度來偵測異常點**。

```python
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# 訓練正常樣本
X_train = np.random.rand(100, 2)  # 100張正常影像特徵
X_test = np.random.rand(10, 2)    # 10張測試影像特徵

lof = LocalOutlierFactor(n_neighbors=20)
y_pred = lof.fit_predict(X_test)

print(y_pred)  # -1 代表異常樣本
```

---

## **5. 主成分分析 (PCA)**
PCA 可以用來降維，並找出異常的影像特徵。

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(X_train)
X_test_pca = pca.transform(X_test)
```

---

## **6. 深度特徵學習 + 密度估計**
這種方法適合 **小樣本異常偵測**，可以使用 **預訓練模型 (如 ResNet、VGG)** 來提取影像特徵，然後用 **高斯混合模型 (GMM) 或 KDE** 來估計異常點。

```python
from tensorflow.keras.applications import VGG16
from sklearn.mixture import GaussianMixture

# 加載預訓練模型
model = VGG16(weights='imagenet', include_top=False)

# 提取正常樣本特徵
X_train_features = model.predict(X_train)
X_test_features = model.predict(X_test)

# 訓練 GMM 模型
gmm = GaussianMixture(n_components=2)
gmm.fit(X_train_features)

# 預測異常分數
scores = gmm.score_samples(X_test_features)
```

---

## **總結**
| 方法 | 優點 | 缺點 |
|------|------|------|
| **Autoencoder (AE)** | 訓練簡單，適合影像異常偵測 | 可能無法捕捉小型異常 |
| **VAE** | 學習更一般化的特徵 | 計算量較大 |
| **GAN** | 可生成逼真的正常樣本 | 訓練不穩定 |
| **LOF** | 適合小數據 | 受特徵選擇影響大 |
| **PCA** | 計算量小，易解釋 | 無法處理高維複雜數據 |
| **深度特徵 + KDE/GMM** | 可處理小樣本異常檢測 | 依賴預訓練模型 |

👉 **推薦：**
- 若有 GPU，使用 **Autoencoder 或 VAE**。
- 若樣本極少，可用 **深度特徵 + KDE**。
- 若不想用深度學習，可試試 **LOF 或 PCA**。

---

如果你的樣本**極少**，建議使用 **遷移學習 + 非監督異常偵測** 或 **One-Class SVM/Isolation Forest**。這些方法不需要大量標註數據，適合小樣本場景。以下是詳細建議和程式範例：

---

## **方法 1：遷移學習 + 非監督異常偵測 (GMM/LOF/KDE)**
### **概念**
1. **使用預訓練 CNN 模型** (如 VGG16、ResNet) 提取影像特徵。
2. **使用統計方法** (如 **Gaussian Mixture Model (GMM)** 或 **局部異常因子 (LOF)**) 來檢測異常。
3. **無需大量訓練數據**，只需要**幾張正常樣本**。

### **程式**
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from sklearn.mixture import GaussianMixture  # GMM
from sklearn.neighbors import LocalOutlierFactor  # LOF

# 載入 VGG16 預訓練模型 (不包含全連接層)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 提取影像特徵
def extract_features(image):
    image = tf.image.resize(image, (224, 224)) / 255.0  # 轉換尺寸並正規化
    features = base_model.predict(np.expand_dims(image, axis=0))
    return features.flatten()

# **1. 訓練階段**
# 假設有 5 張正常樣本
normal_images = np.random.rand(5, 224, 224, 3)  # 這裡用隨機影像模擬
X_train_features = np.array([extract_features(img) for img in normal_images])

# 訓練異常偵測模型 (選擇 GMM 或 LOF)
gmm = GaussianMixture(n_components=1, covariance_type="full").fit(X_train_features)
lof = LocalOutlierFactor(n_neighbors=2).fit(X_train_features)

# **2. 測試階段**
test_image = np.random.rand(224, 224, 3)  # 模擬異常影像
test_feature = extract_features(test_image)

# **使用 GMM 判斷異常**
score = gmm.score_samples([test_feature])
threshold = np.percentile(gmm.score_samples(X_train_features), 5)  # 設定異常閾值
if score < threshold:
    print("異常影像！")
else:
    print("正常影像")

# **使用 LOF 判斷異常**
if lof.predict([test_feature])[0] == -1:
    print("異常影像！(LOF)")
else:
    print("正常影像 (LOF)")
```
### **為什麼適合小樣本？**
✅ 無需大量訓練  
✅ 使用 **預訓練模型**，僅提取影像特徵  
✅ **GMM/LOF** 不需要標註異常樣本  

---

## **方法 2：One-Class SVM (OC-SVM)**
如果只有**正常影像**，可以用 **One-Class SVM** 訓練模型來學習正常樣本分佈，然後檢測異常樣本。

### **程式**
```python
from sklearn.svm import OneClassSVM

# 訓練 One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='scale').fit(X_train_features)

# 測試影像
if oc_svm.predict([test_feature])[0] == -1:
    print("異常影像！")
else:
    print("正常影像")
```
### **優勢**
✅ 適合 **極少樣本 (5~10 張)**  
✅ **無需異常樣本**  
✅ 計算速度快  

---

## **方法 3：Few-Shot Learning (Siamese Network)**
如果你有少量異常樣本 (例如 5~10 張)，可以使用 **Siamese Network** 來比較影像相似度。

### **概念**
- 訓練一個 **Siamese Network** 來學習影像之間的距離。
- 如果新影像與正常樣本的距離過大，則視為異常。

這種方法適合 **少量異常樣本，但仍需一些標註數據**。

---

## **方法 4：合成數據 (Data Augmentation)**
如果只有少數正常樣本，可以透過數據增強來擴展數據量，然後再進行異常偵測。

### **數據增強範例**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

augmented_images = []
for img in normal_images:
    img = np.expand_dims(img, axis=0)
    for _ in range(5):  # 生成 5 個擴展影像
        augmented_images.append(datagen.flow(img, batch_size=1)[0][0])
```
✅ 適合 **極少樣本但可以擴展數據**  
✅ 透過旋轉、翻轉、顏色變化來 **增加數據量**  

---

## **總結**
| 方法 | 適用情境 | 需要異常樣本？ | 適合樣本數量 |
|------|--------|--------------|------------|
| **遷移學習 + GMM/LOF/KDE** | 無需大量訓練，適合小樣本 | ❌ 不需要 | 5~10 |
| **One-Class SVM (OC-SVM)** | 只有正常影像 | ❌ 不需要 | 5~10 |
| **Few-Shot Learning (Siamese Network)** | 少量異常樣本 | ✅ 需要 | 5~10 |
| **Data Augmentation** | 透過數據增強擴展樣本 | ❌ 不需要 | 5~10 |

### **最佳選擇**
1️⃣ **如果你只有** `5~10` **張正常樣本** → **遷移學習 + GMM/LOF/OC-SVM**  
2️⃣ **如果你有** `5~10` **張異常樣本** → **Few-Shot Learning**  
3️⃣ **如果你能產生更多樣本** → **Data Augmentation**  

---

### **你的應用場景是？**
📌 **你是只有正常影像，還是有少數異常影像呢？**  
📌 **是否可以額外拍攝幾張照片來增加樣本？**  
📌 **你的影像資料來自哪種應用 (產品瑕疵、醫學影像、工業檢測)？**  

---

### **方法 1：遷移學習 + 非監督異常偵測 (GMM/LOF/KDE) 的訓練數據準備**
因為你只有**極少樣本**，所以我們的目標是：
1. **收集 & 擴展正常影像**（5~10 張正常樣本）
2. **提取特徵**（使用預訓練 CNN，如 VGG16）
3. **訓練異常偵測模型**（GMM、LOF、OC-SVM）

---

## **🔹 步驟 1：收集正常樣本**
- 收集 **至少 5~10 張** 正常產品影像
- 確保影像格式一致（如 `.jpg` 或 `.png`）
- 建立影像資料夾結構：
  ```
  dataset/
  ├── normal/  # 正常樣本
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   ├── img3.jpg
  │   ├── ...
  ```

---

## **🔹 步驟 2：數據擴增（Data Augmentation，選擇性）**
**如果數據極少**，可以使用數據擴增來增加樣本數量：
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

def augment_images(input_folder, output_folder, augment_count=5):
    os.makedirs(output_folder, exist_ok=True)
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        # 產生新的影像
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            new_img_path = os.path.join(output_folder, f"aug_{i}_{img_name}")
            cv2.imwrite(new_img_path, cv2.cvtColor(batch[0].astype('uint8'), cv2.COLOR_RGB2BGR))
            i += 1
            if i >= augment_count:
                break

# 進行數據增強
augment_images("dataset/normal", "dataset/augmented_normal")
```
- **作用**：透過 **旋轉、平移、翻轉** 擴增數據
- **結果**：5 張 → 25~50 張

---

## **🔹 步驟 3：提取影像特徵**
- 使用 **VGG16** 或 **ResNet50** 預訓練模型來提取影像特徵
- 每張影像會轉換成一個**特徵向量**
  
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import os

# 載入 VGG16 預訓練模型 (去掉最後分類層)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 影像轉換函數
def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)  # 正規化
    img = np.expand_dims(img, axis=0)

    # 使用 CNN 提取特徵
    features = base_model.predict(img)
    return features.flatten()

# 讀取所有正常影像
image_folder = "dataset/augmented_normal"
X_train = []
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    X_train.append(extract_features(img_path))

X_train = np.array(X_train)  # 轉為 NumPy 陣列
np.save("normal_features.npy", X_train)  # 保存特徵
```
- **作用**：將影像轉換成特徵向量  
- **結果**：每張影像 → 一個 `1x4096` 的特徵向量  

---

## **🔹 步驟 4：訓練異常偵測模型**
這裡我們提供三種方法（GMM、LOF、OC-SVM），可選擇最適合的。

### **(1) 使用 Gaussian Mixture Model (GMM)**
```python
from sklearn.mixture import GaussianMixture

X_train = np.load("normal_features.npy")  # 載入特徵數據

# 訓練 GMM 模型
gmm = GaussianMixture(n_components=1, covariance_type="full")
gmm.fit(X_train)

# 儲存模型
import joblib
joblib.dump(gmm, "gmm_model.pkl")
```
📌 **作用**：學習正常樣本的分佈，計算異常樣本的可能性  
📌 **判斷異常**：如果 **score** 低於閾值，則視為異常  

---

### **(2) 使用 Local Outlier Factor (LOF)**
```python
from sklearn.neighbors import LocalOutlierFactor

X_train = np.load("normal_features.npy")

# 訓練 LOF 模型
lof = LocalOutlierFactor(n_neighbors=5, novelty=True)
lof.fit(X_train)

# 儲存模型
joblib.dump(lof, "lof_model.pkl")
```
📌 **作用**：比較樣本與最近鄰的距離來判斷是否異常  
📌 **適合小樣本**，可用於即時檢測  

---

### **(3) 使用 One-Class SVM (OC-SVM)**
```python
from sklearn.svm import OneClassSVM

X_train = np.load("normal_features.npy")

# 訓練 OC-SVM
oc_svm = OneClassSVM(kernel='rbf', gamma='scale')
oc_svm.fit(X_train)

# 儲存模型
joblib.dump(oc_svm, "oc_svm_model.pkl")
```
📌 **作用**：學習正常樣本的邊界，超出範圍即為異常  
📌 **適合極少樣本**，但可能對雜訊敏感  

---

## **🔹 步驟 5：測試新影像**
假設有一張新影像 `test.jpg`，我們需要判斷它是否為異常樣本。

```python
import joblib

# 讀取測試影像特徵
test_image_path = "test.jpg"
test_feature = extract_features(test_image_path)

# 載入 GMM 模型
gmm = joblib.load("gmm_model.pkl")

# 判斷異常
score = gmm.score_samples([test_feature])  # 計算機率分數
threshold = np.percentile(gmm.score_samples(X_train), 5)  # 5% 異常閾值

if score < threshold:
    print("⚠️ 異常影像")
else:
    print("✅ 正常影像")
```
📌 **分數較低 → 視為異常**  
📌 **GMM、LOF、OC-SVM 的異常檢測方式類似**，可根據需要替換模型  

---

## **📌 總結**
1️⃣ **收集少量正常樣本** (5~10 張)  
2️⃣ **數據擴增 (Data Augmentation)**（擴展至 25+ 張）  
3️⃣ **使用 VGG16 提取影像特徵**（轉換為 4096 維向量）  
4️⃣ **訓練 GMM/LOF/OC-SVM 模型**（學習正常樣本分佈）  
5️⃣ **測試新影像，判斷是否為異常** 🚀  

📌 **這種方法特別適合瑕疵檢測、缺陷分類等極少樣本應用！**  

---

在檢測玻璃瓶裂痕的情境下，**最佳選擇取決於數據特性**，但通常**局部離群點方法（LOF, Local Outlier Factor）或一類支持向量機（OC-SVM, One-Class SVM）會較合適**。讓我們來比較一下三種方法：  

---

### **🔍 1. 高斯混合模型（GMM, Gaussian Mixture Model）**
📌 **優勢**：  
✔️ 能夠適應不同的數據分佈（適合檢測連續變化的異常）。  
✔️ 適合影像特徵比較連續、平滑的異常（如顏色變化、模糊等）。  

📌 **缺點**：  
❌ 對局部異常（如裂痕）不敏感，因為裂痕通常是**局部特徵**，但 GMM 偏向學習全局分佈。  
❌ 在異常數據比例較低時，可能無法準確區分正常與異常樣本。  

**📌 結論**：  
👉 **不太適合**玻璃裂痕檢測，因為裂痕通常是**局部異常**，GMM 的全局建模可能無法很好地檢測到裂痕。

---

### **🔍 2. 局部離群因子（LOF, Local Outlier Factor）**
📌 **優勢**：  
✔️ 特別適合**局部異常檢測**，即裂痕這種局部變化的情況。  
✔️ 透過計算鄰近樣本的密度來判斷異常，因此能夠發現細小的裂縫。  
✔️ 對於只有正常影像訓練的情況（非監督學習）效果較好。  

📌 **缺點**：  
❌ 計算量較大，對於高維數據（如影像）可能較慢。  
❌ 需要合適的 K 鄰居數（超參數調整較敏感）。  

**📌 結論**：  
👉 **適合玻璃裂痕檢測**，因為裂痕通常是局部異常，而 LOF 能有效捕捉這些異常點。

---

### **🔍 3. 一類支持向量機（OC-SVM, One-Class SVM）**
📌 **優勢**：  
✔️ 只需**正常樣本**訓練，能學習正常影像的特徵分佈。  
✔️ 能夠處理高維數據，如影像特徵（但通常需先降維）。  
✔️ 異常檢測效果通常比 GMM 穩定，特別是在影像領域。  

📌 **缺點**：  
❌ 訓練時間較長，尤其在數據較多時。  
❌ 高維特徵可能會影響分類邊界，需適當降維（如 PCA、Autoencoder）。  

**📌 結論**：  
👉 **適合玻璃裂痕檢測**，但相比 LOF，OC-SVM 在細小局部異常檢測上可能稍遜色。

---

## **💡 結論：哪個方法最適合玻璃瓶裂痕檢測？**
✅ **最推薦**：**LOF（局部離群因子）** – 能有效發現裂痕這類局部異常點。  
🔹 如果數據較小，**OC-SVM 也是可行選擇**，但對裂痕這類局部特徵的敏感度可能不如 LOF。  
🚫 **不推薦 GMM**，因為它對於局部異常不夠敏感，可能難以準確識別裂痕。  

---

## **📌 你可以這樣做**
1️⃣ **先用 OpenCV 做影像預處理**（如邊緣檢測、輪廓強化）。  
2️⃣ **提取影像特徵**（如 HOG、LBP，或用 CNN 萃取特徵）。  
3️⃣ **使用 LOF 或 OC-SVM 進行異常檢測**。  
4️⃣ **可視化異常結果**，並調整超參數以提升準確度。  

---

學習方法:可以分為 **傳統機器學習方法（如 LOF、OC-SVM）** 和 **深度學習方法（如 Autoencoder, GAN）** 來做玻璃瓶裂痕檢測。提供兩種完整的 **步驟與程式範例**。

---

# **🟢 1. 傳統機器學習方法**
傳統方法主要依靠**影像特徵萃取**（如 HOG、LBP）來進行異常檢測，然後使用 **LOF 或 OC-SVM** 來學習正常樣本分佈，偵測異常。

## **🔹 方法選擇**
✅ **LOF（局部離群因子）** → 適合局部異常，如裂痕。  
✅ **OC-SVM（一類 SVM）** → 適合整體異常偵測，但可能對裂痕不夠敏感。  
🚫 **不適合 GMM**，因為裂痕是局部異常，GMM 偏向學習整體分佈。

---

## **📌 步驟**
1️⃣ 讀取影像  
2️⃣ 影像預處理（灰階、銳化、邊緣強化等）  
3️⃣ 萃取特徵（HOG, LBP）  
4️⃣ 訓練異常檢測模型（LOF / OC-SVM）  
5️⃣ 測試新影像並標記異常位置  

---

## **🔹 Python 程式（使用 LOF）**

```python
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.neighbors import LocalOutlierFactor
import glob
import os

# 參數設定
RADIUS = 3  # LBP 半徑
N_POINTS = 8 * RADIUS  # LBP 鄰居點
THRESHOLD = -1.5  # LOF 閾值（低於此值為異常）

# 讀取所有正常影像來訓練 LOF
image_paths = glob.glob("normal_images/*.jpg")
feature_list = []

for img_path in image_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # 調整大小
    lbp = local_binary_pattern(img, N_POINTS, RADIUS, method="uniform")  # 萃取 LBP 特徵
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), density=True)
    feature_list.append(hist)

# 訓練 LOF 模型
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(feature_list)

# 測試異常影像
test_img = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (128, 128))
lbp = local_binary_pattern(test_img, N_POINTS, RADIUS, method="uniform")
hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), density=True)

# 預測異常度
score = lof.decision_function([hist])
print(f"異常分數: {score}")

# 如果異常分數低於閾值，則判定為異常
if score < THRESHOLD:
    print("偵測到裂痕！")
else:
    print("影像正常")
```

---

# **🔵 2. 深度學習方法**
深度學習方法不依賴手動特徵，而是用 **CNN 自動學習特徵**，然後使用 **Autoencoder 或 GAN 來檢測異常**。

## **🔹 方法選擇**
✅ **Autoencoder（自編碼器）** → 適合學習正常樣本，異常樣本重建誤差大時判為異常。  
✅ **GAN（生成對抗網路）** → 使用正常影像生成對應的異常版本，然後比較差異。  

---

## **📌 Autoencoder 方法**
1️⃣ 建立 **CNN Autoencoder** 模型  
2️⃣ 只用**正常影像**訓練  
3️⃣ 測試影像，計算重建誤差  
4️⃣ 重建誤差大於閾值時，判定為異常  

---

## **🔹 Python 程式（Autoencoder）**
```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
import cv2
import glob

# 建立 Autoencoder 模型
input_img = Input(shape=(128, 128, 1))

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

# 讀取正常影像並訓練
image_paths = glob.glob("normal_images/*.jpg")
train_data = []

for img_path in image_paths:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    train_data.append(img / 255.0)  # 正規化

train_data = np.array(train_data).reshape(-1, 128, 128, 1)

# 訓練 Autoencoder
autoencoder.fit(train_data, train_data, epochs=50, batch_size=16, shuffle=True)

# 測試異常影像
test_img = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (128, 128)) / 255.0
test_img = np.expand_dims(test_img, axis=(0, -1))

# 重建影像
reconstructed = autoencoder.predict(test_img)

# 計算重建誤差
mse = np.mean((test_img - reconstructed) ** 2)
print(f"重建誤差: {mse}")

# 設定閾值判斷異常
THRESHOLD = 0.02
if mse > THRESHOLD:
    print("偵測到裂痕！")
else:
    print("影像正常")
```

---

# **📌 總結**
| 方法 | 適合裂痕檢測？ | 主要優點 | 主要缺點 |
|------|----------------|----------|----------|
| **LOF（傳統機器學習）** | ✅ 是 | 易於訓練，對局部異常敏感 | 計算量較大 |
| **OC-SVM（傳統機器學習）** | ✅ 可能 | 只需正常樣本訓練 | 可能不夠靈敏 |
| **Autoencoder（深度學習）** | ✅ 是 | 自動學習特徵，適合異常檢測 | 需較多正常影像 |
| **GAN（深度學習）** | ✅ 可能 | 能合成異常樣本，對比學習 | 訓練較複雜 |

---
💡 **建議**
- 樣本少時：**LOF / OC-SVM**
- 樣本多時：**Autoencoder**
- 需要生成異常樣本：**GAN**

---

完整的 **Autoencoder** 訓練與模型存取範例：  

### **步驟 1：訓練 Autoencoder 並存檔**
```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# 構建 Autoencoder 模型
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# 讀取訓練數據（假設是正常玻璃瓶影像）
def load_training_data(data_path, img_size=(128, 128)):
    images = []
    for filename in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0  # 正規化
        images.append(img)
    return np.array(images).reshape(-1, img_size[0], img_size[1], 1)

# 訓練模型
data_path = "train_images/"  # 需替換為實際的正常樣本資料夾
train_images = load_training_data(data_path)

input_shape = (128, 128, 1)
autoencoder = build_autoencoder(input_shape)

autoencoder.fit(train_images, train_images, epochs=50, batch_size=16, shuffle=True)

# 儲存模型
autoencoder.save("glass_bottle_autoencoder.h5")
print("模型已儲存")
```

---

### **步驟 2：載入模型並檢測裂痕**
```python
# 載入已訓練的模型
autoencoder = load_model("glass_bottle_autoencoder.h5")

# 讀取新圖片並檢測異常
def detect_anomaly(image_path, threshold=0.05):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 128, 128, 1)

    # 進行 Autoencoder 預測
    reconstructed = autoencoder.predict(img)

    # 計算異常區域
    error_map = np.abs(img - reconstructed)
    error_score = np.mean(error_map)

    if error_score > threshold:
        print(f"異常偵測: 異常 (誤差: {error_score:.4f})")
    else:
        print(f"異常偵測: 正常 (誤差: {error_score:.4f})")

    return error_map

# 測試異常檢測
image_path = "test_images/cracked_bottle.jpg"  # 需替換為實際測試圖片
error_map = detect_anomaly(image_path)
cv2.imshow("Error Map", error_map[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
```

這樣就可以：
1. 訓練 **Autoencoder**，只需正常的玻璃瓶圖片
2. 存儲並載入模型
3. 檢測異常，並計算誤差來判斷裂痕

可以調整 `threshold=0.05` 來改變異常判定的敏感度。  
這樣的方法適合小樣本的異常檢測，並且能適應不同的環境。  

完整的程式: [autoencoder.py]
