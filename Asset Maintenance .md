```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
```


```python
DATADIR = "../Documents/Asset Maintenance"
CATEGORIES = ["Rust","Norust"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(img_array)
        plt.show()
        break
    break
```


    
![output_1_0](https://github.com/user-attachments/assets/f8e4124a-e48e-4b23-9cfe-eb85cbb82aec)

    



```python
img_array
```




    array([[[142, 158, 164],
            [143, 159, 165],
            [141, 160, 165],
            ...,
            [255, 255, 255],
            [255, 254, 255],
            [254, 253, 255]],
    
           [[142, 158, 164],
            [143, 159, 165],
            [141, 160, 165],
            ...,
            [255, 255, 255],
            [255, 254, 255],
            [255, 254, 255]],
    
           [[142, 158, 164],
            [142, 158, 164],
            [141, 160, 165],
            ...,
            [254, 254, 254],
            [255, 255, 255],
            [254, 254, 254]],
    
           ...,
    
           [[188, 244, 201],
            [188, 244, 201],
            [188, 244, 201],
            ...,
            [245, 253, 252],
            [245, 253, 252],
            [245, 253, 252]],
    
           [[189, 245, 202],
            [189, 245, 202],
            [188, 244, 201],
            ...,
            [245, 253, 252],
            [245, 253, 252],
            [245, 253, 252]],
    
           [[190, 246, 203],
            [189, 245, 202],
            [188, 244, 201],
            ...,
            [245, 253, 252],
            [245, 253, 252],
            [245, 253, 252]]], dtype=uint8)




```python
img_array.shape
```




    (4624, 3472, 3)




```python
IMG_SIZE =100
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array,'gray')
plt.show()
```


    
![png](output_4_0.png)
    



```python
training_data=[]
def create_training_data(DIR,lis):
    for category in CATEGORIES:
        path = os.path.join(DIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):

            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                lis.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data(DATADIR,training_data)
```


```python
print(len(training_data))
```

    760
    


```python
import random
random.shuffle(training_data)
```


```python
for sample in training_data[:10]:
    if sample[1] == 1:
        print("No rust")
    elif sample[1] == 0:
        print("Rust")
```

    Rust
    Rust
    Rust
    No rust
    Rust
    Rust
    No rust
    Rust
    No rust
    No rust
    


```python
X_t=[]
y_t=[]
```


```python
for features, label in training_data:
    X_t.append(features)
    y_t.append(label)
r = np.array(X_t).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array(y_t).reshape(-1,1)
```


```python
r.shape
```




    (760, 100, 100, 3)




```python
import pickle
pickle_out = open("X.pickle","wb")
pickle.dump(r,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
```


```python
r[1]
```




    array([[[246, 192,  98],
            [243, 192, 100],
            [241, 191, 108],
            ...,
            [247, 194, 103],
            [236, 191, 110],
            [240, 188,  95]],
    
           [[243, 190,  93],
            [245, 192,  95],
            [249, 193,  94],
            ...,
            [248, 194,  98],
            [236, 193, 120],
            [247, 196, 104]],
    
           [[247, 192,  93],
            [249, 192,  93],
            [247, 190,  91],
            ...,
            [250, 195, 104],
            [247, 195, 100],
            [247, 193,  99]],
    
           ...,
    
           [[254, 201,  91],
            [253, 200,  89],
            [251, 200,  90],
            ...,
            [234, 199, 113],
            [233, 197, 109],
            [233, 199, 110]],
    
           [[255, 202,  92],
            [254, 201,  91],
            [252, 200,  93],
            ...,
            [232, 194, 105],
            [232, 196, 108],
            [234, 196, 103]],
    
           [[255, 200,  89],
            [254, 200,  93],
            [252, 200,  93],
            ...,
            [243, 198, 101],
            [241, 198, 105],
            [237, 197, 104]]], dtype=uint8)




```python
import PIL
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
```


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
```


```python
from tensorflow import keras as kf
```


```python
r_1 = r/255.0
```


```python
resnet_model = Sequential()
pretrained_model = tf.keras.applications.ResNet50(include_top=False,input_shape=(100,100,3),pooling='avg',classes=2,weights='imagenet')
for layer in pretrained_model.layers:
    layer.trainable = False
resnet_model.add(pretrained_model)
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    [1m94765736/94765736[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m15s[0m 0us/step
    


```python
resnet_model.add(Flatten())
resnet_model.add(Dense(512,activation='relu'))
resnet_model.add(Dense(1,activation = 'sigmoid'))
```


```python
resnet_model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ resnet50 (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)                │      <span style="color: #00af00; text-decoration-color: #00af00">23,587,712</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)                │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)                 │       <span style="color: #00af00; text-decoration-color: #00af00">1,049,088</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">513</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">24,637,313</span> (93.98 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,049,601</span> (4.00 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,587,712</span> (89.98 MB)
</pre>




```python
resnet_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
```


```python
history = resnet_model.fit(r_1,y,batch_size=10,validation_split=0.1,epochs=100)
```

    Epoch 1/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m22s[0m 184ms/step - accuracy: 0.5295 - loss: 1.0125 - val_accuracy: 0.6579 - val_loss: 0.6453
    Epoch 2/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 156ms/step - accuracy: 0.5739 - loss: 0.6677 - val_accuracy: 0.5921 - val_loss: 0.6418
    Epoch 3/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 159ms/step - accuracy: 0.5894 - loss: 0.6306 - val_accuracy: 0.6842 - val_loss: 0.5834
    Epoch 4/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 163ms/step - accuracy: 0.6684 - loss: 0.6091 - val_accuracy: 0.6974 - val_loss: 0.5927
    Epoch 5/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 161ms/step - accuracy: 0.6818 - loss: 0.5905 - val_accuracy: 0.7500 - val_loss: 0.5672
    Epoch 6/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 162ms/step - accuracy: 0.7059 - loss: 0.5896 - val_accuracy: 0.6842 - val_loss: 0.5963
    Epoch 7/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 163ms/step - accuracy: 0.7178 - loss: 0.5718 - val_accuracy: 0.6974 - val_loss: 0.5492
    Epoch 8/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 164ms/step - accuracy: 0.6917 - loss: 0.5912 - val_accuracy: 0.6053 - val_loss: 0.6276
    Epoch 9/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 165ms/step - accuracy: 0.6990 - loss: 0.5866 - val_accuracy: 0.8026 - val_loss: 0.5165
    Epoch 10/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 166ms/step - accuracy: 0.7597 - loss: 0.5164 - val_accuracy: 0.6711 - val_loss: 0.5447
    Epoch 11/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 166ms/step - accuracy: 0.7179 - loss: 0.5457 - val_accuracy: 0.7500 - val_loss: 0.5472
    Epoch 12/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 167ms/step - accuracy: 0.7744 - loss: 0.4892 - val_accuracy: 0.7368 - val_loss: 0.5095
    Epoch 13/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 168ms/step - accuracy: 0.7630 - loss: 0.5141 - val_accuracy: 0.7105 - val_loss: 0.5675
    Epoch 14/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 168ms/step - accuracy: 0.7791 - loss: 0.4791 - val_accuracy: 0.6579 - val_loss: 0.5986
    Epoch 15/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 170ms/step - accuracy: 0.7712 - loss: 0.5035 - val_accuracy: 0.7895 - val_loss: 0.4925
    Epoch 16/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 169ms/step - accuracy: 0.7522 - loss: 0.5128 - val_accuracy: 0.8026 - val_loss: 0.4995
    Epoch 17/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 171ms/step - accuracy: 0.7488 - loss: 0.5172 - val_accuracy: 0.7895 - val_loss: 0.4958
    Epoch 18/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 170ms/step - accuracy: 0.7302 - loss: 0.5223 - val_accuracy: 0.7895 - val_loss: 0.4924
    Epoch 19/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 172ms/step - accuracy: 0.7707 - loss: 0.5056 - val_accuracy: 0.7895 - val_loss: 0.4908
    Epoch 20/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 171ms/step - accuracy: 0.7965 - loss: 0.4606 - val_accuracy: 0.7895 - val_loss: 0.4880
    Epoch 21/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 172ms/step - accuracy: 0.7499 - loss: 0.5189 - val_accuracy: 0.7763 - val_loss: 0.5002
    Epoch 22/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 172ms/step - accuracy: 0.7716 - loss: 0.4492 - val_accuracy: 0.8026 - val_loss: 0.4903
    Epoch 23/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 174ms/step - accuracy: 0.7400 - loss: 0.4909 - val_accuracy: 0.7237 - val_loss: 0.5611
    Epoch 24/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 172ms/step - accuracy: 0.8365 - loss: 0.4371 - val_accuracy: 0.7500 - val_loss: 0.5203
    Epoch 25/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 173ms/step - accuracy: 0.7481 - loss: 0.5150 - val_accuracy: 0.6974 - val_loss: 0.5429
    Epoch 26/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 180ms/step - accuracy: 0.8186 - loss: 0.4554 - val_accuracy: 0.7237 - val_loss: 0.5840
    Epoch 27/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 174ms/step - accuracy: 0.7449 - loss: 0.5082 - val_accuracy: 0.8026 - val_loss: 0.4801
    Epoch 28/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 172ms/step - accuracy: 0.7901 - loss: 0.4442 - val_accuracy: 0.7632 - val_loss: 0.5012
    Epoch 29/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 178ms/step - accuracy: 0.7805 - loss: 0.4572 - val_accuracy: 0.7632 - val_loss: 0.5201
    Epoch 30/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 173ms/step - accuracy: 0.8093 - loss: 0.4526 - val_accuracy: 0.6842 - val_loss: 0.5651
    Epoch 31/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 175ms/step - accuracy: 0.7768 - loss: 0.4749 - val_accuracy: 0.6184 - val_loss: 0.6537
    Epoch 32/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 174ms/step - accuracy: 0.7740 - loss: 0.4773 - val_accuracy: 0.6447 - val_loss: 0.7103
    Epoch 33/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 173ms/step - accuracy: 0.7638 - loss: 0.4891 - val_accuracy: 0.7632 - val_loss: 0.5340
    Epoch 34/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8089 - loss: 0.4165 - val_accuracy: 0.6842 - val_loss: 0.6294
    Epoch 35/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 174ms/step - accuracy: 0.7810 - loss: 0.4484 - val_accuracy: 0.8158 - val_loss: 0.4721
    Epoch 36/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.8220 - loss: 0.4123 - val_accuracy: 0.7763 - val_loss: 0.4832
    Epoch 37/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.8150 - loss: 0.4200 - val_accuracy: 0.6842 - val_loss: 0.5655
    Epoch 38/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.7640 - loss: 0.4801 - val_accuracy: 0.7632 - val_loss: 0.4960
    Epoch 39/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.7895 - loss: 0.4562 - val_accuracy: 0.7237 - val_loss: 0.5820
    Epoch 40/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8165 - loss: 0.4359 - val_accuracy: 0.7500 - val_loss: 0.5188
    Epoch 41/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 181ms/step - accuracy: 0.8076 - loss: 0.4192 - val_accuracy: 0.7763 - val_loss: 0.4854
    Epoch 42/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 182ms/step - accuracy: 0.8119 - loss: 0.4326 - val_accuracy: 0.7237 - val_loss: 0.5264
    Epoch 43/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 179ms/step - accuracy: 0.7929 - loss: 0.4201 - val_accuracy: 0.7500 - val_loss: 0.5179
    Epoch 44/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.7911 - loss: 0.4643 - val_accuracy: 0.7368 - val_loss: 0.5048
    Epoch 45/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 179ms/step - accuracy: 0.7957 - loss: 0.4411 - val_accuracy: 0.7895 - val_loss: 0.4736
    Epoch 46/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 179ms/step - accuracy: 0.7928 - loss: 0.4677 - val_accuracy: 0.8026 - val_loss: 0.4687
    Epoch 47/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.7322 - loss: 0.5356 - val_accuracy: 0.6316 - val_loss: 0.6603
    Epoch 48/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 181ms/step - accuracy: 0.7645 - loss: 0.5059 - val_accuracy: 0.8026 - val_loss: 0.4798
    Epoch 49/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 185ms/step - accuracy: 0.8084 - loss: 0.4232 - val_accuracy: 0.8026 - val_loss: 0.4664
    Epoch 50/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 179ms/step - accuracy: 0.8138 - loss: 0.4120 - val_accuracy: 0.7763 - val_loss: 0.4758
    Epoch 51/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 178ms/step - accuracy: 0.7986 - loss: 0.4234 - val_accuracy: 0.7763 - val_loss: 0.4754
    Epoch 52/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 181ms/step - accuracy: 0.8122 - loss: 0.4264 - val_accuracy: 0.7632 - val_loss: 0.5320
    Epoch 53/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 184ms/step - accuracy: 0.8071 - loss: 0.4283 - val_accuracy: 0.7895 - val_loss: 0.4725
    Epoch 54/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 181ms/step - accuracy: 0.8225 - loss: 0.3919 - val_accuracy: 0.8026 - val_loss: 0.4722
    Epoch 55/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 195ms/step - accuracy: 0.8264 - loss: 0.3762 - val_accuracy: 0.7632 - val_loss: 0.4978
    Epoch 56/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 181ms/step - accuracy: 0.7962 - loss: 0.4188 - val_accuracy: 0.7105 - val_loss: 0.5862
    Epoch 57/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 181ms/step - accuracy: 0.7497 - loss: 0.5292 - val_accuracy: 0.7763 - val_loss: 0.4797
    Epoch 58/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 183ms/step - accuracy: 0.8257 - loss: 0.3959 - val_accuracy: 0.7895 - val_loss: 0.4788
    Epoch 59/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 185ms/step - accuracy: 0.8135 - loss: 0.4098 - val_accuracy: 0.7105 - val_loss: 0.5480
    Epoch 60/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 182ms/step - accuracy: 0.7984 - loss: 0.4482 - val_accuracy: 0.8026 - val_loss: 0.4786
    Epoch 61/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8029 - loss: 0.3935 - val_accuracy: 0.8026 - val_loss: 0.4746
    Epoch 62/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8282 - loss: 0.4097 - val_accuracy: 0.7500 - val_loss: 0.4819
    Epoch 63/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 175ms/step - accuracy: 0.7472 - loss: 0.5061 - val_accuracy: 0.8026 - val_loss: 0.4625
    Epoch 64/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 187ms/step - accuracy: 0.8264 - loss: 0.4089 - val_accuracy: 0.7632 - val_loss: 0.5313
    Epoch 65/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.7822 - loss: 0.4639 - val_accuracy: 0.7763 - val_loss: 0.4873
    Epoch 66/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 181ms/step - accuracy: 0.8061 - loss: 0.3982 - val_accuracy: 0.8158 - val_loss: 0.4728
    Epoch 67/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 179ms/step - accuracy: 0.8256 - loss: 0.3909 - val_accuracy: 0.8026 - val_loss: 0.4771
    Epoch 68/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.8168 - loss: 0.4115 - val_accuracy: 0.7500 - val_loss: 0.5056
    Epoch 69/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 179ms/step - accuracy: 0.8498 - loss: 0.3910 - val_accuracy: 0.7500 - val_loss: 0.5317
    Epoch 70/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.8031 - loss: 0.3950 - val_accuracy: 0.7632 - val_loss: 0.5856
    Epoch 71/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.7348 - loss: 0.5908 - val_accuracy: 0.7632 - val_loss: 0.5177
    Epoch 72/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.8052 - loss: 0.4193 - val_accuracy: 0.7632 - val_loss: 0.4840
    Epoch 73/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8354 - loss: 0.3896 - val_accuracy: 0.8026 - val_loss: 0.4696
    Epoch 74/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8210 - loss: 0.4180 - val_accuracy: 0.8026 - val_loss: 0.4660
    Epoch 75/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8304 - loss: 0.4005 - val_accuracy: 0.8026 - val_loss: 0.4700
    Epoch 76/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 179ms/step - accuracy: 0.8037 - loss: 0.4168 - val_accuracy: 0.7500 - val_loss: 0.4995
    Epoch 77/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.8328 - loss: 0.3988 - val_accuracy: 0.6316 - val_loss: 0.6587
    Epoch 78/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 179ms/step - accuracy: 0.8264 - loss: 0.4372 - val_accuracy: 0.7500 - val_loss: 0.5275
    Epoch 79/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 180ms/step - accuracy: 0.8325 - loss: 0.3778 - val_accuracy: 0.7500 - val_loss: 0.5088
    Epoch 80/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.8193 - loss: 0.3951 - val_accuracy: 0.7763 - val_loss: 0.4765
    Epoch 81/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.8023 - loss: 0.4142 - val_accuracy: 0.7632 - val_loss: 0.4901
    Epoch 82/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8101 - loss: 0.4013 - val_accuracy: 0.7632 - val_loss: 0.5433
    Epoch 83/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8187 - loss: 0.3842 - val_accuracy: 0.7500 - val_loss: 0.5376
    Epoch 84/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 178ms/step - accuracy: 0.8075 - loss: 0.4410 - val_accuracy: 0.7632 - val_loss: 0.4870
    Epoch 85/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 181ms/step - accuracy: 0.8497 - loss: 0.3564 - val_accuracy: 0.7500 - val_loss: 0.4811
    Epoch 86/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 179ms/step - accuracy: 0.8369 - loss: 0.3743 - val_accuracy: 0.7105 - val_loss: 0.5616
    Epoch 87/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 178ms/step - accuracy: 0.7913 - loss: 0.4055 - val_accuracy: 0.6711 - val_loss: 0.6402
    Epoch 88/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 178ms/step - accuracy: 0.8157 - loss: 0.4145 - val_accuracy: 0.7500 - val_loss: 0.5038
    Epoch 89/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 178ms/step - accuracy: 0.8363 - loss: 0.3877 - val_accuracy: 0.8026 - val_loss: 0.4698
    Epoch 90/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 178ms/step - accuracy: 0.8307 - loss: 0.3896 - val_accuracy: 0.7105 - val_loss: 0.5421
    Epoch 91/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.7918 - loss: 0.4470 - val_accuracy: 0.7632 - val_loss: 0.4911
    Epoch 92/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 180ms/step - accuracy: 0.8374 - loss: 0.3622 - val_accuracy: 0.7632 - val_loss: 0.5015
    Epoch 93/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.7991 - loss: 0.4135 - val_accuracy: 0.7632 - val_loss: 0.4991
    Epoch 94/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 179ms/step - accuracy: 0.8337 - loss: 0.3858 - val_accuracy: 0.6974 - val_loss: 0.5531
    Epoch 95/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 192ms/step - accuracy: 0.8221 - loss: 0.3955 - val_accuracy: 0.7500 - val_loss: 0.5212
    Epoch 96/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 183ms/step - accuracy: 0.8274 - loss: 0.3954 - val_accuracy: 0.8026 - val_loss: 0.4826
    Epoch 97/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8243 - loss: 0.3623 - val_accuracy: 0.7895 - val_loss: 0.4759
    Epoch 98/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 177ms/step - accuracy: 0.8321 - loss: 0.3827 - val_accuracy: 0.7895 - val_loss: 0.4783
    Epoch 99/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 178ms/step - accuracy: 0.8426 - loss: 0.3596 - val_accuracy: 0.7632 - val_loss: 0.4935
    Epoch 100/100
    [1m69/69[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 176ms/step - accuracy: 0.8367 - loss: 0.3818 - val_accuracy: 0.7500 - val_loss: 0.5106
    


```python
image = cv2.imread("./9337f30f3ba7051ee09bea5bdf0964e8.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
```


```python
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x1f27c243c20>




    
![png](output_24_1.png)
    



```python
image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
```


```python
image = image/255.0
```


```python
image_array = np.array(image).reshape(-1,IMG_SIZE,IMG_SIZE,3)
```


```python
pred = resnet_model.predict(image_array)
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 2s/step
    


```python
output_class = np.argmax(pred)
if output_class == 0:
    print("Rust")
else:
    print("No rust")
```

    Rust
    


```python
resnet_model.evaluate(r_1,y)
```

    [1m24/24[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 348ms/step - accuracy: 0.8481 - loss: 0.3655
    




    [0.3945354223251343, 0.8276315927505493]


