#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as nping
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import load_img,img_to_array


# In[3]:


#Accesing Google drive


# In[4]:


from google.colab import drive
drive.mount('/content/drive')


# In[5]:


get_ipython().system('unzip /content/drive/MyDrive/drive-download-20220930T072955Z-001.zip')


# In[ ]:


get_ipython().system('unzip /content/Training.zip')


# In[ ]:


get_ipython().system('unzip /content/Validation.zip')


# In[8]:


#Training Data


# In[9]:


epochs=50
lr=1e-3
batch_size=128
data=[]
labels=[]


# In[10]:


size=224


# In[11]:


train_datagen=ImageDataGenerator(horizontal_flip=True,width_shift_range=0.4,height_shift_range=0.4,zoom_range=0.3,rotation_range=20,rescale=1/255)


# In[12]:


test_gen=ImageDataGenerator(rescale=1/255)


# In[13]:


target_size=(size,size)
target_size


# In[14]:


train_generator=train_datagen.flow_from_directory(directory="/content/Training",target_size=target_size,batch_size=batch_size,class_mode="binary")


# In[15]:


validation_generator=test_gen.flow_from_directory(directory="/content/Validation",target_size=target_size,batch_size=batch_size,class_mode="binary")


# In[16]:


train_generator.class_indices


# In[17]:


len(train_generator.classes)


# In[18]:


train_generator.class_mode


# In[19]:



x,y=train_generator.next()


# In[20]:


x[0].shape


# In[21]:


x[0]


# In[22]:


#Building Model for Gender prediction


# In[23]:


model=Sequential()
model.add(InceptionV3(include_top=False,pooling="avg",weights="imagenet"))
model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(2048,activation="relu"))
model.add(BatchNormalization())

model.add(Dense(1024,activation="relu"))
model.add(BatchNormalization())

model.add(Dense(1,activation="sigmoid"))

model.layers[0].trainable=False


# In[24]:


model.summary()


# In[25]:


model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# In[26]:


len((train_generator.filenames)),batch_size,len((train_generator.filenames))//batch_size


# In[27]:


model.fit(train_generator,steps_per_epoch=len(train_generator.filenames)//batch_size,epochs=5,validation_data=validation_generator,validation_steps=len(validation_generator.filenames)//batch_size)


# In[29]:


#Testing model by passing a random image


# In[28]:


img_path="/content/Training/male/090648.jpg.jpg"


# In[29]:


img=load_img(img_path,target_size=(size,size,3))
plt.imshow(img)


# In[30]:


img=img_to_array(img)
img


# In[31]:


img=img/255.0
img=img.reshape(1,size,size,3)


# In[32]:


img.shape


# In[33]:


res=model.predict(img)
res


# In[34]:


train_generator.class_indices


# In[35]:


if res[0][0]<=0.5:
  prediction="women"
else:
  prediction="men"
print("prediction:",prediction)


# In[36]:


#predicting whole Dataset 1st validation data set


# In[37]:


import matplotlib.pyplot as plt
import matplotlib.image as nping
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img,img_to_array
directroy=r"/content/Validation"
Catogeries=["female","male"]


# In[38]:


size=224
data=[]
i=0
data=pd.read_csv("/content/work.csv")
l_pred=list()
l_imgid=list()
d={}


# In[ ]:


for catogery in Catogeries:
  folder=os.path.join(directroy,catogery)
  for img in os.listdir(folder):
    img_path=os.path.join(folder,img)
    img_arr=cv2.imread(img_path)
    img_arr=cv2.resize(img_arr,(size,size))
    img_arr=img_to_array(img_arr)
    img_arr=img_arr/255.0
    img_arr=img_arr.reshape(1,size,size,3)
    res=model.predict(img_arr)
    i=i+1
    if res[0][0]<0.5:
      l_imgid.append(i)
      l_pred.append("Women")
    else:
      l_imgid.append(i)
      l_pred.append("Men")


# In[40]:


len(l_imgid)


# In[41]:


#saving prediction to csv file


# In[42]:


d=pd.DataFrame({"Image_id":l_imgid,"prediction":l_pred})
d.to_csv("/content/work.csv")
s=pd.read_csv("/content/work.csv")
s


# In[43]:


s.head(50)


# In[44]:


s.tail(50)


# In[45]:


#predicting whole training data set


# In[46]:


directroy=r"/content/Training"
Catogeries=["female","male"]


# In[47]:


size=224
data=[]
i=0
data=pd.read_csv("/content/training_pred.csv")
l_pred=list()
l_imgid=list()
d={}


# In[ ]:


for catogery in Catogeries:
  folder=os.path.join(directroy,catogery)
  for img in os.listdir(folder):
    img_path=os.path.join(folder,img)
    img_arr=cv2.imread(img_path)
    img_arr=cv2.resize(img_arr,(size,size))
    img_arr=img_to_array(img_arr)
    img_arr=img_arr/255.0
    img_arr=img_arr.reshape(1,size,size,3)
    res=model.predict(img_arr)
    i=i+1
    if res[0][0]<0.5:
      l_imgid.append(i)
      l_pred.append("Women")
    else:
      l_imgid.append(i)
      l_pred.append("Men")


# In[49]:


#saving to csv file


# In[50]:


d1=pd.DataFrame({"Image_id":l_imgid,"prediction":l_pred})
d1.to_csv("/content/training_pred.csv")
s1=pd.read_csv("/content/training_pred.csv")
s1


# In[51]:


s1.head(50)


# In[52]:


s1.tail(50)

