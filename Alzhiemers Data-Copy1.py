#!/usr/bin/env python
# coding: utf-8

# In[42]:


from keras.preprocessing.image import ImageDataGenerator

# create a new generator
imagegen=ImageDataGenerator(rescale=1.0/255,validation_split=0.2)

#load train data
train=imagegen.flow_from_directory("AlzheimersDataset/train",class_mode='categorical',shuffle=False,batch_size=16,target_size=(176,208),subset="training")

#load val data

val=imagegen.flow_from_directory("AlzheimersDataset/train",class_mode='categorical',shuffle=False,batch_size=16,target_size=(176,208),subset="validation")

#test

test=imagegen.flow_from_directory("AlzheimersDataset/test",class_mode='categorical',shuffle=False,batch_size=16,target_size=(176,208))


# In[46]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Input,BatchNormalization,Dropout


#build a sequential model
model= Sequential()
#model.add(Input(input_shape=(176,208,3)))

#1st conv block #input_shape=(176,176,3)

model.add(Conv2D(16,(5,5),activation='relu',strides=(1,1),padding='same',input_shape=(176,208,3)))  #filters multiples of 8 or 16
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

#2nd conv block
model.add(Conv2D(32,(5,5),activation='relu',strides=(2,2),padding='same'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(BatchNormalization())

#3rd conv block
model.add(Conv2D(64,(3,3),activation='relu',strides=(2,2),padding='same'))
model.add(MaxPool2D(pool_size=(2,2),padding='valid'))
model.add(BatchNormalization())

#4th conv block
model.add(Conv2D(128,(3,3),activation='relu',strides=(2,2),padding='same'))
model.add(MaxPool2D(pool_size=(2,2),padding='valid'))
model.add(BatchNormalization())

#5th conv block
#model.add(Conv2D(256,(3,3),activation='relu',strides=(2,2),padding='same'))
#model.add(MaxPool2D(pool_size=(2,2),padding='valid'))
#model.add(BatchNormalization())

#Ann block
model.add(Flatten())
model.add(Dense(units=500,activation='relu'))#128,256
model.add(Dense(units=500,activation='relu'))#multiple s of 16 start with 64 add 5layers
model.add(Dense(units=500,activation='relu'))
model.add(Dense(units=500,activation='relu'))
model.add(Dense(units=500,activation='relu'))
model.add(Dropout(0.25))  #25% deopout means it will delete the 25% of neurons i.e reduce the features ,this is to avoid the overfitting

#outputlayer
model.add(Dense(units=4,activation='softmax'))



# In[47]:


#compile model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) #SGD optimiser

#fit on data for 5 epochs
history=model.fit(train,epochs=100,validation_data=val)


# In[48]:


model.summary()

