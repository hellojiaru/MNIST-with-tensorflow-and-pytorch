import tensorflow as tf
import numpy as np

(x,y),(_x,_y) = tf.keras.datasets.mnist.load_data()
x,_x = x/255.,_x/255.   #将图片数据压缩到0-1的范围，这步可选，不压缩会导致准确率降低

#使用keras建模和自动化训练
def easy_keras_train():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())    #图片本身是(28,28)的尺寸，为了使用全连接层要先打平成(28*28)
    model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))   #将结果转化为概率分布，（可选）
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),   #keras有很多优化器，随便选一个
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),   #多分类，并且分类是数字表示，要用SparseCategoricalCrossentropy。如果分类是onehot，就用CategoricalCrossentropy
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) #同上，这个可以在训练看到准确率
    model.fit(x,y,epochs=5,validation_data=(_x,_y)) #默认batch_size是32

#使用类继承的方式建模并自定义训练过程
def custom_train():
    #要自行切分数据
    batch_size = 100
    data_length = len(x)
    episodes = int(data_length/batch_size)
    val_data_length = len(_x)
    val_episodes = int(val_data_length/batch_size)
    #定义类，要在init函数里定义层，否则trainable_variables会是空
    class Model(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(128,activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(10,activation=tf.nn.softmax)
        #将输入数据按层一次传入，最后返回
        def call(self,input):
            digits = self.flatten(input)
            digits = self.dense1(digits)
            digits = self.dense2(digits)
            return digits
    model = Model()
    optimizer = tf.keras.optimizers.Adam()  #同上
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()   #同上
    metric = tf.keras.metrics.SparseCategoricalAccuracy()   #同上

    for epoch in range(5):
        #训练步骤
        acc = []
        metric.reset_states()
        for episode in range(episodes):
            #加载各批次的数据
            start = episode*batch_size
            end = (episode+1)*batch_size
            data = x[start:end]
            labels = y[start:end]
            #使用tape将要求导的计算步骤包起来，tf就可以实现自动求导了
            with tf.GradientTape() as tape:
                preds = model(data, training=True)  #数据经过模型的各个层得到预测结果，即前向传播
                loss = loss_fn(labels,preds)    #将输出与实际的标签对比，即使用损失函数计算损失值
            grads = tape.gradient(loss,model.trainable_variables)   #tf将上述过程对损失值求导（为了让损失值尽可能小，即预测尽可能接近目标）
            optimizer.apply_gradients(zip(grads,model.trainable_variables)) #将求导结果应用到模型的参数上，即反向传播

            acc.append(metric(labels,preds).numpy())    #计算每次的准确率，并保存，后续取均值       
        print('epoch',epoch,'acc',np.average(acc))

        #验证步骤，与上面类似，验证不需要改变模型，所以不用求导不用反向传播
        acc = []
        metric.reset_states()
        for val_episode in range(val_episodes):
            start = val_episode*batch_size
            end = (val_episode+1)*batch_size
            data = _x[start:end]
            labels = _y[start:end]
            preds = model(data, training=False)
            acc.append(metric(labels,preds).numpy())
        print('epoch',epoch,'val_acc',np.average(acc))
        


if __name__ == "__main__":
    # easy_keras_train()
    custom_train()




            

