from mxnet import init
import mxnet as mx
from mxnet.gluon import nn
import numpy as np
from mxnet import nd
from mxnet.gluon import rnn
import pickle
from mxnet import gluon
from mxnet import autograd
"""
readme:
今日更新，更改为BIMPM w/o max attentive的形式进行测试

readme:
今日更新，使用预训练的embedding以及合适的词表大小

readme:
今日更新，更改为最高成绩的smn

readme：
今日更新，使用cnn进行分类
"""
val_size = 10000
inter = 2000
intra = 2000
batch_size = 1000
ctx = mx.gpu(0)

max_turn = 1

import pickle
vocab_raw = open('vocab.pkl','rb')
vocab = pickle.load(vocab_raw)

def batch_attention(encoder,decoder):
    attention = nd.softmax(nd.batch_dot(encoder,nd.transpose(decoder,axes = (0,2,1))),axis=1)
    new_decoder = nd.batch_dot(attention,nd.transpose(encoder,axes=(0,1,2)))
    return new_decoder

train = open('fenlei_train','r')
val = open('fenlei_val','r')

def word2token(text,vocab):
    count = 0
    for qes in text:
            text[count] = vocab.get(int(qes),1)
            count = count+1
    return text

def get_single_data(raw_data,batch_size,max_l):
    X,y = np.zeros((4000000,30)),np.zeros((3500000,2000))
    i = 0
    t = 0
    count = 0
    err = 0
    for line in raw_data:
        count = count +1
        line = line.strip()
        
        try:
            question = line.split('\t')[0]
            topic = line.split('\t')[1]
        except:
            print(line)
        
        try:
            question = question.split(' ')
            topic = topic.split(' ')
            question = word2token(question,vocab)

        except:
            err = err + 1
            print(err)

        if(False):
            break
        else:
#            print(t)
            question = question[0:30]
            try:
                X[t] = [0]*(30-len(question)) + question
                y[t,min(int(topic[0]),1999)] = 1
                y[t,min(int(topic[1]),1999)] = 1
                y[t,min(int(topic[2]),1999)] = 1
                y[t,min(int(topic[3]),1999)] = 1
                y[t,min(int(topic[4]),1999)] = 1
                y[t,1999] = 0
                # y[t,0] = topic[0]
                # y[t,1] = topic[1]
                # y[t,2] = topic[2]
                # y[t,3] = topic[3]
                # y[t,4] = topic[4]
            except:
                print(line)
                print(t)
            t = t + 1
            if(t%10000==0):
                print(t)
    y = y[0:t]
    y = y[0:((len(y)//batch_size)*batch_size)]
    X = X[0:len(y)]
    X = nd.array(X,ctx=ctx)
    # y = nd.array(y,dtype='int32')
    print("get data")
    train_dataset = gluon.data.ArrayDataset(X, y)
    train_data_iter = gluon.data.DataLoader(train_dataset, batch_size, shuffle=False)
    return train_data_iter 

class SMN_Last(nn.Block):
    def __init__(self,**kwargs):
        super(SMN_Last,self).__init__(**kwargs)
        with self.name_scope():
            
            self.Embed = nn.Embedding(411721,256)
            # agg param
            self.gru = rnn.GRU(1024,2,layout='NTC')
            self.mlp_1 = nn.Dense(units=60,flatten=False,activation='relu')
            self.mlp_2 = nn.Dense(units=1,flatten=False)
            # lstm param
            self.topic_embedding = self.params.get('param_test',shape=(1024,2000))


    def forward(self,x):
        """
        return shape:(batch_size,2000,2)
        """
        # Encode layer
        question = x[:,0:30]
        question = self.Embed(question)
        question = self.gru(question)

        #interaction layer
        interaction = nd.dot(question,self.topic_embedding.data())
        interaction = nd.transpose(interaction,axes=(0,2,1))
        interaction = interaction.reshape((batch_size*2000,-1))
        # interaction = interaction.expand_dims(axis=1)
        # print("interaction done")

        #agg layer
        # interaction = self.pooling(self.conv_2(self.conv_1(interaction)))
        # print("agg done")
        res = self.mlp_2(self.mlp_1(interaction))
        res = res.reshape((batch_size,2000))

        return res

#Train Model
SMN = SMN_Last()
SMN.initialize(ctx=ctx)
embed_raw = open('embed_clean','rb')

word2vec = (nd.array(np.loadtxt(embed_raw))).copyto(ctx)

SMN.Embed.weight.set_data(word2vec)

train_iter = get_single_data(train,batch_size,max_l=50)
val_iter =  get_single_data(val,batch_size,max_l=50)
max_epoch = 300

Sloss = gluon.loss.SigmoidBCELoss()
trainer = gluon.Trainer(SMN.collect_params(), 'adam', {'learning_rate': 0.001})

from score import score

def get_label(label):
    dense_label = nd.zeros((batch_size,2000),ctx=ctx)
    for row in range(0,batch_size):
        for col in range(0,5):
            index = label[row,col]
            if(index<2000):
                dense_label[row,index] = 1
    return dense_label
SMN.Embed.weight.lr_mut = 0

for epoch in range(max_epoch):
    train_loss = 0.
    train_acc = 0.
    count = 0
    kk = 0
    for data, label in train_iter:
        if(epoch>0):
            SMN.Embed.weight.lr_mut = 0.001
        # data = data.copyto(ctx)
        # dense_label = get_label(label)
        with autograd.record():
            output = SMN(data)
            loss = Sloss(output, nd.array(label,ctx=ctx).astype('float32'))
            count = count + 1
        loss.backward()
        trainer.step(batch_size)
        if(count%10==1):
            print("loss:")
            print(nd.mean(loss).asscalar()
    print("Epoch %d. Loss: %f" % (
        epoch, train_loss/len(train_iter))
