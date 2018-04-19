import numpy as np
import string
from gensim import corpora 
import Pretreatment


# 新建LDA模型
lda =Pretreatment.LDA()
Corpus ,Product,Component,Developer= lda.read_csv(dir="AspectJ.csv",summary=2,description=3,product = 4,component=5,assigned_to= 6,comment=7)
Terms,Topics = lda.build_from_corpus(Corpus)


# 生成MLkNN
mlknn = Pretreatment.BR_MLkNN()
Terms = mlknn.makematrix(Terms,lenth=len(lda.dictionary))
Topics = mlknn.makematrix(Topics,lenth=lda.num_topics)
X,y=mlknn.make_Xy(Terms=Terms,Topics=Topics,Product=Product,Component=Component,Developer=Developer)

step = int(len(X)/11)
# 分11组验证
# 仅br分析的结果
for n in range(step, len(X)-step,step ):
    testX_Validation = X[n:n+step ,:]
    testy_Validation = y[n:n+step ,:]
    trainX_Validation = X[0:n ,:]
    trainy_Validation = y[0:n ,:]
    print("test only br-analysis")
    mlknn.br_mlknn.fit(trainX_Validation,trainy_Validation)
    score = mlknn.br_mlknn.predict(testX_Validation)
    
    r5 = mlknn.recall(y=testy_Validation,score=score,n=5)
    r10 = mlknn.recall(y=testy_Validation,score=score,n=10)
    print('r5:',r5)
    print('r10:',r10)

# 双重分析的结果，如果设置gama[0]=0,则为仅分析开发者
for n in range(step, len(X)-step,step ):
    testX_Validation = X[n:n+step ,:]
    testy_Validation = y[n:n+step ,:]
    trainX_Validation = X[0:n ,:]
    trainy_Validation = y[0:n ,:]
    r5 = 0
    r10 =0
    if n == step:
        mlknn.findgama(testX = trainX_Validation,testy=trainy_Validation,X=X,y=y)
    else:
        mlknn.set(trainX_Validation,trainy_Validation)
        r5 = mlknn.test(testX_Validation,testy_Validation,recall = 5)
        r10 = mlknn.test(testX_Validation,testy_Validation,recall = 10)
    print('r5:',r5)
    print('r10:',r10)


