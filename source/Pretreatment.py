# -*- coding: utf-8 -*-
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer 
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora 
from gensim.models import LdaModel
import csv
import string
import random
import numpy
import mlknn


class LDA:

    """生成LDA模型，抽取话题时抽取术语量的百分之五"""
    
    def __init__(self):
        self.ldamodel = None
        self.dictionary = None
        self.num_topics = 0      
        
    def loadLocalModel(self,modelName='lda.model',dictName='dictionary'):
        # 默认模型的名字是lda.model，字典默认存在dictionary
        self.ldamodel = LdaModel.load(modelName)
        self.dictionary = corpora.Dictionary.load(dictName)
        print(" + [OK] LDA model is ready!")
          
    def build_from_corpus(self,docs,save_dict_to='dictionary',save_model_to='lda.model'):
        
        # 为每一个词语分配一个整数，做成字典，再讲每个文档处理成向量语料
        self.dictionary = corpora.Dictionary(docs) 
        
        # 删除低频词语
        self.dictionary.filter_extremes(no_below=10)
        
        self.dictionary.save(save_dict_to)
        print(" + [OK] dictionary is saved in " + save_dict_to + " ! ")

        # 提取出术语向量
        trainTerms = self.cheackTerms(docs)

        # 主题数设置为词语的百分之5，生成模型
        num_topics = int(len(self.dictionary) / 20)
        self.num_topics = num_topics 
        print("num_topics = "+str(num_topics))
        self.ldamodel = LdaModel(corpus=trainTerms, num_topics=num_topics , alpha = 50 / num_topics,eta = 0.01,id2word = self.dictionary,iterations=5, passes= 20) 
        self.ldamodel.save(save_model_to)
        print(" + [OK] LDA model is saved!")
        self.num_topics =num_topics
        # 提取每一个样本对应话题的可能性并保存
        trainTopicProbability = self.cheackTopics(trainTerms)
        
        print(" + [OK] LDA model is ready!")
        return trainTerms,trainTopicProbability
    
    def read_csv(self,dir,summary=-1, description=-1, product=-1, component=-1,assigned_to= -1,comment=-1,title=1):
        csvFile = open(dir, "r") 
        #读取文件
        BugReports = csv.reader(csvFile)
        print(" + [OK] read file!")

        # 读取summary和description
        charact =[] 
        
        products =[]
        components = []
        developer =[]
        # 读取产品、模块、开发者列表
        for br in BugReports:
            charact.append(br[summary] + ' ' + br[description])
            products.append(br[product] )
            components.append(br[component]) 
            developer.append(br[assigned_to] + ' ' + br[comment])
        
        # 处理除了第一行标题以外的文档
        charact = self.stemwords(charact[title:])
        products = self.stemwords(products[title:])
        components = self.stemwords(components[title:])
        developer = self.stemwords(developer[title:])
                
        print(" + [OK] the file has be Standardized!")
        return charact, products,components,developer

    def stemwords(self,data):
        # 格式化字符串
        p_stemmer = PorterStemmer() 
        docs = []
        
        # 停词表
        list_stopWords = list(set(stopwords.words('english')))
        for doc in data:
            # 将所有标点/数字替换成空格以方便分词(除了连接符号-.+)
            c= ['-','+','.']
            remove_punctuation_map = dict((ord(char), " ") for char in string.punctuation if not char in c)
            remove_number_map = dict((ord(char), " ") for char in string.digits)
            doc = doc.translate(remove_punctuation_map)
            doc = doc.translate(remove_number_map)
            # 待处理文本，先分词
            list_words = word_tokenize(doc)
            # 删除停词,还原词根
            new_list_words = [w for w in list_words if 2 < len(w) < 15 ]
            filtered_words = [p_stemmer.stem(w) for w in new_list_words if not w in list_stopWords]
            docs.append(filtered_words)

        return docs
    
    
    def cheackTerms(self,docs):
        # 按字典将语料翻译成向量
        termCorpus = [self.dictionary.doc2bow(doc) for doc in docs]
        return termCorpus  

    def cheackTopics(self, testcorpus):
        # 计算每个文本对应各个主题的概率
        probability = [self.ldamodel.get_document_topics(bow=bow) for bow in testcorpus]

        return probability


class BR_MLkNN:
    def __init__(self):
        self.br_mlknn = mlknn.MLkNN(k=15,s=1.0)
        self.gama5 = [0.3,0.05,0.39,0.7,0.1]
        self.gama10 = [0.3,0.05,0.77,0.5,0.4]
        self.termlenth = 0
        self.topiclenth = 0
        self.productlenth = 0
        self.componentlenth = 0
        self.dl_term = None
        self.dl_topic = None
        self.dl_product =  None
        self.dl_component =  None

        
    def make_Xy(self, Terms,Topics,Product,Component,Developer):
        # 生成产品、模块、开发者向量的字典
        pdDict =corpora.Dictionary(Product) 
        cpDict = corpora.Dictionary(Component) 
        dlDict = corpora.Dictionary(Developer) 
        dlDict.filter_extremes(no_below=10)
        
        # 生成产品、模块、开发者的向量表
        Product = [pdDict.doc2bow(bugreport) for bugreport in Product]
        Component = [cpDict.doc2bow(bugreport) for bugreport in Component]
        Developer = [dlDict.doc2bow(bugreport) for bugreport in Developer]
        
        # 将向量表转化为矩阵
        Product = self.makematrix(data=Product, lenth=len(pdDict))
        Component = self.makematrix(data=Component, lenth=len(cpDict))
        Developer = self.makematrix(data=Developer, lenth=len(dlDict),default = 1)

        self.termlenth = len(Terms[0])
        self.topiclenth = len(Topics[0])
        self.productlenth = len(Product[0])
        self.componentlenth = len(Component[0])

        # 拼接矩阵得到ML-kNN的X和Y
        X = numpy.column_stack((Terms,Topics))
        X = numpy.column_stack((X,Product))
        X = numpy.column_stack((X,Component))
        
        y = Developer 
        
        return X,y

    def makematrix(self, data,lenth,default = 0):
        
        matrix = numpy.zeros((len(data),lenth))
        
        for row in range(len(data)):
            for col in data[row]: 
                if default>0:
                    matrix[row,col[0] - 1] = 1
                else:
                    matrix[row,col[0] - 1] = col[1]
                
        return matrix

    def findgama(self, testX,testy,X,y):
        self.set(X,y)
        recall5 = 0
        recall10 = 0
        for n in range(10):
            print(" + [OK] wait for "+str(10-n)+" s!")
            gamatest = [random.random() for i in range(5)] 
            recalltest5 = 0
            recalltest10 = 0
            
            for i in range(5):
                print(" + [OK] tese gama["+str(i)+"] !")
                while True:
                    recalltest5 = self.test(testX, testy,5,gama = gamatest)
                    recalltest10 = self.test(testX, testy,10,gama = gamatest)
                    if recalltest5>recall5:
                        self.gama5= gamatest
                        recall5 = recalltest5
                        gamatest[i]+=0.03
                        
                    if recalltest10>recall10:
                        self.gama10= gamatest
                        recall10 = recalltest10
                        gamatest[i]+=0.03
                        
                    if gamatest[i]>0 or (recalltest5<recall5 and recalltest10<recall10):
                        break
        print(" + [OK]find gama:",self.gama5,self.gama10)

    def set(self,X,y):
        self.br_mlknn.fit(X,y)
        
        lenth = self.termlenth
        term_score = (y.T).dot(X[:,:lenth])
        

        for developer in range(term_score.shape[0]):
            t_num = term_score[developer]
            t_num.sort()
            t_num = t_num[-10]
            for t in range(term_score.shape[1]):
                if term_score[developer,t] <t_num:
                    term_score[developer,t] = 0
        self.dl_term= term_score
        
        topic_score = (y.T).dot( X[:,lenth:lenth + self.topiclenth])
        all = numpy.ones((y.shape[1],y.shape[0])).dot(X[:,lenth:lenth + self.topiclenth])
        self.dl_topic =1 - topic_score/all
        lenth += self.topiclenth

        product_score = (y.T).dot(  X[:,lenth:lenth + self.productlenth])
        all = numpy.ones((y.shape[1],y.shape[0])).dot(X[:,lenth:lenth + self.productlenth])
        self.dl_product = product_score / all
        lenth += self.productlenth

        component_score = (y.T).dot(  X[:,lenth:lenth + self.componentlenth])
        all = numpy.ones((y.shape[1],y.shape[0])).dot(X[:,lenth:lenth + self.componentlenth])
        self.dl_component = component_score / all
        print("+ [OK] set already! ")
        return True

    def test(self,X,y,recall=5,gama = None):
        print(" + [OK] test now! ")
        if gama == None:
            if recall == 5:
                gama = self.gama5
            else:
                gama = self.gama10
        # 计算br-score
        brscore =  self.br_mlknn.predict_proba(X)
        score = gama[0] * brscore
        # 计算term分数
        lenth = self.termlenth
        # 提取术语矩阵
        term = X[:,0:lenth]
        Nd = self.dl_term.dot( numpy.ones((self.dl_term.shape[1],1)))
        Nb = term.dot( numpy.ones((self.dl_term.shape[1],1)))
        self.dl_term[self.dl_term>0]=1
        Nbd = term.dot(self.dl_term.T)
        term_score = Nbd/(-Nbd +Nd.T+Nb)
        score += gama[1]*(1 - term_score)

        # 计算每个开发者的话题亲和分数
        topic_score = X[:,lenth:lenth+ self.topiclenth]
        topic_score[topic_score>0]=1
        topic_score= 1-(topic_score.dot(1-self.dl_topic.T))
        topic_score=gama[2]*topic_score    
        score +=  topic_score
        
        lenth += self.topiclenth
        product_score = X[:,lenth:lenth+self.productlenth]
        product_score = product_score.dot(self.dl_product.T)
        score +=  gama[3]*product_score
        
        lenth += self.productlenth
        component_score = X[:,lenth:lenth+self.componentlenth]
        component_score = component_score.dot(self.dl_component.T)
        score +=  gama[3]*component_score
        recallp = self.recall(y=y,score=score,n=recall)
        return recallp

    def recall(self, y, score,n):
        brcount = len(y)
        recall_persent = 0 
        for br in range(len(y)):
            brn = 0 # 真实开发者
            scr = 0 # 报告推荐开发者在前K个数
            for dl in range(len(y[br])):
                if y[br,dl] == 1:
                    brn +=1
                    count = 0
                    for x in range(score.shape[1]):
                        if score[br,x] >score[br,dl]:
                            count +=1
                    if count <n:
                        scr += 1
            if brn != 0:
                recall_persent += (1.0*scr) /brn
            else:
                brcount-=1
        recall_persent = recall_persent/brcount

        return recall_persent

    

        


if __name__ == "__main__":   
    print(__doc__)

    