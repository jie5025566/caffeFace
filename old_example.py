# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:52:55 2016

@author: zmj
@brief：在lfw数据库上验证训练好了的网络
"""
import sklearn
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import skimage 
caffe_root = '/home/flyvideo/caffe-master/'  
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import common
import sklearn.metrics.pairwise as pw
from sklearn.preprocessing import normalize



filelist_left='left.list'
filelist_right='right.list'
filelist_label='label.list'
with open('/home/flyvideo/caffe-master/models/vgg_face_caffe/A.pkl', 'rb') as f:
    A = cPickle.load(f)
with open('/home/flyvideo/caffe-master/models/vgg_face_caffe/G.pkl', "rb") as f:
    G = cPickle.load(f)
pca='/home/flyvideo/caffe-master/models/vgg_face_caffe/pca_model.m'

def read_imagelist(filelist):
    '''
    @brief：从列表文件中，读取图像数据到矩阵文件中
    @param： filelist 图像列表文件
    @return ：4D 的矩阵
    '''
    fid=open(filelist)
    lines=fid.readlines()
    test_num=len(lines)
    fid.close()
    X=np.empty((test_num,3,224,224))
    averageImg = [129.1863,104.7624,93.5940]
    i =0
    for line in lines:
        word=line.split('\n')
        filename=word[0]
        im1=skimage.io.imread(filename,as_grey=False)
        image =skimage.transform.resize(im1,(224, 224))*255
        if image.ndim<3:
            print 'gray:'+filename
            X[i,0,:,:]=image[:,:]-averageImg[0]
            X[i,1,:,:]=image[:,:]-averageImg[1]
            X[i,2,:,:]=image[:,:]-averageImg[2]
        else:
            X[i,0,:,:]=image[:,:,0]-averageImg[0]
            X[i,1,:,:]=image[:,:,1]-averageImg[1]
            X[i,2,:,:]=image[:,:,2]-averageImg[2]
        i=i+1
    return X

def read_labels(labelfile):
    '''
    读取标签列表文件
    '''
    fin=open(labelfile)
    lines=fin.readlines()
    labels=np.empty((len(lines),))
    k=0;
    for line in lines:
        labels[k]=int(line)
        k=k+1;
    fin.close()
    return labels

def draw_roc_curve(fpr,tpr,title='cosine',save_name='roc_lfw'):
    '''
    画ROC曲线图
    '''
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Face Verification ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(save_name+'.png')



def evaluate():
    '''
    @brief: 评测模型的性能
    @param：metric： 度量的方法
    '''
     # 转换均值图像数据　-->npy格式文件
    #fin='/media/crw/MyBook/TrainData/LMDB/CASIA-WebFace/10575_64X64/mean.binaryproto'
    #fout='/media/crw/MyBook/TrainData/LMDB/CASIA-WebFace/10575_64X64/mean.npy'
    #blob = caffe.proto.caffe_pb2.BlobProto()
    #data = open( fin , 'rb' ).read()
    #blob.ParseFromString(data)
    #arr = np.array( caffe.io.blobproto_to_array(blob) )
    #out = arr[0]
    #np.save( fout , out )
    #设置为gpu格式
    caffe.set_mode_gpu()
    caffe.set_device(0)
    model_dir = '/home/flyvideo/caffe-master/models/vgg_face_caffe/'
    model = model_dir+'VGG_FACE_deploy.prototxt'
    weights = model_dir+'VGG_FACE.caffemodel'
    #mean_p=model_dir+'my.npy'
    net=caffe.Classifier(model,weights,caffe.TEST,channel_swap=(2, 1, 0))
    #net = caffe.Classifier('./deploy.prototxt', 
    #'/media/crw/MyBook/Model/FaceRecognition/try5_2/snapshot_iter_'+str(itera)+'.caffemodel',
    #mean=np.load(fout))
    #需要对比的图像，一一对应

    
    #print 'network input :' ,net.inputs  
    #print 'network output： ', net.outputs
    #提取左半部分的特征

    X=read_imagelist(filelist_left)
    test_num=np.shape(X)[0]
    #data_1 是输入层的名字
    out = net.forward_all(blobs=['fc7'],data=X)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    feature1 =np.float64(out['fc7'])
    feature1=np.reshape(feature1,(test_num,4096))
    feature1 = normalize(feature1, norm='l2')
    #np.savetxt('feature1.txt', feature1, delimiter=',')

    #提取右半部分的特征
    X=read_imagelist(filelist_right)
    out = net.forward_all(blobs=['fc7'],data=X)
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    feature2 = np.float64(out['fc7'])
    feature2=np.reshape(feature2,(test_num,4096))
    feature2 = normalize(feature2, norm='l2')
    #np.savetxt('feature2.txt', feature2, delimiter=',')
    
    #提取标签    
    labels=read_labels(filelist_label)
    labels= np.array(labels)
    #assert(len(labels)==test_num)
    #计算每个特征之间的距离
    #mt=pw.pairwise_distances(feature1,feature2,metric='cosine')
    #print mt
    #predicts=np.empty((test_num,))
    #for i in range(test_num):
          #predicts[i]=mt[i][i]
        # 距离需要归一化到0--1,与标签0-1匹配
    #for i in range(test_num):
            #predicts[i]=(predicts[i]-np.min(predicts))/(np.max(predicts)-np.min(predicts))
    fea1=common.pca_transform(pca, feature1)
    fea2=common.pca_transform(pca, feature2)
    mt=[]
    for i in range(test_num):
        mt.append(pw.cosine_similarity(fea1[i],fea2[i]))
    print len(mt)
    distance=[]
    threshold1=-30
    threshold2=-90
    for i in range(test_num):
        distance.append(Verify(A, G, fea1[i], fea2[i]))
        if distance[i]>threshold1:
	    mt[i]=mt[i]+0.1
        if distance[i]<threshold2:
	    mt[i]=mt[i]-0.05
    print 'accuracy is :',calculate_accuracy(fea1,fea2,mt,labels,test_num)
    predicts=np.array(mt)
    fpr, tpr, thresholds=sklearn.metrics.roc_curve(labels,predicts,pos_label=1)
    draw_roc_curve(fpr,tpr,title='cosine',save_name='lfw_roc')


def Verify(A, G, x1, x2):
    x1.shape = (-1, 1)
    x2.shape = (-1, 1)
    ratio = np.dot(np.dot(np.transpose(x1), A), x1) + np.dot(np.dot(np.transpose(x2), A), x2) - 2 * np.dot(np.dot(np.transpose(x1), G), x2)
    return float(ratio)   


    
def calculate_accuracy(feature1,feature2,predicts,labels,num):    
    '''
    #计算识别率,
    #选取阈值，计算识别率
    '''     
    accuracy = []
    predict = np.empty((num,))
    threshold = 0.4
    for i in range(num):
        if predicts[i] >= threshold:
            predict[i] =1
        else:
            predict[i] =0
    predict_right =0.0
    for i in range(num):
	if predict[i]==labels[i]:
	    predict_right = 1.0+predict_right
    current_accuracy = (predict_right/num)
    accuracy.append(current_accuracy)
    print predict
    return accuracy

if __name__=='__main__':
    evaluate()
