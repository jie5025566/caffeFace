# -*- coding: UTF-8 -*-
import os
import cv2.cv as cv
import cv2
import time
import cPickle
import datetime
import logging
import flask
from flask import url_for 
import cStringIO
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import scipy.io as sio 
import string
import urllib2
from PIL import Image
from numpy import *
import random
import h5py
import caffe
import exifutil
import sklearn.metrics.pairwise as pw
from flask import Blueprint
from sklearn.externals import joblib 
from sklearn.decomposition import PCA
import cStringIO as StringIO
import skimage
import common
from sklearn.preprocessing import normalize

 
REPO_DIRNAME = os.path.abspath(os.path.dirname(__file__) + '/../..')
UPLOAD_FOLDER = '/tmp/upload'

ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])
with open('/home/flyvideo/caffe-master/models/vgg_face_caffe/A.pkl', 'rb') as f:
    A = cPickle.load(f)
with open('/home/flyvideo/caffe-master/models/vgg_face_caffe/G.pkl', "rb") as f:
    G = cPickle.load(f)
pca='/home/flyvideo/caffe-master/models/vgg_face_caffe/pca_model.m'

# Obtain the flask app object
app = flask.Flask(__name__)
 


 

 
@app.route('/')
def index():
    return flask.render_template('example.html', has_result=False, result_img=setimage())
 

# classify the random image
@app.route('/classify_static', methods=['GET'])
def classify_static():
    imagepath = flask.request.args.get('imagepath','')
    try:
	logging.info('Image: %s', imagepath)
        path=os.path.dirname(__file__) + imagepath
        image = caffe.io.load_image(path)
         
    except Exception as err:
        return flask.render_template(
            'example.html', has_result=True,
            result=(False, 'Cannot open image from static.'))
       
    
    rst_img,result_time=app.clf.search_image(image)

    img_tmp=''''''
    for value in rst_img:
        img_tmp+='''<div class="col-xs-4 col-sm-4 col-md-1 clo-lg-3 marginDown" style="margin: 2% 2% 2% 0; "> '''+'''<a href="#">'''+'''<img src='''+url_for('static',filename=value)+''' width="100" height="100"/>'''+'''</a>'''+'''</div>
    '''
    return flask.render_template(
        'quer.html', has_result=True, result =(True, 'you did it.'), result_img=img_tmp,time=result_time
    )
  
@app.route('/face_recongnition', methods=['POST'])
def face_recongition():
    try:
        # We will save the file to disk for possible data collection
        uploaded_files=flask.request.files.getlist('imgs')
	filenames = []
	for _ in uploaded_files:
	    if _ and allowed_file(_.filename):
		filename = werkzeug.secure_filename(_.filename)
		_.save(os.path.join(UPLOAD_FOLDER,filename))
		filenames.append(filename)
        dect_image(os.path.join(UPLOAD_FOLDER,filenames[0]))
        dect_image(os.path.join(UPLOAD_FOLDER,filenames[1])) 
        image1 = exifutil.open_oriented_im(os.path.join(UPLOAD_FOLDER,filenames[0]))
	image2 = exifutil.open_oriented_im(os.path.join(UPLOAD_FOLDER,filenames[1]))
  

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'example.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')

        )
    word=app.clf.computer_ratio(image1,image2)
    return flask.render_template(
        'result.html', has_result=True, result =(True, 'you did it.'), imagesrc1=embed_image_html(image1),imagesrc2=embed_image_html(image2),results=word
    )



# classify the Internet image    
@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        #if os.path.exists('./static/bb.jpg'):
        	#os.remove('./static/bb.jpg')

        string_buffer = cStringIO.StringIO(urllib2.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)
        #f=Image.open(string_buffer)
        #f.save(os.path.join(UPLOAD_FOLDER, 'bb.jpg'))
        #f.save(os.path.join('/home/flyvideo/zmj/flaskapp/static/','bb.jpg'))
        logging.info('Image: %s', imageurl)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'example.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )
    
    

    rst_img,result_time=app.clf.search_image(image)
    img_tmp=''''''
    for value in rst_img:
        img_tmp+='''<div class="col-xs-12 col-sm-9 col-md-1 clo-lg-3 marginDown" style="margin: 2% 2% 2% 0; "> '''+'''<a href="#">'''+'''<img src='''+url_for('static',filename=value)+''' width="100" height="100"/>'''+'''</a>'''+'''</div>
    '''
    return flask.render_template(
    'urlquery.html', has_result=True, result =(True, 'you did it.'), result_img=img_tmp,img_src=imageurl,time=result_time
    )
  

    
@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)
    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'example.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')

        )

    rst_img,result_time= app.clf.search_image(image)
    img_tmp=''''''
    for value in rst_img:
        img_tmp+='''<div class="col-xs-12 col-sm-9 col-md-1 clo-lg-3 marginDown" style="margin: 2% 2% 2% 0; "> '''+'''<a href="#">'''+'''<img src='''+url_for('static',filename=value)+''' width="100" height="100"/>'''+'''</a>'''+'''</div>
    '''
    return flask.render_template(
        'quer.html', has_result=True, result =(True, 'you did it.'), result_img=img_tmp,time=result_time
    )
 
cascade = cv.Load("/home/flyvideo/caffe-master/examples/Face_rect/haarcascade_frontalface_alt.xml")    
def detect(img, cascade):
    rects = cv.HaarDetectObjects(img, cascade, cv.CreateMemStorage(), 1.1, 2,cv.CV_HAAR_DO_CANNY_PRUNING, (255,255)) 
    if len(rects) == 0:
        return []
    result = []
    for r in rects:
        result.append((r[0][0], r[0][1], r[0][0]+r[0][2], r[0][1]+r[0][3]))
    if result[0][2]> 300 and result[0][3] > 300:
        return result
    else:
        return []

def dect_image(filename):
    img=cv.LoadImage(filename)
    gray=cv.CreateImage(cv.GetSize(img), 8, 1)
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)
    cv.EqualizeHist(gray,gray)
    rects = detect(img, cascade)
    if len(rects)!=0:
        rect=(rects[0][0],rects[0][1],rects[0][2]-rects[0][0],rects[0][3]-rects[0][1])
        cv.SetImageROI(img,rect)
    cv.SaveImage(filename,img)     

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data
    
def allowed_file(filename):
    return (
        '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS)


  


class faceSearch(object):
    default_args = {
        'model_def_file': (
            '{}/models/vgg_face_caffe/VGG_FACE_deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/vgg_face_caffe/VGG_FACE.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/models/vgg_face_caffe/my.npy'.format(REPO_DIRNAME)),
        
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 224
    default_args['raw_scale'] = 255


    def __init__(self, model_def_file, pretrained_model_file,mean_file,
                 raw_scale,image_dim, gpu_mode=1):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)) 
    
    def get_feature(self,img_list):
	test_num=len(img_list)
        averageImg = [129.1863,104.7624,93.5940]
        X=np.empty((test_num,3,224,224))
    	i =0
    	for line in img_list:
        	#im1=skimage.io.imread(line,as_grey=False)
        	image =skimage.transform.resize(line,(224, 224))*255
        	if image.ndim<3:
            		X[i,0,:,:]=image[:,:]-averageImg[0]
            		X[i,1,:,:]=image[:,:]-averageImg[1]
            		X[i,2,:,:]=image[:,:]-averageImg[2]
        	else:
            		X[i,0,:,:]=image[:,:,0]-averageImg[0]
            		X[i,1,:,:]=image[:,:,1]-averageImg[1]
            		X[i,2,:,:]=image[:,:,2]-averageImg[2]
        	i=i+1
    	out = self.net.forward_all(blobs=['fc7'],data=X)
    	feature=out['fc7']
    	feature=np.reshape(feature,(test_num,4096))
    	return feature


    def computer_ratio(self,img1,img2):#,A,G):
        try:
            starttime = time.time()
	    feat=faceSearch.get_feature(self,[img1,img2])
	    feat = normalize(feat, norm='l2')
	    #self.net.forward_all(data=img1)
	    #feat1 = np.float64(self.net.blobs['fc7'].data[:])
	    #feat1=np.reshape(feat1,(1,4096))
	    #img2=faceSearch.read_image(self,img2)
	    #out=self.net.forward_all(data=img2)
	    #feat2 = np.float64(self.net.blobs['fc7'].data[:])
	    #feat2=np.reshape(feat2,(1,4096))
             
    	    fea = common.pca_transform(pca, feat)
    	    distance = faceSearch.Verify(self,A, G, fea[0], fea[1])
	    mt=pw.cosine_similarity(feat[0],feat[1])
            threshold1=-30
	    threshold2=-80
	    if distance>threshold1:
		mt=mt+0.1
	    if distance<threshold2:
		mt=mt-0.05
    	    if mt>=0.4:
       	        results='相同'
    	    else:
                results='不同'
	    endtime = time.time()
            return results,'相似度' '%.2f' % mt,'%.1f' % distance,'时间' '%.3f' % (endtime - starttime),'秒'
	    
        except Exception as err:
            logging.info('Search error: %s', err)
            return (False, 'Something went wrong when compare the images')

    def Verify(self,A, G, x1, x2):
        x1.shape = (-1, 1)
        x2.shape = (-1, 1)
        ratio = np.dot(np.dot(np.transpose(x1), A), x1) + np.dot(np.dot(np.transpose(x2), A), x2) - 2 * np.dot(np.dot(np.transpose(x1), G), x2)
        return float(ratio)   

    
    def search_image(self, image):
        try:
            starttime = time.time()
            self.net.predict([image], oversample=True).flatten()
            feat = self.net.blobs['fc7'].data[:].flatten()
            endtime = time.time()
            f=h5py.File('./examples/vgg/faceNorml.mat')
            a=f['feats'][:]
	    f.close()
	    feat_vec=a.T
	    n,d=feat_vec.shape
	    score=zeros(n)
	     
            for i in range(n):
    	        vect=feat_vec[i,:]
    		score[i]=dot(feat,vect)
	    result=(-score).argsort()[:4]
            
	    g=open('./examples/vgg/ImgList.txt','r')
    	    list_img=g.readlines()
     	    img_fa=[]
	    for loc in result:
  	        img_fa.append(list_img[loc].strip())
            endtime = time.time()

            return img_fa,'%.3f' % (endtime - starttime)

        except Exception as err:
            logging.info('Search error: %s', err)
            return (False, 'Something went wrong when search the image. Maybe try another one?')


 
 # random to show the images   
def  setimage():
    imlist = os.listdir('./examples/Face_rect/static/')
    nbr_images = len(imlist)
    ndx = range(nbr_images)
    maxres = 12
    random.shuffle(ndx)
    img_tmp=''''''
    for i in  ndx[:maxres]:
        img_tmp+='''<div class="col-xs-12 col-sm-9 col-md-2 col-lg-3 marginDown" style="margin: 2% 2% 2% 0; "> '''+'''<a href="/classify_static?'''+'''imagepath='''+url_for('static',filename=imlist[i])+'''" >'''+'''<img src="'''+url_for('static',filename=imlist[i])+'''" width="100" height="100"/>'''+'''</a>'''+'''</div>
    ''' 
    return img_tmp


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
    tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-c', '--cpu',
        help="use cpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    #faceSearch.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = faceSearch(**faceSearch.default_args)
    app.clf.net.forward()

 
   #app.clf = ImagenetClassifier()
   #app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
   
    
