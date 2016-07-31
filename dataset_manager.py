import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2
from PIL import Image     
import random
import bisect
import os.path
from dataset import *


class DataSetManager(object):

  def compute_average_number(self,db,input_size):
    
    datum = caffe_pb2.Datum()

    print input_size
    mean_image = np.zeros((input_size[0], input_size[1],input_size[2]))

    count_images = 0

    for key, value in db.RangeIter():
      datum.ParseFromString(value)

      label = datum.label
      data = caffe.io.datum_to_array(datum)

      #CxHxW to HxWxC in cv2
      image = np.transpose(data, (1,2,0))
      mean_image =  mean_image + image
      count_images +=1
      print count_images
      #break



    mean_image = mean_image/count_images

    return mean_image,count_images

  def read_image_number(self,db):
    pass


  def __init__(self,config):


    self.db = leveldb.LevelDB(config.train_db_path)
    self.db_val = leveldb.LevelDB(config.validation_db_path)



    if config.compute_average:
      mean_image = self.compute_average_number(self.db,input_size)
      np.save(config.train_db_path +'meanimage.npy',mean_image)
    else:
      mean_image = np.load(config.train_db_path + 'meanimage.npy')

    #print mean_image



    

    self.input_size = config.input_size

    self.train = DataSet(self.db,config.train_db_path,config.input_size,mean_image,config.number_images_epoch,config.number_steering_bins,config.positions)
    self.validation = DataSet(self.db_val,config.validation_db_path,config.input_size,mean_image,config.number_images_epoch_val,config.number_steering_bins,config.positions)  




#from config import *


#config = configMain()


#dataset = DataSetManager(config.train_db_path,config.validation_db_path,5800,3400,config.input_size,1)

#dataset = DataSetManager('/home/adas/caffe/TORCS_Training_test','/home/adas/caffe/tools/TORCS_Training_1F_full/',200,1700,[210,280,3])

#batch = dataset.train.next_batch(5)

#print batch[1]
