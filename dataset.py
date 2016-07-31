import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2
from PIL import Image     
import random
import bisect
import os.path





class DataSet(object):
  def __init__(self, db,db_path,input_size,mean_image,epoch_size,number_steering_levels,positions):
   
    self._db = db
    self._mean_image = mean_image

    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._input_size= input_size
    self._epoch_size = epoch_size
    self._number_steering_levels = number_steering_levels
    self._positions = positions  # The positions of the desired variables on the dataset

    if os.path.isfile(db_path + 'keys.npy'): 
      self._splited_keys = np.load(db_path + 'keys.npy')
    else:
      self._keys,self._steerings = self.get_key_steering_by_steering()
      self._splited_keys = self.partition_keys_by_steering(self._steerings,self._keys)
      np.save(db_path +'keys.npy',self._splited_keys)


    self._rand_list = self.get_keys_for_epoch() #rand_list  # This is the acess order that is going to be made on the dataset




  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def epochs_completed(self):
    return self._epochs_completed


  def get_key_steering_by_steering(self):
    datum = caffe_pb2.Datum()
    keys=[]
    steerings =[]



    for key, value in self._db.RangeIter():

      datum.ParseFromString(value)

      label = datum.label
      data = caffe.io.datum_to_array(datum)
      steer = datum.float_data[self._positions[2]]
      
      # Get the position to insert
      position = bisect.bisect_left(steerings,steer)



      keys.insert(position,key)
      steerings.insert(position,steer)

    
    return keys,steerings

  def partition_keys_by_steering(self,steerings,keys):

    max_steer = max(steerings)
    min_steer = min(steerings)

    #print steerings
    
    steerinterval =  (max_steer - min_steer)/(self._number_steering_levels)

    iter_value = min_steer + steerinterval
    iter_index = 0
    splited_keys = []
    print 'len steerings'
    print len(steerings)
    for i in range(0,len(steerings)):

      if steerings[i] >= iter_value:
        # We split

        splited_keys.append(keys[iter_index:i])
        iter_index=i
        iter_value = iter_value + steerinterval
        print 'split on ', i
        

    #splited_keys.append(keys[i:len(steerings)])


    return splited_keys


  def get_keys_for_epoch(self):

 
    samples_per_split = self._epoch_size/ (self._number_steering_levels)
    print ' n samples'
    print len(self._splited_keys)
    final_keys = []
    print 'Getting Keys'
    for list_splited in self._splited_keys:
      print len(list_splited)
      range_split = []
      for i in range(0,samples_per_split):
        rand_pos = random.randint(0,len(list_splited)-1)
        range_split.append(list_splited[rand_pos])
      final_keys.append(range_split)

    #print final_keys
    return final_keys





  def turn_logits(self,min_value,max_value,number_logits,data):


    interval = float(max_value-min_value)/float(number_logits)
    #print interval

    labels =np.zeros((number_logits))

    for i in range(1,number_logits+1):

      value = min_value + interval*i
      #print 'steer'
      #print datum.float_data[15]
      #print 'value'
      #print value


      if data <= value:
        labels[i-1] = 1
        #print 'break'
        break
    return labels


  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    datum = caffe_pb2.Datum()
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch  >= self._epoch_size:
      # Finished eph
      print 'end epoch'
      self._epochs_completed += 1
      # Shuffle the data
      """ Shufling all the Images with a single permutation """
      #rand_list = range(1,self._num_examples)
      #random.shuffle(rand_list)
      #rand_list = map(str, rand_list)
      #rand_list = map(lambda i: i.zfill(8),rand_list)
      #self._rand_list = rand_list[1:500]
      #random.shuffle(self._rand_list)
      # Start next epoch
      start = 0
      self._rand_list = self.get_keys_for_epoch()
      self._index_in_epoch = batch_size
      assert batch_size <= self._epoch_size



    if batch_size >  (self._epoch_size - self._index_in_epoch):
      batch_size = self._epoch_size - self._index_in_epoch




    images = np.empty((batch_size, self._input_size[0], self._input_size[1],self._input_size[2]))

    labels = np.zeros((batch_size,len(self._positions)))   # logits version
    #labels = np.empty((batch_size,3))

    # print ' len list '
    #print len(self._rand_list)
   # print 'epoch size'
   # print self._epoch_size

    #print batch_size

    for outer_n in range(self._number_steering_levels): # for each steering.

      for inner_n in range(batch_size/self._number_steering_levels):  # The number of images for each steering
    
        #print start
        #print n
        #print start+n
        # self._rand_list[start+n]

        value = list(self._db.RangeIter(key_from=self._rand_list[outer_n][start/self._number_steering_levels+inner_n], key_to=self._rand_list[outer_n][start/self._number_steering_levels+inner_n]))

        """ GAMBISVIOLENTIS """
        if len(value)<1:
          continue




        datum.ParseFromString(value[0][1])

        #print datum.float_data[0]

        # labels[n][0] = abs(min(datum.float_data[15],0))
        # labels[n][1] = max(datum.float_data[15],0)
        # labels[n][2] = datum.float_data[16]
        # labels[n][3] = datum.float_data[17]


        #labels[n][0] = datum.float_data[15]
        #positions_used = [3,4,5,6,7,8,9,10,13,14,15,16,17,18,25,26]
        #positions_used = [3,4,17,18,25,26]
        

        for i in range(0,len(positions_used)):
          labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][i] = datum.float_data[self._positions[i]]


        #last = len(positions_used) 

        # if( datum.float_data[7] > 0):
        #   labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last] = 1
        #   labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last +1] = 0
        # else:
        #   labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last] = 0
        #   labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last +1] = 1

        # if( datum.float_data[8] > 0):
        #   labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last +2] = 1
        #   labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last +3] = 0
        # else:
        #   labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last +2] = 0
        #   labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last +3] = 1


        # #if ((datum.float_data[6] + datum.float_data[7]) >1.0 or (datum.float_data[6] + datum.float_data[7]) <1.0):
        #   #print [datum.float_data[4],datum.float_data[2],datum.float_data[3]] 

        # labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last +4] = datum.float_data[3]
        # labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last +5] = datum.float_data[4]
        # labels[outer_n*(batch_size/self._number_steering_levels) + inner_n][last +6] = datum.float_data[5]

        # datum.float_data[positions_used[p]]
        #

        data = caffe.io.datum_to_array(datum)
         
        #CxHxW to HxWxC in cv2
        image = np.transpose(data, (1,2,0))

        image = np.asarray(image)
        image = image.astype(np.float32)
        image = image - self._mean_image
        image = np.multiply(image, 1.0 / 127.0)

        images[outer_n*(batch_size/self._number_steering_levels) + inner_n] = image;


      #print images[n]
      #images[n] = Image.fromarray(image, 'RGB')

    """ TODO : CHANGE THIS , this is just a bad function on create_structure problem """
    images = images.reshape(images.shape[0],images.shape[1] * images.shape[2]*images.shape[3])
    

    return images, labels


