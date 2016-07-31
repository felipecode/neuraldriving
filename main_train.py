"""Neural Driving libs"""
from input_data_driving import DataSetManager
from config import *
from input_class import *
from train_class import *
from output import *

"""Structure"""
import sys
sys.path.append('structures')
sys.path.append('utils')
from alexdrop import create_structure

"""Core libs"""
import tensorflow as tf
import numpy as np




"""Python libs"""
import os
from optparse import OptionParser
from PIL import Image
import subprocess
import time




""" Initialize the input class to get the configuration """
config= configMain()


input_manager = Input(config.config_input)

manager = input_manager.get_dataset_manager()


training_manager = Train(config.config_train)


training_manager.build_network(config.config_train)
training_manager.build_optimization(config.config_train)

""" Initializing Session as variables that control the session """


global_step = tf.Variable(0, trainable=False, name="global_step")

training_manager.start_session()

#sess = tf.InteractiveSession()
#sess.run(tf.initialize_all_variables())
#saver = tf.train.Saver(tf.all_variables())

if config.restore == True

"""Load a previous model if restore is set to True"""


if config.restore:
  if not os.path.exists(config.models_path):
  os.mkdir(config.models_path)
  ckpt = tf.train.get_checkpoint_state(config.models_path)
  if ckpt:
    print 'Restoring from ', ckpt.model_checkpoint_path  
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
  ckpt = 0





"""Training"""


""" Get the Last Iteration Trained """

if ckpt:
  initialIteration = int(ckpt.model_checkpoint_path.split('-')[1])
else:
  initialIteration = 1





feedDict = {dout:config.dropout}

training_start_time =time.time()


output_manager = Output(config.config_output,input_manager,training_manager,sess)

for i in range(initialIteration, number_iterations):



 
  """ Get the training batch """
  batch = dataset.train.next_batch(config.batch_size)
  
  """Save the model every 300 iterations"""

  if i%300 == 0:
    saver.save(sess, config.models_path + 'model.ckpt', global_step=i)
    print 'Model saved.'

  


  start_time = time.time()

  """ Run the training step and monitor its execution time """

  feedDict.update({x: batch[0], y_: batch[1]})

  sess.run(training_manager.get_train_step(), feed_dict=feedDict)

  duration = time.time() - start_time

  """ With the current trained net, let the outputmanager print and save all the outputs """
  output_manager.print_outputs(i) 






