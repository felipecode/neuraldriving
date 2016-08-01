"""Neural Driving libs"""
from input_data_driving import DataSetManager
from config import *
from input_class import *
from train_class import *
from output_class import *

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





""" Initialize the input class to get the configuration """
config= configMain()


input_manager = Input(config.config_input)

manager = input_manager.get_dataset_manager()


training_manager = Train(config.config_train)


training_manager.build_network()
training_manager.build_optimization()

""" Initializing Session as variables that control the session """



training_manager.start_session()



"""Load a previous model if it is configured to restore """
training_manager.restore_session()




"""Training"""


""" Get the Last Iteration Trained """

initialIteration = training_manager.get_last_iteration()




output_manager = Output(config.config_output,input_manager,training_manager)

for i in range(initialIteration, config.number_iterations):



 
  """ Get the training batch """
  batch = manager.train.next_batch(config.batch_size)
  
  """Save the model every 300 iterations"""

  if i%300 == 0:

    training_manager.save_model(i)



  start_time = time.time()

  """ Run the training step and monitor its execution time """

  training_manager.run_train_step(batch)
  

  duration = time.time() - start_time

  """ With the current trained net, let the outputmanager print and save all the outputs """
  output_manager.print_outputs(i,duration,training_manager.get_feed_dict()) 






