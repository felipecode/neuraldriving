import loss_functions
import tensorflow as tf

class Train(object):

 
	def __init__(self,config):



		self._input_data = tf.placeholder("float",shape=[None,config.input_size[0]*config.input_size[1]*config.input_size[2]], name="input_image")
		self._output_data = tf.placeholder("float", name="output_data")
		self._dout = tf.placeholder("float",shape=[len(config.dropout)])


		
		self._create_structure = __import__(config.network_name).create_structure


		
		self._loss_function = getattr(loss_functions, config.loss_function ) # The function to call 

	def build_network(self,config):


		self._output_network,self._regularizer = self._create_structure(tf, self._input_data,config.input_size,self._dout,config)

		self._loss,self._variable_error = self._loss_function(self._output_data,self._output_network,self._regularizer,config)

		

	def build_optimization(self,config):

		# Add more optimizers possibilities

		self._train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(self._loss)


	@property
	def get_train_step(self):
		return self._train_step

	@property
	def get_loss(self):
		return self._loss
	@property
	def get_variable_error(self):
		return self._variable_error
	
	


