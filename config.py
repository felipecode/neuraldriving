class configInput:

	def __init__(self,input_size_main,number_images_epoch_main,number_images_epoch_val_main,number_steering_bins_main):

		self.train_db_path = 'datasets/Ruudskogen_1_cam_dist_test/'
		self.validation_db_path = 'datasets/Ruudskogen_1_cam_dist_test_val/'
		self.balance_data = 1 # IMPLEMENT
		self.compute_average = 0
		self.input_size = input_size_main
		self.positions = [25,26]
		self.number_images_epoch = number_images_epoch_main
		self.number_images_epoch_val = number_images_epoch_val_main
		self.number_steering_bins = number_steering_bins_main


class configTrain:

	def __init__(self,input_size_main,output_size_main,dropout_main):
		
		self.input_size = input_size_main
		self.output_size = output_size_main
		self.dropout = dropout_main
		self.loss_function = 'KL_divergence'  # Chose between, KL_divergence, MSE,
		self.learning_rate = 1e-3
		self.lambda_l2 = 1e-4
		self.output_weigth = [1]
		self.network_name = 'alexdrop'
		self.restore = False

		""" Add sannity Checks """

		if self.restore not in (True, False):
			raise Exception('Wrong restore option. (True or False)')


class configOutput:


	def __init__(self,models_path_main,batch_size_main,batch_size_val_main,number_images_epoch_main,number_images_epoch_val_main):

		self.print_interval = 2
		self.summary_writing_period = 20
		self.validation_period = 200  # I consider validation as an output thing since it does not directly affects the training in general
		self.feature_save_interval  = 100
		
		self.models_path =  models_path_main
		self.variable_names = ['Steer']
		self.number_images_epoch = number_images_epoch_main
		self.number_images_epoch_val = number_images_epoch_val_main
		self.batch_size = batch_size_main
		self.batch_size_val = batch_size_val_main

		""" Feature Visualization Part """
		#self.histograms_list=[]
		#self.features_list=["B_relu","A_relu"]
		#self.features_opt_list=[["S1_conv1", 0],["S1_conv1", 63],["S3_incep1",-1]]
		#self.opt_every_iter=100
		#self.save_features_to_disk=True




class configMain:


	def __init__(self):


		self.number_steering_bins = 8
		self.batch_size = self.number_steering_bins*5 # HAS TO BE MULTIPLE OF NUMBER OF STEERING BINS
		self.batch_size_val = 50
		self.number_images_epoch = 200* self.batch_size
		self.number_images_epoch_val = 50* self.batch_size_val
		self.n_epochs = 500
						
		self.input_size = (210,280,3)
		self.output_size = (2)
		self.dropout = [1,1,1,1,1,1,0.5,0.5]
		#self.dropout = [1,1,1,1,1,1,1,1]
		
		self.models_path = 'models/MultiVariable_regress_divergence2/'

		# Just in case we are on the discrete problem


		self.number_iterations =self.n_epochs*self.number_images_epoch


		self.config_input = configInput(self.input_size,self.number_images_epoch,self.number_images_epoch_val,self.number_steering_bins)
		self.config_train = configTrain(self.input_size,self.output_size,self.dropout)
		self.config_output = configOutput(self.models_path,self.batch_size,self.batch_size_val,self.number_images_epoch,self.number_images_epoch_val)




