import tensorflow as tf

""" The KL divergence assumes two variables per loss """

def KL_divergence(outputs,ground_truths,regularizer,config):



	variables = tf.split(1, config.output_size , ground_truths)

	variables_outputs  = tf.split(1, config.output_size , outputs)
	variables_loss_vec = []



  	for i in range(0,config.output_size,2):
	    
	    std_var = variables_outputs[i]
	    mean_var = variables_outputs[i+1]
	    std_var_gt = variables[i]
	    mean_var_gt = variables[i+1]
	    variable_loss = tf.log(tf.abs(tf.truediv(std_var_gt,std_var+0.001))+0.01)
	    variable_loss = tf.add(variable_loss,tf.truediv(tf.add(tf.pow(std_var,2), tf.pow(tf.sub(mean_var,mean_var_gt),2)),2*tf.pow(std_var_gt+0.0001,2))) - 0.5
	    variables_loss_vec.append(variable_loss)
	    if i==0:
	      loss_function = config.output_weigth[i/2]*variable_loss
	    else:
	      loss_function = tf.add(config.output_weigth[i/2]*variable_loss,loss_function)

  	loss_function = loss_function  + config.lambda_l2*regularizer/(100)

  	return loss_function,variables_loss_vec