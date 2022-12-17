
import tensorflow as tf

import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
elements = { "None": 0, "X": 1, "A": 2 }
input = [ [[ 180, 1 ], [ 180, 1 ], [ 360, 2 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], 
          [[ 120, 1 ], [ 120, 1 ], [ 120, 1 ], [ 360, 2 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]],
		  [[ 109.28, 1 ], [ 0, 1 ], [ 0, 1 ], [ 0, 1 ], [ 360, 2 ], [ 0, 0 ], [ 0, 0 ]],
		  [[ 90, 1 ], [ 90, 1 ], [ 90, 1 ], [ 90, 1 ], [ 90, 1 ], [ 90, 1 ], [ 360, 2 ]] ]	#  shape=(4, 7, 2), dtype=float32)

label = [ 180, 120, 109.28, 90 ]	  
input = tf.constant( input )
step = 0

history = { "loss_value" : [], "step" : [] }

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, num_outputs):
		super(MyDenseLayer, self).__init__()
		self.num_outputs = num_outputs

	def build(self, input_shape):
		self.kernel = self.add_weight("kernel",
									  shape=[int(input_shape[-1]),
											 self.num_outputs])

	def call(self, inputs):
		return tf.matmul(inputs, self.kernel)
	
	def get_weight(self ):
		return self.kernel
		
	def set_weight(self, weight ):
		self.kernel = self.kernel + weight

layer = MyDenseLayer(10)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
layer = MyDenseLayer(10)

data = layer(input[0])
score = tf.nn.softmax(data[0])
target_1 = int(tf.math.argmax(score))

data = layer(input[1])
score = tf.nn.softmax(data[0])
target_2 = int(tf.math.argmax(score))

data = layer(input[2])
score = tf.nn.softmax(data[0])
target_3 = int(tf.math.argmax(score))

data = layer(input[3])
score = tf.nn.softmax(data[0])
target_4 = int(tf.math.argmax(score))

optimizer = 0.1 * tf.ones([ 10, 2 ])
optimizer = tf.constant( optimizer, dtype=tf.float32 )

loss_value = 0

for step in range( 1000 ):
	step = step + 1

	for i in range( input.shape[0] ):
		data = layer(input[i])
		score = tf.nn.softmax(data[0])
		target_predict = int(tf.math.argmax(score))

		if label[i] != target_predict :
			log_y_pred = tf.math.log( 1.0 * target_predict)
			y_true = label[i]
			
			log_y_pred = tf.cast( log_y_pred, dtype=tf.float32 )
			y_true = tf.cast( y_true, dtype=tf.float32 )
			
			data = layer(optimizer)
			
			loss_value = -tf.math.multiply_no_nan(x=log_y_pred.numpy(), y=y_true.numpy())
			
		else:
			pass
		
		history["loss_value"].append(loss_value)
		history["step"].append(step)
		
		print( str( target_1 ) + ": " + str( target_2 ) + ": " + str( target_3 ) + ": " + str( target_4 ) + ": " + str( loss_value.numpy() ) )
		layer.set_weight( tf.reshape( optimizer, ( 2, 10 ) ) )
		print( layer.get_weight() )


plt.plot( history["loss_value"], history["step"] )
plt.show()
