# simple_dense_layer
Simple dense layer to study how does it works


### My custom Dense layer ####
```
class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, num_outputs):
		super(MyDenseLayer, self).__init__()
		self.num_outputs = num_outputs

	def build(self, input_shape):
		self.kernel = self.add_weight("kernel",
                       shape=[int(input_shape[-1]), self.num_outputs])

	def call(self, inputs):
		return tf.matmul(inputs, self.kernel)
	
	def get_weight(self ):
		return self.kernel
		
	def set_weight(self, weight ):
		self.kernel = self.kernel + weight

layer = MyDenseLayer(10)
```

### Input ###
```
elements = { "None": 0, "X": 1, "A": 2 }
input = [ [[ 180, 1 ], [ 180, 1 ], [ 360, 2 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]], 
          [[ 120, 1 ], [ 120, 1 ], [ 120, 1 ], [ 360, 2 ], [ 0, 0 ], [ 0, 0 ], [ 0, 0 ]],
          [[ 109.28, 1 ], [ 0, 1 ], [ 0, 1 ], [ 0, 1 ], [ 360, 2 ], [ 0, 0 ], [ 0, 0 ]],
          [[ 90, 1 ], [ 90, 1 ], [ 90, 1 ], [ 90, 1 ], [ 90, 1 ], [ 90, 1 ], [ 360, 2 ]] ]	#  shape=(4, 7, 2), dtype=float32)

label = [ 180, 120, 109.28, 90 ]
		  
input = tf.constant( input )
```

### Input to output prediction ###
```
layer = MyDenseLayer(10)
data = layer(input[0])
score = tf.nn.softmax(data[0])
target_1 = int(tf.math.argmax(score))
```

### Sample prediction and loss function value ###
```
6: 6: 6: 6: -322.51672
```

## weight output ###
```
6: 6: 6: 6: -195.80348
tf.Tensor(
[[118.57729  118.25526  118.69474  119.2015   118.44541  119.07255
  119.270836 118.0884   118.831604 119.230415]
 [118.77038  119.30522  118.62853  118.32755  119.24682  119.24558
  119.060715 118.68435  118.02784  118.90322 ]], shape=(2, 10), dtype=float32)
6: 6: 6: 6: -161.25836
tf.Tensor(
[[118.67729  118.355255 118.79474  119.3015   118.54541  119.17255
  119.370834 118.1884   118.9316   119.330414]
 [118.87038  119.40522  118.72853  118.42755  119.34682  119.34558
  119.16071  118.78435  118.12784  119.00322 ]], shape=(2, 10), dtype=float32)
6: 6: 6: 6: -322.51672
tf.Tensor(
[[118.77729  118.45525  118.89474  119.4015   118.64541  119.272545
  119.47083  118.2884   119.0316   119.43041 ]
 [118.970375 119.50522  118.82853  118.52755  119.446815 119.44558
  119.26071  118.884346 118.22784  119.10322 ]], shape=(2, 10), dtype=float32)
```
