# simple_dense_layer
Simple dense layer to study how does it works, we know that the backward propagation algorithms compare of the network or layer weight from current to previous or compare of the result from the current to previous input. We add some small value as a learning rates adjusting by optimizers and assume that our layer has weight and bias ( W + b ). We now remove bias variable because to see the change of weight as a linear result running but the networks or the layer bias value can be any value and summing up to response the input by win the weight accumulating from its self-working.

### My custom Dense layer ####

It is simply we talking about custom layer where it has weight and bias ( W + b ) now we leave the bias to see the loss value changing as the result in result bullet. The function of MyDenseLayer class composed of initial, create super class and copy the target output value. The custom dense layer with target 10 will result in ( input shape X 10 ) as it is the result from input.shape[-1] X num_outputs by default. The build function is stored wieght when we create and we can inistail with custom value, multiple calling of the same class kernel value accumulating not reimplement of the calss with new value. The call function is logical objective we implementing and get_weight() and set_weight() are assign and copy weight value as a return.
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

Simple input from chemical bonds, it indicated 0 padded make shape sticking and matched with label with the same couting or similar in some dimension.
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

Input to output prediction is simply call layer with new value the weight is change with very small value but it is very slow for task running they using guids called optimizers that shortern of learning process not only gradient descents but step optimize and loss function return value handling.
```
layer = MyDenseLayer(10)
data = layer(input[0])
score = tf.nn.softmax(data[0])
target_1 = int(tf.math.argmax(score))
```

### Sample prediction and loss function value ###

It is not finish learning but we to see the first step all input they are finding mean and sticking to the mean value, when it see some different it start change one output from number of output or small value because some number indicated from mean and continue until indicated the patterns. The loss value we simply calculation from label prediction and label true that is some backhoff update ( loss value change, find some matching patterns ).
```
6: 6: 6: 6: -322.51672
```

## weight output ###

You notice sine the first value from the weight as the array return { 118.57729, 118.67729, 118.77729 ... } it matched with optimizer weight + a small value from call the layer { 118.25526, 118.355255, 118.45525 ... }
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

### Files and Directory ###
1. sample.py : sample codes
2. Figure_24.png : loss value from loss function show as the result from running time optimizing.
3. README.md : read me file

### Result : Loss value ###

![Alt text](https://github.com/jkaewprateep/simple_dense_layer/blob/main/Figure_24.png?raw=true "Title")
