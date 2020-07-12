import tensorflow as tf
# print(tf.__version__)


class GLU(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, dim=-1, inp_shape=(128, 100, 64)):
        super(GLU, self).__init__()
        self.dim = dim
        self.filter = filters // 2
        self.kernel_size = kernel_size
        self.sig = tf.sigmoid

    # Function to Slice Tensor Equally along Last Dim
    def equal_slice(self, x):
        ndim = len(x.shape)
        slice_idx = x.shape[self.dim] // 2
        if ndim == 3:
            linear_output = x[:, :, :slice_idx]
            gated_output = x[:, :, slice_idx:]
        elif ndim == 4:
            linear_output = x[:, :, :, :slice_idx]
            gated_output = x[:, :, :, slice_idx:]
        elif ndim == 5:
            linear_output = x[:, :, :, :, :slice_idx]
            gated_output = x[:, :, :, :, slice_idx:]
        else:
            raise ValueError(
                "This GLU Activation only support for 1D, 2D, 3D Conv, but the Input's Dim is={}".format(ndim))

        # Return the 2 slices
        return linear_output, gated_output

    def call(self, inputs, **kwargs):
        assert inputs.shape[self.dim] % 2 == 0

        # Slicing the Tensor in 2 Halfs
        lin_out_slice, gated_out_slice = self.equal_slice(inputs)

        # Applying Sigmoid Activation to 2nd Slice
        siggat_out_slice = self.sig(gated_out_slice)

        # Returning Element-wise Multiply of two Slices
        return lin_out_slice * siggat_out_slice


'''
# * Testing-Code *
model = GLU(128, 3, 2)
batch_size, n, m = 128, 100, 128
A = tf.Variable(tf.random_normal(shape=(batch_size, n, m)))

y = model(A)
init = tf.global_variables_initializer()
print(y.shape)
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer)
    sess.run(init)
    print("Input:", sess.run(A))
    print("Output:", sess.run(y))

model.summary()
# if input.dim() == 0:
#     raise RuntimeError("glu does not support scalars because halving size must be even")
'''