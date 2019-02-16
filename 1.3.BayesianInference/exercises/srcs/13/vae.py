import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
Variable = tf.Variable
import tensorflow_probability as tfp
tfd = tfp.distributions
ed = tfp.edward2
import warnings
warnings.filterwarnings('ignore')

xdim, d, zdim = 784, 200, 2
gen_net = tf.keras.Sequential([tf.keras.layers.Dense(d, activation=tf.nn.elu),
                               tf.keras.layers.Dense(d, activation=tf.nn.elu),
                               tf.keras.layers.Dense(xdim, activation=None)])
inference_net = tf.keras.Sequential([tf.keras.layers.Dense(d, activation=tf.nn.elu),
                                     tf.keras.layers.Dense(d, activation=tf.nn.elu),
                                     tf.keras.layers.Dense(2*zdim, activation=None)])

def generative_model(batch_size=None):
    z = ed.MultivariateNormalDiag(loc=tf.zeros([zdim]), scale_identity_multiplier=1.,
                                  sample_shape=batch_size, name='z')
    x = ed.Bernoulli(gen_net(z), name='x')
    return x

def variational_model(x):
    outs = inference_net(x)
    _loc, _scale = outs[:, :zdim], 1e-3 + tf.nn.softplus(outs[:, zdim:])
    z = ed.MultivariateNormalDiag(loc=_loc, scale_diag=_scale, name='z_posterior')
    return z

def get_loss(inputs):
    z = variational_model(inputs)
    energy = log_joint(z=z, x=inputs)
    entropy = tf.reduce_sum(z.distribution.entropy())
    return (-energy - entropy)

def replace_z(z):
    def interceptor(rv_constructor, *rv_args, **rv_kwargs):
        name = rv_kwargs.pop('name')
        if name == 'z':
            rv_kwargs['value'] = z
        return rv_constructor(*rv_args, **rv_kwargs)
    return interceptor
    
log_joint = ed.make_log_joint_fn(generative_model)

mnist = tf.keras.datasets.mnist
(x_train, _), _ = mnist.load_data()
x_train = np.rint((x_train / 255.).reshape(-1, xdim)).astype(np.float32)
assert x_train.max() <= 1 and x_train.min() >= 0 #assert correctly

bs = 64
dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle((len(x_train)))
dataset = dataset.batch(bs, drop_remainder=True)
dataset = dataset.prefetch(16)
data_iterator = dataset.make_initializable_iterator()
batch_x = data_iterator.get_next()

loss = get_loss(batch_x) / bs  # divide by bs since loss is summed.
opt = tf.contrib.opt.AdamWOptimizer(0, learning_rate=1e-3)
train = opt.minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(50): # 50 epochs; takes <3m on my GPU.
    sess.run(data_iterator.initializer)
    try:
        while True:
            _, _loss = sess.run([train, loss])
    except tf.errors.OutOfRangeError:
        print(_loss, ', end of epoch {}'.format(i))

# Generate imgs.
generated = generative_model(batch_size=64)
img = sess.run(generated).reshape(-1, 28, 28)
plt.figure(figsize=(16, 4))
for i in range(64):
    plt.subplot(4, 16, i+1)
    plt.imshow(img[i], cmap='Greys')
    plt.axis('off')
plt.savefig('vae1.png')
plt.show()

# Reconstruct imgs. Left is original, right is reconstructed.
batch = batch_x
var_z = variational_model(batch)
with ed.interception(replace_z(z=var_z)):
    recon = generative_model()
sess.run(data_iterator.initializer)
_batch, _recon = sess.run([batch, recon])
_batch, _recon = _batch.reshape(-1, 28, 28), _recon.reshape(-1, 28, 28)
plt.figure(figsize=(16, 4))
for i in range(32):
    plt.subplot(4, 16, 2*i+1)
    plt.imshow(_batch[i], cmap='Greys')
    plt.axis('off')
    plt.subplot(4, 16, 2*i+2)
    plt.imshow(_recon[i], cmap='Greys')
    plt.axis('off')
plt.savefig('vae2.png')
plt.show()

from scipy.stats import norm
N = 20
invcdf = norm.ppf(np.linspace(.1, .9, num=N))
zs = [[invcdf[i], invcdf[j]] for i in range(N) for j in range(N)]
with ed.interception(replace_z(z=zs)):
    recon = generative_model(N*N)
sess.run(data_iterator.initializer)
_recon = sess.run(recon).reshape(-1, 28, 28)
_batch, _recon = _batch.reshape(-1, 28, 28), _recon.reshape(-1, 28, 28)
plt.figure(figsize=(16, 16))
for i in range(N):
    for j in range(N):
        plt.subplot(N, N, N*j+i+1)
        plt.imshow(_recon[N*j+i], cmap='Greys')
        plt.axis('off')
plt.savefig('vae3.png')
plt.show()