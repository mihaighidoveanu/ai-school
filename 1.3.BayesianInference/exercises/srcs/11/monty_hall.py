import tensorflow as tf
from tensorflow_probability import edward2 as ed

N = 10000

car_door = ed.Categorical(probs=tf.constant([1. / 3., 1. / 3., 1. / 3.]), sample_shape = N, name = 'car_door')
picked_door = ed.Categorical(probs=tf.constant([1. / 3., 1. / 3., 1. / 3.]), sample_shape = N, name = 'picked_door')
preference = ed.Bernoulli(probs=tf.constant(0.5), sample_shape = N, name = 'preference')

host_choice = tf.where(tf.not_equal(car_door,  picked_door),
                       3 - car_door - picked_door,
                       tf.where(tf.equal(car_door, 2 * tf.ones(N, dtype=tf.int32)),
                                preference,
                                tf.where(tf.equal(car_door, tf.ones(N, dtype=tf.int32)),
                                         2 * preference,
                                         1 + preference)), name = 'host_choice')

#changed_door = 3 - host_choice - picked_door
changed_door = tf.subtract(tf.subtract(3, host_choice), picked_door, name = 'changed_door')

writer = tf.summary.FileWriter('./graphs_tfp', tf.get_default_graph())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    car_door_samples, picked_door_samples, changed_door_samples = sess.run([car_door, picked_door, changed_door])
writer.close()

print("probability to win of a player who stays with the initial choice:",
      (car_door_samples == picked_door_samples).mean())
print("probability to win of a player who switches:",
      (car_door_samples == changed_door_samples).mean())
