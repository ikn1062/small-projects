import os
import tensorflow as tf


def main():
    # initialization of Tensors
    x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
    print(x)
    x = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(x)
    # Can use zeros and eyes
    x = tf.ones((3, 1))
    print(x)
    x = tf.eye(3)
    print(x)
    x = tf.random.normal((3, 3), mean=0, stddev=1)
    print(x)
    x = tf.random.uniform((3, 3), minval=0, maxval=1)
    print(x)
    x = tf.range(9)
    print(x)
    x = tf.range(start=1, limit=10, delta=2)
    print(x)
    x = tf.cast(x, dtype=tf.float32)
    print(x)

    # Mathematical Operations
    x = tf.constant([1, 2, 3])
    y = tf.constant([9, 8, 7])
    z = x + y
    print(z)
    z = x - y
    print(z)
    z = x / y
    print(z)
    z = x * y
    print(z)
    z = tf.tensordot(x, y, axes=1)
    # z = tf.reduce_sum(x*y, axis=0) also works
    print(z)
    z = x ** 5
    print(z)

    x = tf.random.normal((2, 3))
    y = tf.random.normal((3, 4))
    z = x @ y
    print(z)

    # Indexing
    x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
    print(x[:])
    print(x[1:])
    print(x[1:3])  # not inclusive of 3
    print(x[::2])  # skip every other element
    print(x[::-1])  # print in reverse

    indicies = tf.constant([0, 2])
    x_ind = tf.gather(x, indicies)
    print(x_ind)

    x = tf.constant([[1, 2], [3, 4], [5, 6]])
    print(x[0, :])
    print(x[0:2, :])

    # Reshaping
    x = tf.range(9)
    x = tf.reshape(x, (3, 3))
    print(x)

    x = tf.transpose(x, perm=[1, 0])
    print(x)



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
