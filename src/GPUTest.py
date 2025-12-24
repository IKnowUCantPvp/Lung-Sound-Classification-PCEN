import tensorflow as tf

# List all physical devices recognized by TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print("GPUs: ", gpus)

if gpus:
    # If at least one GPU is found, print some details
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Memory growth enabled for", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

import time

# Create a large random matrix on GPU
with tf.device('/GPU:0'):
    A = tf.random.normal((10000, 10000))
    B = tf.random.normal((10000, 10000))

start = time.time()
with tf.device('/GPU:0'):
    C = tf.matmul(A, B)
end = time.time()

print("Result shape:", C.shape)
print(f"Time taken with GPU: {end - start:.4f} seconds")
