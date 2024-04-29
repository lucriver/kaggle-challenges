# set GPU to be visible
import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

# ---------------------------------------

# initialize tensorflow to use GPU
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
    
tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
with tf.device(logical_gpus[0]):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Run on the GPU
c = tf.matmul(a, b)
print(c)

# -----------------------------

xgboost_params_grid = {
    'n_estimators': [200, 300, 400, 500, 600, 700, 800],
    'learning_rate': [0.1, 0.01, 0.001]
}