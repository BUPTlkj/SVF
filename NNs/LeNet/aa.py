import tensorflow as tf

## 创建一个维度为 (3, 6, 7, 8) 的 NCHW 格式整数张量
nchw_tensor = tf.random.uniform([3, 4, 1, 2], minval=0, maxval=100, dtype=tf.int32)

# 将 NCHW 转换为 NHWC
nhwc_tensor = tf.transpose(nchw_tensor, perm=[0, 2, 3, 1])

print("NC 格式的张量：\n", nchw_tensor)
print("NHWC 格式的张量：\n", nhwc_tensor)
