import tensorflow as tf
#IMDb internet Movie Database 互联网电影资料库
keras=tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = keras.load_data()
print(len(train_data),len(test_data))
# 自然语言 分词