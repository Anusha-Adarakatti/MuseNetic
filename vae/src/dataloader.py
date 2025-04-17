import os
import numpy as np
import tensorflow as tf
import librosa
from .config import BASE_PATH, SAMPLING_RATE

def DatasetLoader(class_):
    music_list = np.array(sorted(os.listdir(os.path.join(BASE_PATH, class_))))
    train_music_1 = list(music_list[[0,52,19,39,71,12,75,85,3,45,24,46,88]])
    train_music_2 = list(music_list[[4,43,56,55,45,31,11,13,70,37,21,78]])
    TrackSet_1 = [os.path.join(BASE_PATH, class_, x) for x in train_music_1]
    TrackSet_2 = [os.path.join(BASE_PATH, class_, x) for x in train_music_2]
    return TrackSet_1, TrackSet_2

def load(file_):
    data_, _ = librosa.load(file_, sr=SAMPLING_RATE, offset=0.0, duration=30)
    data_ = data_.reshape(1, 90001)
    return data_

def map_data(filename):
    return tf.compat.v1.py_func(load, [filename], [tf.float32])[0]

def create_datasets(TrackSet_1, TrackSet_2, batch_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(TrackSet_1)
        .map(map_data, num_parallel_calls=AUTOTUNE)
        .shuffle(3)
        .batch(batch_size)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(TrackSet_2)
        .map(map_data, num_parallel_calls=AUTOTUNE)
        .shuffle(3)
        .batch(batch_size)
    )
    return train_dataset, test_dataset
