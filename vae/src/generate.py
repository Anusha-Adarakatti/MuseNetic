import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import glob
import imageio

def generate_and_save_images(model, epoch, test_sample, save):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(18, 15))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        wave = np.asarray(predictions[i])
        librosa.display.waveplot(wave[0], sr=3000)
    plt.savefig(f'{save}_{epoch:04d}.png')
    plt.show()

def create_animation(file_prefix, output_name):
    with imageio.get_writer(output_name, mode='I') as writer:
        filenames = sorted(glob.glob(f'{file_prefix}*.png'))
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

def inference(test_dataset, model):
    save_music = []
    for test in test_dataset:
        mean, logvar = model.encode(test)
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(z)
        for pred in predictions:
            wave = np.asarray(pred)
            save_music.append(wave)
    return save_music
