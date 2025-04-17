from IPython.display import display, HTML, Audio

def play_audio(waveform, rate=3000):
    return Audio(waveform, rate=rate)
