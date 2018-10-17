import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn 
from sklearn.decomposition import PCA
import librosa, librosa.display
import urllib, contextlib
import IPython.display
import wave

plt.rcParams['figure.figsize'] = (14,4)

sample = '/Volumes/PARTH/music samples/Uproar.wav'
with contextlib.closing(wave.open(sample,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    length = frames / float(rate)

y, sr = librosa.load(sample, offset=(length/2) - 5, duration=10)
mfccs = librosa.feature.mfcc(y=y, sr=sr)
mfccs = np.delete(mfccs, 1, 0)

deltas = librosa.feature.delta(mfccs, order=1)
acceleration = librosa.feature.delta(mfccs, order=2)

print(np.shape(mfccs))

librosa.display.waveplot(y, sr=sr)
IPython.display.Audio(y, rate=sr)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
acceleration = sklearn.preprocessing.scale(acceleration, axis=1)
deltas = sklearn.preprocessing.scale(deltas, axis=1)

print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))

librosa.display.specshow(mfccs, sr=sr, x_axis='time', y_axis='off')

all_features = np.concatenate((mfccs, deltas, acceleration))
librosa.display.specshow(all_features, sr=sr, x_axis='time', y_axis='off')

print(all_features)
all_features = all_features.T

pca = PCA(n_components=5)
reduced = pca.fit(all_features)
print(reduced.components_.shape)
print(reduced.explained_variance_)

