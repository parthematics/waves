import numpy as np
import librosa
import pickle
import sys
import os

SONG_LENGTH = 30000
SIZE_HAMMING = 100
STRIDE_HAMMING = 40

def update(curr):
    print '\r[{0}] {1}%'.format('#' * (curr / 10), curr)

def replace_old(song, old, new, instance):
    li = song.rsplit(old, instance)
    return new.join(li)

def preprocessor(song_path, file_path):
    featuresArray = []
    for i in range(0, SONG_LENGTH, STRIDE_HAMMING):
        if i + SIZE_HAMMING <= SONG_LENGTH - 1:
            y, sr = librosa.load(song_path, offset=i / 1000.0, duration=SIZE_HAMMING / 1000.0)

            # make/display a mel-scaled power spectrogram
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

            # convert to log scale (dB)
            log_S = librosa.logamplitude(S, ref_power=np.max)

            mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=13)
            featuresArray.append(mfcc)

            if len(featuresArray) >= 599:
                break

    with open(file_path, 'w') as f:
        f.write(pickle.dumps(featuresArray))
    f.close()

if __name__ == "__main__":

    count = 0.0
    walk_directory = sys.argv[1]

    for root, dirs, files in os.walk(walk_directory):
        for file_name in files:
            if file_name.endswith('.au'):
                file_path = os.path.join(root, file_name)
                ppFileName = replace_old(file_path, ".au", ".pp", 1)

                try:
                    preprocessor(file_path, ppFileName)
                except Exception as e:
                    print "Error accured" + str(e)

            if file_name.endswith('au'):
                sys.stdout.write("\r%d%%" % int(count / 7620 * 100))
                sys.stdout.flush()
                count += 1
