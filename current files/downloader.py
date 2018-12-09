import sys
import os
import urllib
import py7D
import hdf5_getters

def update(curr):
    print('\r[{0}] {1}%'.format('#' * (curr / 10), curr))

if __name__ == "__main__":

    file_path = sys.argv[1]
    i = 0.0
    for folder_name in os.listdir(file_path):
        inner_path = file_path + '/' + folder_name
        if (os.path.isdir(inner_path)):
            for folder2 in os.listdir(inner_path):
                inner_path2 = inner_path + '/' + folder2
                if (os.path.isdir(inner_path2)):
                    for file_f in os.listdir(inner_path2):
                        previewFilePath = inner_path2 + '/' + os.path.splitext(file_f)[0] + '.mp3'
                        if not (not file_f.endswith('h5') or os.path.isfile(previewFilePath)):
                            h5FilePath = inner_path2 + '/' + file_f
                            try:
                                h5 = hdf5_getters.open_h5_file_read(h5FilePath)
                                id7Digital = hdf5_getters.get_track_7digitalid(h5)
                                h5.close()

                                url = py7D.preview_url(id7Digital)
                                urlretrieve = urllib.urlretrieve(url, previewFilePath)
                            except Exception as e:
                                print("Error occurred")

                        if file_f.endswith('h5'):
                            sys.stdout.write("\r%d%%" % int(i/7620 * 100))
                            sys.stdout.flush()
                            i += 1