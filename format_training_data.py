import numpy as np
import os
from numpy import array
import pickle
import sys

def update(curr):
    print ('\r[{0}] {1}%'.format('#' * (curr / 10), curr))

labels_dict = {
    'Blues': 0,
    'Classical': 1,
    'Country': 2,
    'Disco': 3,
    'Hiphop': 4,
    'Jazz': 5,
    'Metal': 6,
    'Pop': 7,
    'Reggae': 8,
    'Rock': 9,
}

if __name__ == "__main__":

    data_dict = {}
    all_labels = []
    i = 0.0

    walk_directory = sys.argv[1]
    print('walk_dir = ' + walk_directory)

    for root, dirs, files in os.walk(walk_directory):
        for file in files:
            if file.endswith('pp'):
                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    try:
                        song_id = os.path.splitext(file)[0]
                        content = f.read()
                        pp = pickle.loads(content)
                        pp = np.asarray(pp)
                        data_dict[song_id] = pp

                        labelName = file.split('.')[0]
                        label_list = [0] * len(labels_dict)
                        label_list[labels_dict[labelName]] = 1
                        all_labels.append(label_list)
                    except Exception as e:
                        print ("Error occurred" + str(e))

            if file.endswith('pp'):
                sys.stdout.write("\r%d%%" % int(i / 1000 * 100))
                sys.stdout.flush()
                i += 1

    # write lables and data to file using pickle
    with open("data", 'w') as f:
        f.write(pickle.dumps(data_dict.values()))

    with open("labels", 'w') as f:
        f.write(pickle.dumps(array(all_labels)))
