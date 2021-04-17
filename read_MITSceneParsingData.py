__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


def read_dataset(data_dir):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    # if not os.path.exists(pickle_filepath):
    #     utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
    # SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
    result = create_image_lists(os.path.join(data_dir, "data"))
    print ("Pickling ...")
    with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        train_records = result['train']
        valid_records = result['valid']
        test_records = result['test']
        del result

    return train_records, valid_records, test_records


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['train','valid', 'test']
    image_list = {}

    for directory in directories:
        file_list = []
        image_list[directory] = []
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'png')
        file_list.extend(glob.glob(file_glob))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file_seg = os.path.join(image_dir, "mask_seg", directory, filename + '.png')
                annotation_file_binary = os.path.join(image_dir, "mask_binary", directory, filename + '.png')

                if os.path.exists(annotation_file_binary and annotation_file_seg):
                    record = {'image': f, 'annotation_seg': annotation_file_seg, 'annotation_binary': annotation_file_binary,'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list
