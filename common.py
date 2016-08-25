import cPickle as pickle
import time
import numpy as np
from scipy.io import loadmat, savemat
from PIL import Image
from sklearn.externals import joblib


def get_time_str():
    return time.strftime("%Y-%m-%d, %H:%M:%S ", time.localtime((time.time())))


def print_info(msg):
    print get_time_str(), msg


# saving data into pkl
def data_to_pkl(data, file_path):
    print "Saving data to file(%s). " % (file_path)

    with open(file_path, "w") as f:
        pickle.dump(data, f)
        return True

    print "Occur Error while saving..."
    return False


def read_pkl(file_path):
    with open(file_path, "r") as f:
        return pickle.load(f)


def save_draw_file(draw_list):
    st = ""
    for row in draw_list:
        threshold = row[1]
        st += str(threshold) + " "
        for r in row[0]:
            for c in r:
                st += str(c) + " "
        st += "\n"
    with open("draw_file.txt", "w") as f:
        f.write(st)


def report_format(report):
    report = report.split()
    result = np.zeros((3, 3))
    result[0][0] = report[5]
    result[0][1] = report[6]
    result[0][2] = report[7]

    result[1][0] = report[10]
    result[1][1] = report[11]
    result[1][2] = report[12]

    result[2][0] = report[17]
    result[2][1] = report[18]
    result[2][2] = report[19]

    return result


def trans_img_to_grey(rgb_img):
    """
    transform a img from RGB to grey
    :param rgb_img: origin image
    :return: grey image
    """
    if rgb_img.mode == 'L':
        return rgb_img
    else:
        return rgb_img.convert('L')


def load_mat_directly(filename):
    """
    load .mat file, key of this dict need to be same with filename
    :param filename: for example:../data/googlenet_lfw.mat
    """
    return loadmat(filename)[(filename.split('/')[-1]).split('.')[0]]


def save_pac_mat(data, filename):
    """
    save data after pca to .mat,for example,origin file is test.mat, save pca file to test_pca.mat and the key is test_pca
    :param filename: for example:../data/googlenet_lfw.mat
    """
    pca_file = get_pca_filename(filename)
    savemat(pca_file, {(pca_file.split('/')[-1]).split('.')[0]: data})


def get_pca_filename(filename):
    """
    for example,origin file is test.mat, pca filename is test_pca.mat
    """
    filename = filename.split('.mat')[0]
    filetype = '.mat'
    return ''.join((filename, '_pca', filetype))


def substract_mean(data):
    """
    substract it's mean for every row in data, to make it's mean equal to 0
    """
    mean = np.mean(data, axis=0)
    return data - mean


def name_to_path(name):
    """
    get image path though people name
    """
    return ''.join(('/home/liuxuebo/CV/BLUFR/nosym(90x90)/', '_'.join(name.split('_')[:-1]), '/', name, '.jpg'))


def pca_transform(model_path, data):
    """
    transform data using pca model
    """
    clt_pca = joblib.load(model_path)
    data = clt_pca.transform(data)
    return substract_mean(data)


if __name__ == "__main__":
    pass
