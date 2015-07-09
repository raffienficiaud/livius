from multiprocessing import Pool
import glob
import cv2


def extract_info(file_name):
    image = cv2.imread(file_name)
    return image.shape



if __name__ == '__main__':
    directory = '/home/livius/Code/livius/SourceCode/Example Data/thumbnails/'

    files = glob.glob(directory + '*.png')

    pool = Pool(processes=8)


    result = pool.map(extract_info, files)

    print result

