import os
import cv2
from xml.etree import ElementTree
from tqdm import tqdm
import numpy as np
import pickle

class Distribution:

    def __init__(self,images_path,ocr_path,table_path):
        self.images_path=images_path
        self.ocr_path=ocr_path
        self.table_path=table_path
        self.all_words=[]
        self.all_numbers=[]
        self.pickle_filename='distribution_pickle'

    def load_from_pickle(self):
        file=open(self.pickle_filename,'rb')
        #print('\nloaded from pickle')
        self.all_words,self.all_numbers=pickle.load(file)


    def store_to_pickle(self):
        print('\n saved to pickle')
        file=open(self.pickle_filename,'wb')
        pickle.dump([self.all_words,self.all_numbers],file)
        file.close()

    def get_distribution(self):

        if(os.path.exists(self.pickle_filename)):
            self.load_from_pickle()
            return self.all_words,self.all_numbers

        for filename in tqdm(os.listdir(self.images_path)):
            im = cv2.imread(os.path.join(self.images_path, filename))

            root = ElementTree.parse(os.path.join(self.table_path, filename.replace('.png', '.xml'))).getroot()
            im, table_coords = self.table_rectangle(root, im)

            root = ElementTree.parse(os.path.join(self.ocr_path, filename.replace('.png', '.xml'))).getroot()
            im, words, numbers = self.words_rectangles(root, table_coords, im)

            self.all_numbers += numbers
            self.all_words += words

        self.store_to_pickle()
        return self.all_words,self.all_numbers

    def get_transformed_pts(self,pts1, pts2, dim, imshape):
        return (int((pts1[0] / imshape[1]) * dim[0]), int((pts1[1] / imshape[0]) * dim[1])), (
        int((pts2[0] / imshape[1]) * dim[0]), int((pts2[1] / imshape[0]) * dim[1]))

    def get_numpy_coords(self,root, height):
        # this function will return in format of x0,y0,x1,y1
        coords_text = np.array([[coords.attrib, coords.text.strip()] for coords in root.iter('word')])
        all_coords = np.array([coords_text[:, 0]]).transpose()
        all_text = np.array([coords_text[:, 1]]).transpose()
        all_coords = np.array([[int(coords[0]['left']), height - int(coords[0]['top']), int(coords[0]['right']),
                                height - int(coords[0]['bottom'])] for coords in all_coords])
        return all_coords, all_text

    def get_gt_within_table(self,table_coords, words_coords, all_text):

        final_words_coords = []
        final_text = []
        for i in range(len(table_coords)):
            table_coord = np.array([table_coords[i, :]])
            mask = np.concatenate(([np.all(words_coords[:, :2] >= table_coord[:, :2], axis=1)],
                                   [np.all(words_coords[:, 2:] <= table_coord[:, 2:], axis=1)]), axis=0).transpose()
            trues = np.array([[True, True]])
            mask = np.all(mask == trues, axis=1)
            final_words_coords.append(words_coords[mask])
            final_text.append(all_text[mask])

        return np.array(final_words_coords), np.array(final_text)

    def words_rectangles(self,root, table_coords, im):

        table_coords = np.array(table_coords)
        all_words = []
        all_numbers = []
        alphabets = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        numbers = '0123456789'
        height, width, _ = im.shape
        all_words_coords, all_text = self.get_numpy_coords(root, height)
        masked_words_coords, masked_text = self.get_gt_within_table(table_coords, all_words_coords, all_text)

        for i in range(len(masked_words_coords)):
            words_coords = masked_words_coords[i]

            text = masked_text[i]

            for coords, s_txt in zip(words_coords, text):
                coords = np.array(coords)
                if (s_txt[0][0][0] in alphabets):
                    all_words.append(s_txt[0])
                elif (s_txt[0][0][0] in numbers):
                    all_numbers.append(s_txt[0])
                x0, y0, x1, y1 = coords[0], coords[1], coords[2], coords[3]
                pts1, pts2 = (x0, y0), (x1, y1)
                cv2.rectangle(im, pts1, pts2, (0, 0, 255), 2)

        return im,all_words,all_numbers

    def table_rectangle(self,root, im):
        height, width, _ = im.shape
        table_coords = []
        for coords in root.iter('Table'):
            coords = coords.attrib
            x0, x1, y0, y1 = int(coords['x0']), int(coords['x1']), int(coords['y0']), int(coords['y1'])
            pts1, pts2 = (x0, y0), (x1, y1)
            table_coords.append([x0, y0, x1, y1])
            cv2.rectangle(im, pts1, pts2, (0, 0, 255), 2)
        return im, table_coords