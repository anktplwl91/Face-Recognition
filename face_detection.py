'''
This file is used to extract faces from images, and store them in seperate directory, so that they can be used by model to be trained.
'''

import os
import cv2
import argparse
import glob


cv2_data_dir = "C:\\Users\\apaliwal\\AppData\\Local\\Continuum\\Anaconda3\\Lib\\site-packages\\cv2\\data\\"
face_cascade = cv2.CascadeClassifier(cv2_data_dir+'haarcascade_frontalface_default.xml')


def _get_faces_eyes(image_file):
    
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    
    return roi_color
		

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='File to extract faces from images of human subjects and store them in different directory')	
	parser.add_argument('--new_image_dir', '-n', help="Name of new directory to be created for storing faces.")
	parser.add_argument('--orig_image_dir', '-o', nargs='+', help="Name of directory in which images are stored from which faces are to be detected")

	args = parser.parse_args()

	new_dir = args.new_image_dir
	orig_dir_list = args.orig_image_dir

	os.mkdir(new_dir)
	
	idx = 1

	for orig_dir in orig_dir_list:

		sub_d = next(os.walk(orig_dir))[1]
		
		for sd in sub_d:
		
			images = glob.glob(orig_dir+"\\"+sd+"\\*.jpg")
			os.mkdir(new_dir+"\\people_"+str(idx))

			for i, img in enumerate(images):

				try:
					face = _get_faces_eyes(img)
					cv2.imwrite(new_dir+"\\people_"+str(idx)+"\\face_"+str(i)+".jpg", face)
				except:
					continue
			
			idx += 1