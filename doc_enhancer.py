import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as Im 
from PIL import ImageShow, ImageOps, ImageEnhance
import argparse

def get_arguments():  #argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputs_dir', type=str, default='./inputs', help='Input Directory Path')
	parser.add_argument('--outputs_dir', type=str, default='./outputs', help='Output Directory Path')
	parser.add_argument("--contrast_val", type= int, default = 3, help='Contrast Value (def=3)')
	args = parser.parse_args()
	return args
	


def unshadow_1(img):

  rgb_planes = cv2.split(img)

  result_planes = []
  result_norm_planes = []
  for plane in rgb_planes:
      dilated_img = cv2.dilate(plane, np.ones((1,1), np.uint8))
      bg_img = cv2.medianBlur(dilated_img, 81)
      #bg_img = dilated_img
      diff_img = 255 - cv2.absdiff(plane, bg_img)
      norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      result_planes.append(diff_img)
      result_norm_planes.append(norm_img)

  result = cv2.merge(result_planes)
  result_norm = cv2.merge(result_norm_planes)
  return result, result_norm
  
 

def run_enhancer():

	args = get_arguments()
	input_dir = args.inputs_dir
	output_dir = args.outputs_dir
	contrast = args.contrast_val
	if os.path.isdir(input_dir) == False:
		print("INPUT DIRECTORY DOES NOT EXIST. CLOSING PROGRAM")
		exit()
	if os.path.isdir(output_dir) == False:
		os.mkdir(output_dir)

	files = os.listdir(input_dir)

	for name in files:
		img_p = input_dir + '/' + name
		img = cv2.imread(img_p, cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		_,unshadow_img = unshadow_1(img)


		sharpen = cv2.GaussianBlur(unshadow_img, (0,0), 3)
		sharpen = cv2.addWeighted(unshadow_img, 1.5, sharpen, -0.5, 0)


		im = Im.fromarray(sharpen)

		#image contrast 'enhancer'
		enhancer = ImageEnhance.Contrast(im)


		factor = contrast #increase contrast default 3x
		im_output = enhancer.enhance(factor)

		save_p = output_dir + '/' + 'enhanced_' + name
		im_output.save(save_p)


def main():
    run_enhancer()

if __name__ == '__main__':
    main()