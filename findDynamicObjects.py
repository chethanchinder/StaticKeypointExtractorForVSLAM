from Detectron_Segmentation import Detectron
from getkeypoints import KeypointsAndDescriptors
from utils import drawlines,dist_between_line_point,findFundamentalMatrix, findEpilines,get_static_dynamic_points
import numpy as np
import cv2

import os
def prepare_image_list(images_path = "images/rgbd_dataset_freiburg3_walking_rpy/rgb"):
	images_list = []
	for dir_path, _, filenames in os.walk(images_path):
		for filename in filenames:
			images_list.append(os.path.join(dir_path+"/"+filename))
	images_list = sorted(images_list)
	return images_list

class DynamicObjects(object):
	def __init__(self):
		super(DynamicObjects, self).__init__()
		self.detectron_seg = Detectron()
		self.keypoint = KeypointsAndDescriptors()

	def getOnlyStaticPointsFromSegmentation(self,image, static_points_homography, dynamic_points_homography,best_keypoints_image1,best_keypoints_image2,static_points_indices,dynamic_points_indices):
		pass
if __name__=="__main__":
	images_list = prepare_image_list()
	dynamicObject = DynamicObjects()
	curr_image = cv2.imread(images_list[0])
	curr_image_resized = cv2.resize(curr_image, (320,240))
	curr_image_resized_gray = cv2.cvtColor(curr_image_resized, cv2.COLOR_BGR2GRAY)
	for index,image_file in enumerate(images_list[5::3]):
		next_image = cv2.imread(image_file)
		next_image_resized = cv2.resize(next_image, (320,240))
		next_image_resized_gray = cv2.cvtColor(next_image_resized, cv2.COLOR_BGR2GRAY)

		best_matches,best_keypoints_image1,best_keypoints_image2 = dynamicObject.keypoint.getKeypoints(curr_image_resized_gray, next_image_resized_gray)
		list_matchedKP_loc1=np.float32(best_matches[:,0])
		list_matchedKP_loc2=np.float32(best_matches[:,1])
		Fundamental_matrix, mask = findFundamentalMatrix(list_matchedKP_loc1, list_matchedKP_loc2, cv2.FM_RANSAC)

		pts1 = list_matchedKP_loc1[mask.ravel()==1]
		pts2 = list_matchedKP_loc2[mask.ravel()==1]

		lines2 = findEpilines(pts1,1,Fundamental_matrix)
		lines1 = findEpilines(pts2,2,Fundamental_matrix)
		static_points, static_points_indices,dynamic_points,dynamic_points_indices =get_static_dynamic_points(pts2, lines2)

		next_image_resized_copy1 = next_image_resized.copy()
		new_static_keypoints = []
		new_dynamic_keypoints = []
		new_static_keypoints_prev_image = []
		new_dynamic_keypoints_prev_image = []
		print(" len(static_points)",len(static_points))
		print(" len(dynamic_points)",len(dynamic_points))
		for s in static_points_indices:
			new_static_keypoints.append(best_keypoints_image2[s])
			new_static_keypoints_prev_image.append(best_keypoints_image1[s])

		for d in dynamic_points_indices:
			new_dynamic_keypoints.append(best_keypoints_image2[d])
			new_dynamic_keypoints_prev_image.append(best_keypoints_image1[d])
		print("lwn of new new_static_keypoints ",len(new_static_keypoints))
		print(" len(new_dynamic_keypoints)",len(new_dynamic_keypoints))
		curr_image_resized_copy1 = curr_image_resized.copy()
		curr_image_resized_copy2 = curr_image_resized.copy()
		next_image_resized_copy1 = next_image_resized.copy()
		next_image_resized_copy2 = next_image_resized.copy()
		cv2.drawKeypoints(next_image_resized,new_static_keypoints, next_image_resized_copy1)
		cv2.drawKeypoints(next_image_resized,new_dynamic_keypoints, next_image_resized_copy2)
		cv2.drawKeypoints(curr_image_resized,new_static_keypoints_prev_image, curr_image_resized_copy1)
		cv2.drawKeypoints(curr_image_resized,new_dynamic_keypoints_prev_image, curr_image_resized_copy2)

		final_result = np.concatenate((next_image_resized_copy1, next_image_resized_copy2, curr_image_resized_copy1,curr_image_resized_copy2), axis = 1)
		cv2.imshow(' premature result', final_result)

		img1, img2 = drawlines(curr_image_resized, next_image_resized, lines1, lines2, list_matchedKP_loc1,list_matchedKP_loc2)
		final_result = np.concatenate((img1, img2), axis = 1)
		cv2.imshow("image epilines",final_result)

		final_image =np.concatenate([np.array(curr_image_resized),np.array(next_image_resized)],axis=1)
		cv2.imshow("concat",final_image)
		curr_image_resized_gray = next_image_resized_gray
		curr_image_resized = next_image_resized
		if cv2.waitKey(100) & 0xFF == ord('q'):
			break
