from Detectron_Segmentation import Detectron
from getkeypoints import KeypointsAndDescriptors
from applyHomography import drawlines,dist_between_line_point,findFundamentalMatrix, findEpilines,get_static_dynamic_points
import numpy as np
import cv2
import torch
import os
def prepare_image_list():
	images_list = []
	for dir_path, _, filenames in os.walk("/home/ranjan/ChethanChinder/dynamicSLAM/images/rgbd_dataset_freiburg3_walking_rpy/rgb"):
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
		panoptic_seg = self.detectron_seg.onImage(image)
		## count the dynamic points on the segmentation
		panoptic_seg = panoptic_seg.numpy()
		classes = np.unique(panoptic_seg)
		count_dict_dynamic = dict.fromkeys(classes, 0)
		dict_dynamic_indices = dict.fromkeys(classes,[])
		for dynamic_point, dynamic_points_indice in zip(dynamic_points_homography, dynamic_points_indices):
			val = panoptic_seg[int(dynamic_point[1]), int(dynamic_point[0])]
			count_dict_dynamic[val] +=1
			dict_dynamic_indices[val].append(dynamic_points_indice)
		count_dict_static = dict.fromkeys(classes,0)

		dict_static_indices = dict.fromkeys(classes,[])
		for static_point, static_points_indice in zip(static_points_homography, static_points_indices):
			val = panoptic_seg[int(static_point[1]), int(static_point[0])]
			count_dict_static[val] +=1
			dict_static_indices[val].append(static_points_indice)
		#print("dynamic points ",dict_dynamic_indices)
		#print("static points ",dict_static_indices)

		for key in count_dict_dynamic.keys():
			if 1 < count_dict_dynamic[key]:
				dict_dynamic_indices[key] = []
				dict_static_indices[key]=[]
		new_static_indices =[]
		for key in dict_dynamic_indices:
			new_static_indices+=dict_static_indices[key]
		new_best_matches_image1 =[]
		new_best_matches_image2 =[]

		#print("new static indices: ", new_static_indices)
		for id, match in enumerate(zip(best_keypoints_image1,best_keypoints_image2)):
			for static_idx in new_static_indices:
				if id == static_idx:
					new_best_matches_image1.append(match[0])
					new_best_matches_image2.append(match[1])
		#print("new best matches: ",new_best_matches_image1)
		return np.array(new_best_matches_image1),np.array(new_best_matches_image2), new_static_indices
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

		##  find static points from dynami ones using thompson error
		static_points, static_points_indices,dynamic_points,dynamic_points_indices =get_static_dynamic_points(pts2, lines2)
		matches_after_seg_image1,matches_after_seg_image2, static_points_loc_after_seg=dynamicObject.getOnlyStaticPointsFromSegmentation(next_image_resized, static_points,dynamic_points,best_keypoints_image1,best_keypoints_image2,static_points_indices,dynamic_points_indices)

		next_image_resized_copy1 = next_image_resized.copy()
		# next_image_resized_copy2 = next_image_resized.copy()
		cv2.drawKeypoints(next_image_resized,matches_after_seg_image2, next_image_resized_copy1)
		# cv2.drawKeypoints(next_image_resized,new_dynamic_keypoints, next_image_resized_copy2)
		# cv2.drawKeypoints(curr_image_resized,new_static_keypoints_prev_image, curr_image_resized_copy1)
		# cv2.drawKeypoints(curr_image_resized,new_dynamic_keypoints_prev_image, curr_image_resized_copy2)
		#
		# final_result = np.concatenate((next_image_resized_copy1, next_image_resized_copy2, curr_image_resized_copy1,curr_image_resized_copy2), axis = 1)
		cv2.imshow(' premature result', next_image_resized_copy1)




		# new_static_keypoints = []
		# new_dynamic_keypoints = []
		# new_static_keypoints_prev_image = []
		# new_dynamic_keypoints_prev_image = []
		# print(" len(static_points)",len(static_points))
		# print(" len(dynamic_points)",len(dynamic_points))
		# for s in static_points_indices:
		# 	new_static_keypoints.append(best_keypoints_image2[s])
		# 	new_static_keypoints_prev_image.append(best_keypoints_image1[s])
		#
		# for d in dynamic_points_indices:
		# 	new_dynamic_keypoints.append(best_keypoints_image2[d])
		# 	new_dynamic_keypoints_prev_image.append(best_keypoints_image1[d])
		# print("lwn of new new_static_keypoints ",len(new_static_keypoints))
		# print(" len(new_dynamic_keypoints)",len(new_dynamic_keypoints))
		# curr_image_resized_copy1 = curr_image_resized.copy()
		# curr_image_resized_copy2 = curr_image_resized.copy()
		# next_image_resized_copy1 = next_image_resized.copy()
		# next_image_resized_copy2 = next_image_resized.copy()
		# cv2.drawKeypoints(next_image_resized,new_static_keypoints, next_image_resized_copy1)
		# cv2.drawKeypoints(next_image_resized,new_dynamic_keypoints, next_image_resized_copy2)
		# cv2.drawKeypoints(curr_image_resized,new_static_keypoints_prev_image, curr_image_resized_copy1)
		# cv2.drawKeypoints(curr_image_resized,new_dynamic_keypoints_prev_image, curr_image_resized_copy2)
		#
		# final_result = np.concatenate((next_image_resized_copy1, next_image_resized_copy2, curr_image_resized_copy1,curr_image_resized_copy2), axis = 1)
		# cv2.imshow(' premature result', final_result)

		#img1, img2 = drawlines(curr_image_resized, next_image_resized, lines1, lines2, list_matchedKP_loc1,list_matchedKP_loc2)
		#final_result = np.concatenate((img1, img2), axis = 1)
		#cv2.imshow("image epilines",final_result)

		#final_image =np.concatenate([np.array(curr_image_resized),np.array(next_image_resized)],axis=1)
		#cv2.imshow("concat",final_image)
		curr_image_resized_gray = next_image_resized_gray
		curr_image_resized = next_image_resized
		if cv2.waitKey(100) & 0xFF == ord('q'):
			break
