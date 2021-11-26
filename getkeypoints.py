import cv2
import numpy as np

class KeypointsAndDescriptors(object):
	def __init__(self):
		super(KeypointsAndDescriptors,self).__init__()

	def drawCorrespondences(self, image1,image2, list_matchedKP1, list_matchedKP2, best_matches):
		final_image = cv2.drawMatches(image1, list_matchedKP1,image2, list_matchedKP2,best_matches,None)
		cv2.imshow("image keypoint matches",final_image)


	def getKeypoints(self,image_gray1, image_gray2):
		orb = cv2.ORB_create(1000)
		keypoints_image1, descriptors1 = orb.detectAndCompute(image_gray1,None)
		keypoints_image2, descriptors2 = orb.detectAndCompute(image_gray2, None)
		best_matches,best_keypoints_image1, best_keypoints_image2 = self.find_correspondences(keypoints_image1, descriptors1, keypoints_image2, descriptors2,0.8)
		# matcher = cv2.BFMatcher( )
		# matches = matcher.match(descriptors1, descriptors2)
		# matches = sorted(matches, key= lambda x:x.distance)
		# list_matchedKP_loc =[]
		# list_matchedKP1 = []
		# list_matchedKP2 = []
		# list_matched_desc1 = []
		# list_matched_desc2 = []
		# for match in matches[:50]:
		# 	image1_idx = match.queryIdx
		# 	image2_idx = match.trainIdx
		# 	keypoint1 = keypoints_image1[image1_idx]
		# 	keypoint2 = keypoints_image2[image2_idx]
		# 	list_matchedKP1.append(keypoint1)
		# 	list_matchedKP2.append(keypoint2)
		# 	list_matched_desc1.append(descriptors1[image1_idx].tolist())
		# 	list_matched_desc2.append(descriptors2[image2_idx].tolist())
		# 	list_matchedKP_loc.append([keypoint1.pt, keypoint2.pt])
		#best_matches = matcher.match(np.float32(list_matched_desc1),np.float32(list_matched_desc2))
		#return np.array(list_matchedKP_loc),list_matchedKP1, list_matchedKP2, best_matches
		return best_matches,best_keypoints_image1, best_keypoints_image2
	def min_eucledian_dist(self,des2_list,des):
		first_min , second_min = np.infty , np.infty
		min1_ind, min2_ind = 0 ,0
		first_min_desc, second_min_desc = None, None
		for i, des2 in enumerate(des2_list):
			dist1=np.sqrt(((des - des2)**2).sum())
			if dist1 <= first_min:
				second_min = first_min
				first_min = dist1
				min2_ind = min1_ind
				min1_ind = i
				second_min_desc = first_min_desc
				first_min_desc = des2_list[i]
			elif dist1 < second_min and dist1>first_min:
				second_min = dist1
				second_min_desc = des2_list[i]
				min2_ind = i
		return first_min,first_min_desc,min1_ind, second_min, second_min_desc,min2_ind
	def find_correspondences(self,kp1, des1, kp2, des2, ratio):
		correspondences = []
		keypoints_image1 = []
		keypoints_image2 = []
		for i,kp_i in enumerate(kp1):
			min1,desc1,min1_ind,min2, desc2,min2_ind = self.min_eucledian_dist(des2,des1[i,:])
			des_index=np.argsort(np.sqrt(((des2-des1[i,:])**2).sum(axis=1)))
			if min1 < ratio*min2:
				x1, y1 = kp_i.pt[0], kp_i.pt[1]
				x2, y2 = kp2[min1_ind].pt[0],kp2[min1_ind].pt[1]
				correspondences.append([np.asarray([x1,y1]),np.asarray([x2,y2])])
				keypoints_image1.append(kp_i)
				keypoints_image2.append(kp2[min1_ind])
		return np.array(correspondences), keypoints_image1, keypoints_image2
if __name__=="__main__":
	videopath = "images/test_countryroad.mp4"
	keypoint = KeypointsAndDescriptors()
	cap = cv2.VideoCapture(videopath)
	while(cap.isOpened()):
		ret1,image1 = cap.read()
		ret2,image2= cap.read()
		if ret1 and ret2:
			image1 = cv2.resize(image1, (320,240))
			image2 = cv2.resize(image2, (320,240))
			image_gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
			image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
			best_matches = keypoint.getKeypoints(image_gray1, image_gray2)
			#keypoint.drawCorrespondences(image1,image2, list_matchedKP1, list_matchedKP2, best_matches)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			print('ret is 0, No image')

	cap.release()
	cv2.destroyWindow()
