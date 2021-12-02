
import cv2
import cv2 as cv
import numpy as np
from findDynamicObjects import prepare_image_list
from Detectron_Segmentation import Detectron
from probabilityModel import probabilityMap

color = np.random.randint(0,255,(100,3))

def apply_homography(points, homography):
	hom_cord = np.append(points,1)
	transformed = np.matmul(homography,hom_cord)
	hetero_transformed = [transformed[0]/transformed[2], transformed[1]/transformed[2]]
	return hetero_transformed

def compute_homography(correspondences):
	A =[]
	for i,pair in enumerate(correspondences):
		row1 =  [0,0,0,-pair[0][0],-pair[0][1],-1, pair[0][0]*pair[1][1],pair[0][1]*pair[1][1],pair[1][1]]
		row2 =  [pair[0][0],pair[0][1], 1,0,0,0, -pair[0][0]*pair[1][0],-pair[0][1]*pair[1][0],-pair[1][0]]
		A.append(row1)
		A.append(row2)
	A = np.array(A)
	symmetric_A = np.matmul(np.transpose(A),A)
	u,sigma, vt = np.linalg.svd(symmetric_A)
	v = np.transpose(vt)
	last_col_v = v[:, -1]
	last_col_V = last_col_v/last_col_v[8]
	return np.reshape(last_col_V,(3,3))

def getSegment_segment_and_count(image):
	detectron_seg = Detectron()
	panoptic_seg_details = detectron_seg.onImage(image)
	## count the dynamic points on the segmentation
	panoptic_seg = panoptic_seg_details["panoptic_seg"][0].numpy()
	segmentation_info = panoptic_seg_details["panoptic_seg"][1]
	classes = np.unique(panoptic_seg)
	return len(classes), panoptic_seg, segmentation_info
def WarpPerspective(im_src, im_dst,depth_frame):
	# Calculate Homography
	#compute_inliers(homography, correspondences, threshold)
	#inlinersFromClusters(zip(pts_src,pts_dst),im_src, im_dst,depth_frame, 2,number_segments,25)
	orb = cv2.ORB_create(1000)
	kp1, des1 = orb.detectAndCompute(im_src,None)
	kp2, des2 = orb.detectAndCompute(im_dst, None)
	index_params = dict(algorithm=6,
						table_number=6,
						key_size=12,
						multi_probe_level=2)
	search_params = {}
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)
	MIN_MATCHES = 50
	# As per Lowe's ratio test to filter good matches
	good_matches = []
	for m, n in matches:
		if m.distance < 0.75 * n.distance:
			good_matches.append(m)

	if len(good_matches) > MIN_MATCHES:
		src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
		m, mask = cv2.findHomography(src_points, dst_points, cv2.LMEDS, 2.0)
		corrected_img = cv2.warpPerspective(im_src, m, (im_dst.shape[1], im_dst.shape[0]))
		#corrected_depth = cv2.warpPerspective(depth_frame, m, (im_dst.shape[1], im_dst.shape[0]))

	foreground_image =cv2.subtract( im_dst,corrected_img)
	foreground_image=cv2.warpPerspective(foreground_image, np.linalg.inv(m), (im_dst.shape[1], im_dst.shape[0]))

	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(foreground_image, cv2.MORPH_OPEN, kernel)
	opening = np.array(opening) > 64

	opening = opening.astype('uint8')*255
	#cv.imshow("difference Image", opening)
	return opening,m

def merge_images(segmentation, potential_values, homography):
	potential_locations_row, potential_locations_col = np.where(potential_values == [255])
	merge_container=np.zeros((segmentation.shape[0]+2,segmentation.shape[1]+2), np.uint8)
	for i in range(len(potential_locations_col)):
		k , l =int(potential_locations_row[i]), int(potential_locations_col[i])
		k , l = apply_homography((k,l), homography)
		k , l = int(k), int(l)
		floodflags = 4
		floodflags |= cv2.FLOODFILL_MASK_ONLY
		floodflags |= (255 << 8)
		if k in range(0,640) and l in range(0,480):
			num,im,mask,rect = cv2.floodFill(segmentation, merge_container, (l,k), (255,0,0), (10,)*3, (10,)*3, floodflags)
		else:
			break
		merge_container = cv2.add(mask, merge_container)
	return merge_container
def RunWarp():
	image_list = prepare_image_list("images/rgbd_dataset_freiburg3_walking_rpy/rgb")
	#depth_image_list = prepare_image_list("images/rgbd_dataset_freiburg3_walking_rpy/depth")
	frame1 = cv.imread(image_list[0])
	old_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
	PreviousProbabilityMap = np.full(old_gray.shape, 0.5)
	for index,image_file in enumerate(image_list[1:]):

		frame = cv.imread(image_file)
		frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		no_segments, panoptic_segmented,segmentation_info = getSegment_segment_and_count(frame)
		## scale the segmentations properly
		max_index = np.max(panoptic_segmented)
		panoptic_segmented = np.array(panoptic_segmented)*(255/max_index)
		panoptic_segmented = panoptic_segmented.astype(np.uint8)
		potential_moving, homography=WarpPerspective(old_gray, frame_gray, panoptic_segmented)
		dynamic_siloutte = merge_images(panoptic_segmented, potential_moving, homography)
		dynamic_siloutte = dynamic_siloutte[1:-1, 1:-1]
		updatedProbabilityMap = probabilityMap(dynamic_siloutte, PreviousProbabilityMap)
		refined_image = (updatedProbabilityMap > 0.5).astype(np.uint8)*255
		PreviousProbabilityMap = updatedProbabilityMap
		#print("probab map update", PreviousProbabilityMap)
		#cv2.imshow("dynamic silhoutte ", dynamic_siloutte)
		#cv2.imshow("segmentation ",np.array(panoptic_segmented))
		cv2.imshow("After probability Update ", refined_image)
		image_inv = 255 - refined_image

		static_orb = cv2.ORB_create(1000)
		#keypoints_image1, descriptors1 = static_orb.detectAndCompute(old_gray,image_inv)
		keypoints_image2, descriptors2 = static_orb.detectAndCompute(frame_gray, image_inv)
		old_gray = frame_gray.copy()
		static_orb_img=cv2.drawKeypoints(frame_gray, keypoints_image2, frame_gray)
		cv2.imshow(" static orb ", static_orb_img)
		k = cv.waitKey(30) & 0xff
		if k == 27:
			break
if __name__=="__main__":
	RunWarp()