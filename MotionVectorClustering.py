import cv2
import cv2 as cv
import numpy
import numpy as np
import pydegensac
from scipy.cluster.vq import kmeans2, whiten, vq
from findDynamicObjects import prepare_image_list
from opticalFlow import optical_flow_points
from utils import eucledian_distance
from Detectron_Segmentation import Detectron
from probabilityModel import probabilityMap


color = np.random.randint(0,255,(100,3))

def apply_homography(points, homography):
	hom_cord = np.append(points,1)
	transformed = np.matmul(homography,hom_cord)
	hetero_transformed = [transformed[0]/transformed[2], transformed[1]/transformed[2]]
	return hetero_transformed

def compute_inliers(homography, correspondences, threshold):
	inliners = []
	outliers =[]
	for correspondence in correspondences:
		transformed12_pt = apply_homography(correspondence[0],homography)
		dist=np.linalg.norm(np.array(correspondence[1]) - np.array(transformed12_pt))
		if dist < threshold:
			inliners.append(correspondence)
		else:
			outliers.append(correspondence)
	return inliners, outliers

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
def getInliersAndOutliers(homography,correspondences, threshold):
	inliners = []
	outliers =[]
	for correspondence in correspondences:
		transformed12_pt = apply_homography(correspondence[0],homography)

		dist=np.linalg.norm(np.array(correspondence[1]) - np.array(transformed12_pt))
		if dist < threshold:
			inliners.append(correspondence)
		else:
			outliers.append(correspondence)
	return inliners, outliers

def findMotionVectors(pts_src,pts_dst, depth_frame):
	motion_vector_list =[]
	#Subtraction of points to get motion vectors
	for src, dest in zip(pts_src,pts_dst):
		x, y = (src[0]-dest[0],src[1]-dest[1])
		motion_vector_list.append((x,y))

	# motion vectors scaled according to the depth for proper motion vectors
	# for src, dest in zip(pts_src,pts_dst):
	# 	pt_x , pt_y =int(dest[1]),int(dest[0])
	# 	if 0 < pt_x < 640 and  0< pt_y < 480:
	# 		if depth_frame[pt_x, pt_y ] == 0:
	# 			x,y = 	(255*(src[0]-dest[0]),255*(src[1]-dest[1]))
	# 		else:
	# 			depth_mag =depth_frame[pt_x, pt_y]
	# 			x, y = (depth_mag*(src[0]-dest[0]),depth_mag*(src[1]-dest[1]))
	# 		motion_vector_list.append((x,y))
	return np.array(motion_vector_list)
def get_correspondences_from_idx(cluster_indices,pts_src,pts_dst,random_cluster_indices):
	random_correspondences = []
	count = 0
	for index,idx in enumerate(cluster_indices):

		if idx in random_cluster_indices:
			count= count + 1
			random_correspondences.append( [pts_src[index],pts_dst[index]])
		if count ==8:
			break
	return random_correspondences

def inlinersFromClusters(pts_src,pts_dst,im_src, im_dst,depth_frame, threshold, number_of_clusters,iterations = 25):
	#correspondences=list(correspondences)
	motionVectors  = findMotionVectors(pts_src,pts_dst, depth_frame)
	motionVectors = whiten(motionVectors)
	centroids, _  = kmeans2(motionVectors, number_of_clusters)
	cluster_indices, _ = vq(motionVectors,centroids)
	unique_indices=list(set(cluster_indices))
	im_dst = cv2.cvtColor(im_dst, cv2.COLOR_GRAY2RGB)
	for id,cluster in enumerate(cluster_indices):
		a = pts_dst[id][0]
		b = pts_dst[id][1]
		frame2 = cv.circle( im_dst,(int(a),int(b)),5,color[cluster].tolist(),-1)
		cv2.imshow("clusters ",frame2)

	max_inliers = -np.infty
	for i in range(iterations):
		random_cluster_indices=np.random.choice(unique_indices, min(4,np.random.choice(range(1,len(unique_indices)))))
		random_correspondences = get_correspondences_from_idx(cluster_indices,pts_src,pts_dst,random_cluster_indices)
		H = compute_homography(random_correspondences)
		inliers, outliers =getInliersAndOutliers(H, random_correspondences,threshold)
		if len(inliers) > max_inliers:
			max_inliers= len(inliers)
			final_inliners = inliers
			final_homography = H
			final_outliers = outliers
	if max_inliers > -numpy.infty:
		return  final_inliners, final_outliers, final_homography
	else:
		print("error: could not find inliers ")
		return None
def getSegment_segment_and_count(image):
	detectron_seg = Detectron()
	panoptic_seg_details = detectron_seg.onImage(image)
	## count the dynamic points on the segmentation
	panoptic_seg = panoptic_seg_details["panoptic_seg"][0].numpy()
	segmentation_info = panoptic_seg_details["panoptic_seg"][1]
	classes = np.unique(panoptic_seg)
	return len(classes), panoptic_seg, segmentation_info
def WarpPerspective(im_src, im_dst,depth_frame, pts_src, pts_dst, number_segments = 8):
	# Calculate Homography
	#compute_inliers(homography, correspondences, threshold)
	final_inliners, final_outliers, final_homography =inlinersFromClusters(pts_src,pts_dst,im_src, im_dst,depth_frame, 2,number_segments,25)

	depth_frame = (np.array(depth_frame)//16)*16
	depth_frame = np.uint8(depth_frame)
	final_homography = compute_homography( zip([ item[0] for item in final_inliners],[ item[1] for item in final_inliners]))
	# Warp source image to destination based on homography
	im_out = cv2.warpPerspective(im_src, final_homography, (im_dst.shape[1],im_dst.shape[0]))
	# Display images
	foreground_image =cv2.subtract( im_dst,im_out)
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(foreground_image, cv2.MORPH_OPEN, kernel)
	opening = np.array(opening) > 25
	opening = opening.astype('uint8')*255
	im_out = cv2.warpPerspective(im_src, np.linalg.inv(final_homography), (im_dst.shape[1],im_dst.shape[0]))

	cv2.imshow(" potententially moving", opening)
	return opening, final_homography

def merge_images(segmentation, potential_values, homography):
	potential_locations_row, potential_locations_col = np.where(potential_values == [255])
	merge_container=np.zeros((segmentation.shape[0]+2,segmentation.shape[1]+2), np.uint8)
	for i in range(len(potential_locations_col)):
		k,l =int(potential_locations_row[i]), int(potential_locations_col[i])
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
	depth_image_list = prepare_image_list("images/rgbd_dataset_freiburg3_walking_rpy/depth")
	frame1 = cv.imread(image_list[0])
	feature_params = dict( maxCorners = 100,
						   qualityLevel = 0.3,
						   minDistance = 7,
						   blockSize = 7 )
	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (15,15),
					  maxLevel = 2,
					  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
	# Create some random colors
	color = np.random.randint(0,255,(100,3))
	# Take first frame and find corners in it
	old_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
	p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	# Create a mask image for drawing purposes
	mask = np.zeros_like(frame1)
	PreviousProbabilityMap = np.full(old_gray.shape, 0.5)
	for index,image_file in enumerate(image_list[1:]):

		depth_frame = cv2.imread(depth_image_list[index-1],0)
		kernel = np.ones((5,5),np.uint8)
		depth_frame = cv2.dilate(depth_frame,kernel,iterations = 2)
		depth_frame = (np.array(depth_frame)//16)*16
		depth_frame = np.uint8(depth_frame)
		frame = cv.imread(image_file)
		frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		good_new, good_old = optical_flow_points(old_gray, frame_gray,p0, lk_params)
		pts_src =[]
		pts_dst =[]
		for i,(new,old) in enumerate(zip(good_new, good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			pts_src.append([c,d])
			pts_dst.append([a,b])
		no_segments, panoptic_segmented,segmentation_info = getSegment_segment_and_count(frame)
		## scale the segmentations properly
		max_index = np.max(panoptic_segmented)
		panoptic_segmented = np.array(panoptic_segmented)*(255/max_index)
		panoptic_segmented = panoptic_segmented.astype(np.uint8)
		number_of_segments = min(5,no_segments)

		potential_moving, homography=WarpPerspective(old_gray, frame_gray, depth_frame, pts_src,pts_dst, number_of_segments)
		dynamic_siloutte = merge_images(panoptic_segmented, potential_moving, homography)
		dynamic_siloutte = dynamic_siloutte[1:-1, 1:-1]
		updatedProbabilityMap = probabilityMap(dynamic_siloutte, PreviousProbabilityMap)
		refined_image = (updatedProbabilityMap > 0.5).astype(np.uint8)*255
		PreviousProbabilityMap = updatedProbabilityMap
		cv2.imshow("dynamic silhoutte ", dynamic_siloutte)
		cv2.imshow("After probability Update ", refined_image)

		k = cv.waitKey(30) & 0xff
		if k == 27:
			break
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1,1,2)
		if len(good_new) < 25:
			p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
			mask = np.zeros_like(frame1)
if __name__=="__main__":
	RunWarp()