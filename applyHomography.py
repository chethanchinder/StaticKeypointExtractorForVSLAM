import cv2
import numpy as np
from getkeypoints import KeypointsAndDescriptors
def drawlines(img1, img2, lines1, lines2, pts1, pts2):
	## we will draw epipolar lines on image 1
	r1, c1,chnl = img1.shape
	r2, c2, chnl = img2.shape
	for r1, r2, pts1, pts2 in zip(lines1, lines2,pts1,pts2):
		color = tuple(map(int,np.random.randint(0,255,3)))
		x0, y0 = map(int, [0, -r1[2]/r1[1]])
		x1, y1 = map(int, [c1, -(r1[2]+r1[0]*c1)/r1[1]])
		cv2.line(img1,(x0,y0), (x1,y1), color, 1)

		x0, y0 = map(int, [0, -r2[2]/r2[1]])
		x1, y1 = map(int, [c2, -(r2[2]+r2[0]*c2)/r2[1]])
		cv2.line(img2,(x0,y0), (x1,y1), color, 1)

		cv2.circle(img1, tuple(pts1),2,color,-1)
		cv2.circle(img2, tuple(pts2),2,color,-1)
	return img1, img2


def findEpilines(pts,image,F ):
	lines = cv2.computeCorrespondEpilines(pts.reshape(-1,1,2),image, F)
	return lines.reshape(-1,3)
def applyHomography(KP_image1, KP_image2, image):
	_, Mask=cv2.findHomography(KP_image1, KP_image2)

	pass

def dist_between_line_point(line_2, points_2, threshold):
	dist=np.abs( line_2[0]*points_2[0] + line_2[1]*points_2[1] + line_2[2])/np.sqrt(line_2[0]**2 + line_2[1]**2)
	return dist < threshold

def findFundamentalMatrix(pts1, pts2, algorithmType=cv2.FM_8POINT):
	F, mask = cv2.findFundamentalMat(pts1,pts2,algorithmType)
	return F, mask
def get_static_dynamic_points(points2, lines2):
	static_points = []
	static_points_indices =[]
	dynamic_points = []
	dynamic_points_indices =[]
	for index,ptline2 in enumerate(zip(points2, lines2)):
		if dist_between_line_point(ptline2[1],ptline2[0],2):
			static_points.append(ptline2[0])
			static_points_indices.append(index)
		else:
			dynamic_points.append(ptline2[0])
			dynamic_points_indices.append(index)

	print(" total static points:", len(static_points), " total dynamic points :", len(dynamic_points))

	return static_points, static_points_indices, dynamic_points,dynamic_points_indices



if __name__=="__main__":
	keypoint = KeypointsAndDescriptors()
	im1 = cv2.imread("images/uttower_left.jpeg")
	im2 = cv2.imread("images/uttower_right.jpeg")
	im1 = cv2.resize(im1, (320,240))
	im2 = cv2.resize(im2, (320,240))
	image_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	image_gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	#list_matchedKP_loc,list_matchedKP1, list_matchedKP2, best_matches = keypoint.getKeypoints(image_gray1, image_gray2)
	best_matches = keypoint.getKeypoints(image_gray1, image_gray2)
	list_matchedKP_loc1=np.float32(best_matches[:,0])
	list_matchedKP_loc2=np.float32(best_matches[:,1])
	Fundamental_matrix, mask = findFundamentalMatrix(list_matchedKP_loc1, list_matchedKP_loc2, cv2.FM_RANSAC)


	pts1 = list_matchedKP_loc1[mask.ravel()==1]
	pts2 = list_matchedKP_loc2[mask.ravel()==1]

	lines2 = findEpilines(pts1,1,Fundamental_matrix)
	lines1 = findEpilines(pts2,2,Fundamental_matrix)

	static_points, static_points_indices =get_static_dynamic_points(pts2, lines2)

	img1, img2 = drawlines(im1, im2, lines1, lines2, list_matchedKP_loc1,list_matchedKP_loc2)
	final_result = np.concatenate((img1, img2), axis = 1)
	cv2.imshow("image epilines",final_result)

	if cv2.waitKey(0) | 0xFF == ord('q'):
		cv2.destroyWindow()
