
import cv2
import numpy as np

def keypoints_to_locations(Keypoints):
	pass
def locations_to_Keypoints(locations):
	keypoint = cv2.KeyPoint()
def eucledian_distance(des1,des2):
	dist=np.sqrt(((des1 - des2)**2).sum())
	return dist

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
