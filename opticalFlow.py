import numpy as np
import cv2 as cv
from findDynamicObjects import prepare_image_list

def optical_flow_points(old_gray, frame2_gray,p0, lk_params):
	p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame2_gray, p0, None, **lk_params)
	if p1 is not None:
		good_new = p1[st==1]
		good_old = p0[st==1]
	return good_new, good_old

def test_optical_flow():
	image_list = prepare_image_list("images/rgbd_dataset_freiburg3_walking_rpy/rgb")
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

	for index,image_file in enumerate(image_list[1:]):


		frame2 = cv.imread(image_file)
		frame2_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
		good_new, good_old =  optical_flow_points(old_gray, frame2_gray,p0, lk_params)
		# draw the tracks
		for i,(new,old) in enumerate(zip(good_new, good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
			frame2 = cv.circle(frame2,(int(a),int(b)),5,color[i].tolist(),-1)
		img = cv.add(frame2,mask)
		cv.imshow('frame',img)
		k = cv.waitKey(30) & 0xff
		if k == 27:
			break
		old_gray = frame2_gray.copy()
		p0 = good_new.reshape(-1,1,2)
		if len(good_new) < 25:
			p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
			mask = np.zeros_like(frame1)