import numpy as np
import cv2
from findDynamicObjects import prepare_image_list

def probability_model(new_good_points,):
	pass

image_list = prepare_image_list()
feature_params = dict( maxCorners = 100,
					   qualityLevel = 0.3,
					   minDistance = 7,
					   blockSize = 7 )
lk_params = dict( winSize  = (15,15),
				 maxLevel = 2,
				 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0,255,(100,3))

old_frame = cv2.imread(image_list[0])
old_frame = cv2.resize(old_frame, (320,240))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
mask = np.zeros_like(old_frame)

for index,image_file in enumerate(image_list[1:]):
	frame = cv2.imread(image_file)


	frame= cv2.resize(frame, (320,240))
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	if p1 is not None:
		good_new = p1[st==1]
		good_old = p0[st==1]
	# draw the tracks
	print(len(good_new), len(good_old))
	for i,(new,old) in enumerate(zip(good_new, good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
		frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
	img = cv2.add(frame,mask)
	cv2.imshow('frame',img)
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break

	# Now update the previous frame and previous points
	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1,1,2)
	if len(good_new) < 25:
		p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
		mask = np.zeros_like(old_frame)