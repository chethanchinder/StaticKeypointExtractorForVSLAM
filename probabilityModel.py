import numpy as np
def probabilityMap(silhoutte, PreviousProbabilityMap):
	#probability_container = np.zeros((silhoutte.shape[0],silhoutte.shape[1]), np.uint8)
	alpha = 0.25
	silhoutte = silhoutte/255.0
	updatedProbability=np.multiply(1-alpha, PreviousProbabilityMap) + np.multiply(alpha, silhoutte)
	return updatedProbability