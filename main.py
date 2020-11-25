# Import numpy and OpenCV
import cv2
import numpy as np
import time, os

resize_factor = 0.5
BRIGHTNESS = 100
BLURINESS = 700
PATH = "./output/"

def movingAverage(curve, radius): 
	window_size = 2 * radius + 1
	# Apply convolution with Padding
	curve_pad = np.pad(curve, (radius, radius), 'edge') 
	curve_smoothed = np.convolve(curve_pad, np.ones(window_size)/window_size, mode='same') 
	curve_smoothed = curve_smoothed[radius:-radius]
	return curve_smoothed 

def smooth(trajectory): 
  smoothed_trajectory = np.copy(trajectory) 
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=50)
  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.08)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

start = time.time()
# Read input video
cap = cv2.VideoCapture('sample_car.mp4');
wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
hgt = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

# Set up output video
out = cv2.VideoWriter(PATH+'video_out.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, (2*wid, hgt))
fast = cv2.FastFeatureDetector_create(threshold = 25, nonmaxSuppression = True, type = 0)
clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8,8))

if not os.path.exists(PATH):
	os.makedirs(PATH)
	os.makedirs(PATH+'blur/')
	os.makedirs(PATH+'brightness/')

# Read first frame
_, img = cap.read() 
prev = cv2.resize(img, (0, 0), fx = resize_factor, fy = resize_factor) 
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 
transforms = np.zeros((n_frames-1, 3), np.float32) 
print("Time: Preprocess - ", time.time() - start)
start = time.time()

for i in range(n_frames-2):
	# Detect feature points in previous frame
	#prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
	pts = fast.detect(prev_gray, None)
	prev_pts = np.array(cv2.KeyPoint_convert(pts))
	success, frame = cap.read() 
	if not success: 
		break 

	curr = cv2.resize(frame, (0, 0), fx = resize_factor, fy = resize_factor) 
	curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
	blurred_img = cv2.GaussianBlur(curr_gray, (11, 11), 0)

	blur = cv2.Laplacian(curr_gray, cv2.CV_64F).var()
	bgtv = np.mean(blurred_img)

	if (blur<BLURINESS):
		#print("Blur Value: ", blur)
		cv2.imwrite(PATH + 'blur/' + str(time.time())+".png", frame)

	elif (bgtv<BRIGHTNESS):
		#print(" Brightness: ", bgtv)
		cv2.imwrite(PATH + 'brightness/' + str(time.time())+".png", frame)

	# View Keypoints
	#cv2.drawKeypoints(curr, pts, curr, color=(255,0,0))
	#cv2.imshow("Temp", curr)
	#cv2.waitKey(0)

	# Calculate optical flow 
	curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
	#curr_pts, status, err = cv2.optflow.calcOpticalFlowSparseRLOF(prev, curr, prev_pts, None) 

	# Filter only valid points
	idx = np.where(status==1)[0]
	prev_pts = prev_pts[idx]
	curr_pts = curr_pts[idx]

	#Find transformation matrix
	m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)  
	dx = m[0,2]
	dy = m[1,2]
	da = np.arctan2(m[1,0], m[0,0])
	transforms[i] = [dx,dy,da]
	prev_gray = curr_gray

trajectory = np.cumsum(transforms, axis=0) 
smoothed_trajectory = smooth(trajectory) 
difference = smoothed_trajectory - trajectory
# Calculate newer transformation array
transforms_smooth = transforms + difference
print("Time: Trajectory - ", time.time() - start)
start_time = time.time()

cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
# Write n_frames-1 transformed frames
for i in range(n_frames-2):
	start = time.time()
	# Read next frame
	success, frame = cap.read() 
	if not success:
		break

	#print("Time: Read", time.time() - start)
	start = time.time()

	# Extract transformations from the new transformation array
	dx = transforms_smooth[i,0] / resize_factor
	dy = transforms_smooth[i,1] / resize_factor
	da = transforms_smooth[i,2] 
	m = np.zeros((2,3), np.float32)
	m[0,0] = np.cos(da); 	m[0,1] = -np.sin(da)
	m[1,0] = np.sin(da); 	m[1,1] = np.cos(da)
	m[0,2] = dx; 			m[1,2] = dy

	# Apply affine wrapping to the given frame
	frame_stabilized = cv2.warpAffine(frame, m, (wid, hgt))
	#print("Time: Stab", time.time() - start)
	start = time.time()

	frame_stabilized = fixBorder(frame_stabilized)
	#print("Time: Border", time.time() - start)
	start = time.time()

	'''	# Color Correction
	frame_stab_lab = cv2.cvtColor(frame_stabilized, cv2.COLOR_BGR2Lab);
	frame_stab_lab[0] = clahe.apply(frame_stab_lab[0])
	frame_stab_color = cv2.cvtColor(frame_stab_lab, cv2.COLOR_Lab2BGR);	'''

	# Write the frame to the file
	frame_out = cv2.hconcat([frame, frame_stabilized])
	#print("Time: Concat", time.time() - start, "\n")

	out.write(frame_out)
	#print("Time: Write", time.time() - start, "\n")
	#cv2.imshow("Before and After", frame_out)
	#cv2.waitKey(1)

print("Time: Processing - ", time.time() - start_time)
cap.release()
out.release()
cv2.destroyAllWindows()
