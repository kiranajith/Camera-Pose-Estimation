## ENPM 673 Perception for autonomous Robots 
### Project 2
---
**1. Camera Pose Estimation**


The task is to compute the rotation and translation between the camera and a frame of reference on the corner of a sheet of paper from a given video. The pipeline used for this task involves several steps: noise reduction, edge detection, line detection, and homography.

**Pipeline**
1. Noise Reduction: This is achieved by applying a Gaussian filter to each grayscale frame from the video, which helps to reduce noise in the signal by smoothing the image.

2. Edge Detection: Canny edge detection is used here, which identifies areas of rapid intensity change. The technique includes non-maximum suppression and hysteresis for a continuous contour. After this, morphological closing of spaces and gaps is performed through dilation and erosion.

3. Line Detection: Hough transformation is used to identify lines in the image.

4. Compute Homography: The homography matrix, a 3x3 matrix, is computed to connect the coordinates of points in one image to another.


**2. Image Sticting**

The task is to create a panoramic image by stitching together four images taken from the same camera position (rotation only, no translation).

**Pipeline**

1. SIFT Detection: Feature detection is performed using the technique of Scale-Invariant Feature Transform (SIFT). This technique detects key points in an image based on scale, rotation, and illumination.

2. Feature Matching: Brute force algorithm is used to find similar features between two images. Each feature is compared using a distance metric and the best match is selected based on the smallest distance.

