## Writeup


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_calibration_image/undistorted-image.png "Undistorted"
[image2]: ./output_images/undistorted_calibration_image/undistorted-image01.png "Road Transformed"
[image3]: ./output_images/thresholded_binary_image/threshold.png "Binary Example"
[image4]: ./output_images/perspective_transformed_image/perspective.png "Warp Example"
[image5]: ./output_images/lanelines_polynomial/polynomail.png "Fit Visual"
[image6]: ./output_images/final_image/final.png "Output"
[video1]: ./result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---


### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "./Advanced Lane Finding Project.ipynb" .

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Undistort image corrected.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Use color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image .  Here's an example of my output for this step.

```python
    sx_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=3,thresh=(20, 200))
    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=(np.pi/6, np.pi/2))
    hls_s_binary = hls_select_s(img, thresh=(150, 255))
    hls_l_binary = hls_select_l(img, thresh=(120, 255))
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[((sx_binary==1)&(dir_binary==1))|((hls_s_binary==1)&(hls_l_binary==1))] = 1
```

![alt text][image3]

#### 3. Perspective transform.

```python
def get_perspective_M():
    bottom_left = [220,720]
    bottom_right = [1110, 720]
    top_left = [570, 470]
    top_right = [722, 470]

    source = np.float32([bottom_left,bottom_right,top_right,top_left])

    pts = np.array([bottom_left,bottom_right,top_right,top_left], np.int32)
    pts = pts.reshape((-1,1,2))
    bottom_left = [320,720]
    bottom_right = [920, 720]
    top_left = [320, 1]
    top_right = [920, 1]

    dst = np.float32([bottom_left,bottom_right,top_right,top_left])

    M = cv2.getPerspectiveTransform(source, dst)
    M_inv = cv2.getPerspectiveTransform(dst, source)

    return M,M_inv
```
The code for my perspective transform includes a function called `get_perspective_M`,I verified that my perspective transform was working as expected by drawing the `source` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
I did this in my code includes two function called `find_lane_pixels()` and `search_around_poly()`

![alt text][image5]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

```python
def measure_curvature_real(ploty,left_fitx,right_fitx):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad
def get_rad_offset_str(left_curve_rad,right_curve_rad):
    average_curve_rad = (left_curve_rad + right_curve_rad)/2
    curvature_string = "Radius of curvature: %.2f m" % average_curve_rad
    #print(curvature_string)
    img_size = (image_shape[1], image_shape[0])
    # compute the offset from the center
    lane_center = (right_fitx[-1] + left_fitx[-1])/2
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    center_offset_pixels = abs(img_size[0]/2 - lane_center)
    center_offset_mtrs = xm_per_pix*center_offset_pixels
    offset_string = "Center offset: %.2f m" % center_offset_mtrs
    #print(offset_string)
    return curvature_string,offset_string
```

#### 6. Plotted back down onto the road.

```python
# Create an image to draw the lines on
def inverse_transform(undistort_img,warped,M_inv,curvature_string,offset_string,ploty,left_fitx,right_fitx,show=False):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (image_shape[1], image_shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistort_img, 1, newwarp, 0.3, 0)
    cv2.putText(result,curvature_string,(120,90),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255), thickness=3)
    cv2.putText(result,offset_string,(120,150),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255), thickness=3)
    if show == True:
        f, ax1 = plt.subplots(1, 1, figsize=(9, 6))
        f.tight_layout()
        ax1.imshow(result)
        ax1.set_title('Final Image', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    return result
```
 Here is an example of my result on a test image:

![alt text][image6]

---

### Video Processing Pipeline:


First I am going to establish a class Line() for the lines to store attributes about the lane lines from one frame to the next. Inside the class I will define several functions which will be used to detect the pixels belonging to each lane line.
Here's a [link to my video result](./project_video.mp4)

---


### Discussion

#### Gradient & Color Thresholding
1. I had to experiment a lot with gradient and color channnel thresholding.
2. The lanes lines in the challenge and harder challenge videos were extremely difficult to detect. They were either too bright or too dull. This prompted me to have R & G channel thresholding and L channel thresholding

#### Bad Frames
1. The challenge video has a section where the car goes underneath a tunnel and no lanes are detected
2. To tackle this I had to resort to averaging over the previous well detected frames
3. The lanes in the challenge video change in color, shape and direction. I had to experiment with color threholds to tackle this. Ultimately I had to make use of R, G channels and L channel thresholds.

### Points of failure & Areas of Improvement

The pipeline seems to fail for the harder challenge video. This video has sharper turns and at very short intervals.I think what I could improve is:
1. Take a better perspective transform: choose a smaller section to take the transform since this video has sharper turns and the lenght of a lane is shorter than the previous videos.
2. Average over a smaller number of frames. Right now I am averaging over 12 frames. This fails for the harder challenge video since the shape and direction of lanes changes quite fast.