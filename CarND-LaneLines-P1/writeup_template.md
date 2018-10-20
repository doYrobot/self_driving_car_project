# **Finding Lane Lines on the Road**


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)
[image1]: ./examples/grayscale.jpg "Grayscale"
---

### Reflection



My pipeline consisted of 5 steps.
- First, I converted the images to grayscale, then I use GaussianBlur.
    ```python
    def grayscale(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    def gaussian_blur(img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    gray=grayscale(img)
    blur_gray=gaussian_blur(gray,5)
    ```
- Second, I define a set of points which describe the region of interest we want to crop out of the original. To actually do the cropping of the image, I define a utility function region_of_interest().
    ```python
    def region_of_interest(img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    height = img.shape[0]
    width = img.shape[1]
    region_of_interest_vertices = np.array([[[0, height],[width/2,height/2],[width, height]]],dtype=np.int32)
    masked_image=region_of_interest(edges,region_of_interest_vertices)
    ```
- Third, try running the Canny Edge Detection algorithm on our cropped image with some reasonable starter thresholds.
    ```python
    def canny(img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)
    edges=canny(blur_gray, 100, 200)
    ```
- Fourth,Using Hough Transforms to Detect Lines.
    ```python
    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return lines,line_img
    lines,line_img=hough_lines(masked_image, rho=6, theta=np.pi/60, threshold=1, min_line_len=40, max_line_gap=25)
    ```
- Fifth,The final step in our pipeline will be to create only one line for each of the groups of lines we found in the last step. This can be done by fitting a simple linear model to the various endpoints of the line segments in each group, then rendering a single overlay line on that linear model.
    ```python
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if math.fabs(slope) < 0.5:
                continue
            if slope <= 0:
                left_line_x.extend([x1,x2])
                left_line_y.extend([y1,y2])
            else:
                right_line_x.extend([x1,x2])
                right_line_y.extend([y1,y2])
    poly_left = np.poly1d(np.polyfit(left_line_y,left_line_x,deg=1))
    poly_right = np.poly1d(np.polyfit(right_line_y,right_line_x,deg=1))
    left_start_x = int(poly_left(img.shape[0]))
    left_end_x = int(poly_left(img.shape[0]*0.6))
    right_start_x = int(poly_right(img.shape[0]))
    right_end_x = int(poly_right(img.shape[0]*0.6))
    draw_lines(line_img, lines=[[[left_start_x,img.shape[0],left_end_x,int(img.shape[0]*0.6)],[right_start_x,img.shape[0],right_end_x,int(img.shape[0]*0.6)]]], color=[255, 0, 0], thickness=5)
    lines_edges=weighted_img(line_img, img, α=0.8, β=1., γ=0.)
    ```


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lane curvature is too large

Another shortcoming could be  a lane is blocked by obstacles.


### 3. Suggest possible improvements to your pipeline

One possible improvement is to use deep learning network to identify lane lines.

Another potential improvement may be the use of 3D images to predict obscured parts.
