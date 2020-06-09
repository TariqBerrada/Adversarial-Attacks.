import numpy as np
import cv2
img = cv2.imread("road_map.jpg")

def edge_detect(file_name, tresh_min, tresh_max):
    image = cv2.imread(file_name)
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    (thresh, im_bw) = cv2.threshold(im_bw, tresh_min, tresh_max, 0)
    cv2.imwrite('bw_'+file_name, im_bw)

    contours = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0], contours[1], contours[2]

def smooth_mask(filename, n):
    mask = cv2.imread(filename)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    nx, ny = np.shape(mask)

    sm_mask = np.ones((nx, ny))
    for i in range(n+1, nx-n):
        for j in range(n+1, ny-n):
            sm_mask[i][j] = np.median(mask[i-n-1:i+n, j-n-1:j+n])
    return sm_mask

#c1, c2, c3 = edge_detect('road_map.jpg', 128, 255)

#cnt = c2


x = np.ones(np.shape(img))

#cv2.drawContours(x, c2, -1, (0,255,0), 3)
x = smooth_mask("road_map.jpg", 5)

"""for i in range(len(c2)-1):
    cv2.line(x, tuple(c2[i][0][0]), tuple(c2[i+1][0][0]), (0,0,0))"""
cv2.imshow("sdf", x)
cv2.waitKey()
