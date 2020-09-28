import cv2 as cv
import numpy as np
######steps of watershed algorithm
####1.input image(remember to filt out noises); 2.BGR2GRAY; 3.attain binary image; 4.distance transform; 5.seeking seed; 6.generate marker;
####7.watershed transform; 8.output image------>end
def watershedalgorithm_demo(image):
    print(image.shape)
    blurred = cv.pyrMeanShiftFiltering(image,10,100)#####filt noise!!!!!
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    #morphology_operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations = 2)
    sure_bg = cv.dilate(mb, kernel, iterations = 3)
    cv.imshow("mor-opt", sure_bg)

    ###distance transform
    dist = cv.distanceTransform(mb, cv.DIST_L2, 3)###mask == 3; will be bugged filling in other nums
    dist_output = cv.normalize(dist,0,1.0,cv.NORM_MINMAX)
    cv.imshow("distance-t",dist_output)

    ret, surface = cv.threshold(dist, dist.max()*0.6, 255, cv.THRESH_BINARY)
    cv.imshow("surface-bin",surface)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)

    #####watershed transform
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers = markers)
    image[markers == -1] = [0,0,255]
    cv.imshow("result",image)

img = cv.imread("/home/subaraci/sda/smarties.png")
cv.namedWindow('Image', cv.WINDOW_AUTOSIZE)
cv.imshow('Image', img)

watershedalgorithm_demo(img)

cv.waitKey(0)

cv.destroyAllWindows()