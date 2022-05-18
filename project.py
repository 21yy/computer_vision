import cv2
import numpy
import random
import sys

sobelKernel_x = numpy.array(([-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]), dtype="float32")
sobelKernel_y = numpy.array(([1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]), dtype="float32")

# to get a gaussian filter
def gaussian_filter(sigma, size):
    m = (size - 1.0) / 2.0
    n = (size - 1.0) / 2.0
    y, x = numpy.ogrid[-m: m + 1, -n: n + 1]
    h = numpy.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    h[h < numpy.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# MOPS-non_maximum_suppression
def non_maximum_suppression(res_list, point_list, kps):
    fmax = max(res_list)
    r_list = []
    for i in range(len(res_list)):
        vector_d = []
        if res_list[i] > fmax * 0.9:
            r_list.append([100000, i])
        else:
            for j in range(len(res_list)):
                if i != j:
                    if res_list[i] < res_list[j] * 0.9:
                        ssd = numpy.sqrt((point_list[i][1] - point_list[j][1])**2 + (point_list[i][0] - point_list[j][0])**2)
                        vector_d.append(ssd)
            r_list.append([min(vector_d), i])
    r_list = sorted(r_list)
    po_list = []
    kp_list = []
    for i in range(len(r_list[:500])):
        index = r_list[i][1]
        kp_list.append(kps[index])
        po_list.append(point_list[index])
    return kp_list, po_list


# find corner in the image
def find_keypoints(img):
    kps = []
    pointlist = []
    thresh = 250
    dx = cv2.filter2D(img.astype(float), -1, 1 / 8 * sobelKernel_x)
    dy = cv2.filter2D(img.astype(float), -1, 1 / 8 * sobelKernel_y)
    ixx = dx ** 2
    ixy = dy * dx
    iyy = dy ** 2
    height = img.shape[0]
    width = img.shape[1]
    gaussian_kernel = gaussian_filter(1, 3)
    ixx = cv2.filter2D(ixx, -1, gaussian_kernel)
    ixy = cv2.filter2D(ixy, -1, gaussian_kernel)
    iyy = cv2.filter2D(iyy, -1, gaussian_kernel)
    det = (ixx * iyy) - (ixy ** 2)
    trace = ixx + iyy + 0.00001
    c = det / trace
    response_list = []
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            window = c[i - 1:i + 2, j - 1: j + 2]
            row, column = divmod(numpy.argmax(window), window.shape[1])
            if window[row, column] > thresh:
                pixel_i = i - 1 + row
                pixel_j = j - 1 + column
                point = cv2.KeyPoint(pixel_j, pixel_i, 20)
                if [pixel_j, pixel_i] not in pointlist:
                    response_list.append(c[pixel_i, pixel_j])
                    kps.append(point)
                    pointlist.append([pixel_j, pixel_i])
    return non_maximum_suppression(response_list, pointlist, kps)
    #return kps, pointlist


def match(kp1, kp2, des1, des2):

    matchlist1 = []
    matchlist2 = []
    ssdList = []
    for i in range(0, len(des1)):
        pos1 = 0
        temp2 = 1000
        temp1 = numpy.sum((des1[0] - des2[0])**2)
        for j in range(1, len(des2)):
            ssd = numpy.sum((des1[i] - des2[j])**2)
            if temp1 > ssd:
                temp2 = temp1
                temp1 = ssd
                pos1 = j
            elif temp1 <= ssd and temp2 > ssd:
                temp2 = ssd
        ratio = temp1 / temp2
        if ratio < 0.1:
            matchlist1.append(kp1[i])
            matchlist2.append(kp2[pos1])
            ssdList.append(temp1)

    matches_1 = []
    for i in range(len(matchlist1)):
        edge = cv2.DMatch(i, i, 0, ssdList[i])
        matches_1.append(edge)
    return matchlist1, matchlist2, matches_1


def find_kps(img1, img2):

    sift = cv2.xfeatures2d.SIFT_create()
    kp1 = sift.detect(img1, None)
    kp2 = sift.detect(img2, None)
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)
    matchlist1, matchlist2, matches_1 = match(kp1, kp2, des1, des2)
    return matchlist1, matchlist2, matches_1


def project(x1, y1, H):
    temp = numpy.dot(H, numpy.float32((x1, y1, 1)).reshape(3, 1))
    x_2 = temp[0]/temp[2]
    y_2 = temp[1]/temp[2]
    return x_2[0], y_2[0]


def computeInlierCount(H, matches, inlierThreshold):
    num_matches = []
    for mch in matches:
        kp1 = keyPoints1[mch.queryIdx].pt
        kp2 = keyPoints2[mch.trainIdx].pt
        x_2, y_2 = project(kp1[0], kp1[1], H)
        distance = numpy.sqrt((kp2[0] - x_2)**2 + (kp2[1] - y_2)**2)
        if distance < inlierThreshold:
            num_matches.append(mch)
    return num_matches


def RANSAC(matches, numIterations, inlierThreshold):

    better_h = []
    num_matches = []

    for i in range(numIterations):
        match1 = matches[random.randint(0, len(matches)-1)]
        match2 = matches[random.randint(0, len(matches)-1)]
        match3 = matches[random.randint(0, len(matches)-1)]
        match4 = matches[random.randint(0, len(matches)-1)]
        pts_src = numpy.array([keyPoints1[match1.queryIdx].pt, keyPoints1[match2.queryIdx].pt,
                               keyPoints1[match3.queryIdx].pt, keyPoints1[match4.queryIdx].pt])
        pts_dis = numpy.array([keyPoints2[match1.queryIdx].pt, keyPoints2[match2.queryIdx].pt,
                               keyPoints2[match3.queryIdx].pt, keyPoints2[match4.queryIdx].pt])
        H, s = cv2.findHomography(pts_src, pts_dis, 0)
        numMatches = computeInlierCount(H, matches, inlierThreshold)
        if len(numMatches) > len(num_matches):
            better_h = H
            num_matches = numMatches
    pts_src = []
    pts_dis = []
    for inlier_mch in num_matches:
        pts_src.append(keyPoints1[inlier_mch.queryIdx].pt)
        pts_dis.append(keyPoints2[inlier_mch.trainIdx].pt)
    hom, s = cv2.findHomography(numpy.array(pts_src), numpy.array(pts_dis), 0)
    homInv, s = cv2.findHomography(numpy.array(pts_dis), numpy.array(pts_src), 0)
    return better_h, num_matches, hom, homInv


def stich(img1, img2, hom, homInv):
    size1 = img1.shape
    size2 = img2.shape
    corner1 = project(0, 0, homInv)
    corner2 = project(size2[1] - 1, 0, homInv)
    corner3 = project(0, size2[0] - 1, homInv)
    corner4 = project(size2[1] - 1, size2[0] - 1, homInv)
    max_width = max((corner1[0], corner2[0], corner3[0], corner4[0])).astype(numpy.int32)
    min_width = min((corner1[0], corner2[0], corner3[0], corner4[0])).astype(numpy.int32)
    max_height = max((corner1[1], corner2[1], corner3[1], corner4[1])).astype(numpy.int32)
    min_height = min((corner1[1], corner2[1], corner3[1], corner4[1])).astype(numpy.int32)

    height = size1[0]
    width = size1[1]
    lb_x = 0
    lb_y = 0
    if min_width < 0:
        lb_y = numpy.abs(min_width)
        width = width + lb_y
    if min_height < 0:
        lb_x = numpy.abs(min_height)
        height = height + lb_x
    if size1[0] < max_height:
        height = height + max_height - size1[0]
    if size1[1] < max_width:
        width = width + max_width - size1[1]
    stichImage = numpy.zeros([height, width, 3], numpy.uint8)
    stichImage[lb_x: lb_x + size1[0], lb_y: lb_y + size1[1]] = img1
    for i in range(height):
        for j in range(width):
            pixel_x, pixel_y = project(j - lb_y, i - lb_x, hom)
            if 0 <= pixel_x < size2[1] and 0 <= pixel_y < size2[0]:
                stichImage[i, j] = cv2.getRectSubPix(img2, (1, 1), (pixel_x, pixel_y))
    return stichImage


image_list = []
for i in range(len(sys.argv)-1):
    image = cv2.imread(sys.argv[1+i])
    image_list.append(image)

stichImage = None
if len(image_list) < 2:
    image = image_list[0]
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    kp1, pixels = find_keypoints(image.astype(numpy.double))
    cv2.imwrite("la.png", cv2.drawKeypoints(image_list[i], kp1, image_list[i], (0, 0, 255)))
for i in range(len(image_list)-1):

    keyPoints1, keyPoints2, matches1 = find_kps(image_list[i], image_list[i+1])
    if i == 0 and len(image_list) >= 2:
        #cv2.imwrite("lb.png", cv2.drawKeypoints(image_list[i], keyPoints1, image_list[i], (0, 0, 255)))
        #cv2.imwrite("lc.png", cv2.drawKeypoints(image_list[i+1], keyPoints2, image_list[i+1], (0, 0, 255)))
        cv2.imwrite("2.png", cv2.drawMatches(image_list[i], keyPoints1, image_list[i + 1], keyPoints2, matches1, None))
    h, matchlist, hom, homInv = RANSAC(matches1, 100, 2)
    if i == 0:
        kp_1 = []
        kp_2 = []
        k = 0
        for mch in matchlist:
            kp_1.append(keyPoints1[mch.queryIdx])
            kp_2.append(keyPoints2[mch.trainIdx])
            mch.queryIdx = k
            mch.trainIdx = k
            k = k + 1
        cv2.imwrite("3.png", cv2.drawMatches(image_list[i], kp_1, image_list[i+1], kp_2, matchlist, None))
    stichImage = stich(image_list[i], image_list[i+1], hom, homInv)
    if i == 0:
        cv2.imwrite("4.png", stichImage)
    image_list[i+1] = stichImage

if len(image_list) > 2:
    cv2.imshow("AllStitched Image", stichImage)
    cv2.imwrite("AllStitched.png", stichImage)
    cv2.waitKey(0)





