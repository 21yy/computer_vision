import cv2
import numpy

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


def fit_parabola(hist, b_no, bin_width):
    center_val = b_no * bin_width + bin_width / 2.
    if b_no == len(hist) - 1:
        right_val = 360 + bin_width / 2.
    else:
        right_val = (b_no + 1) * bin_width + bin_width / 2.
    if b_no == 0:
        left_val = -bin_width / 2.
    else:
        left_val = (b_no - 1) * bin_width + bin_width / 2.
    A = numpy.array([
        [center_val ** 2, center_val, 1],
        [right_val ** 2, right_val, 1],
        [left_val ** 2, left_val, 1]], 'float32')
    b = numpy.array([
        hist[b_no],
        hist[(b_no + 1) % len(hist)],
        hist[(b_no - 1) % len(hist)]], 'float32')
    x = numpy.linalg.lstsq(A, b, rcond=None)[0]
    if x[0] == 0:
        x[0] = 1e-6
    return -x[1] / (2 * x[0])


# find dominant gradient orientation of feature points
def gradient_orientation(img, x, y, dx, dy):
    height = img.shape[0]
    width = img.shape[1]
    theta_list = numpy.zeros(36)

    for i in range(x - 4, x + 4):
        for j in range(y - 4, y + 4):
            if i < 0 or i > height - 1:
                continue
            if j < 0 or j > width - 1:
                continue
            if i == 0:
                a = dx[i + 1, j] - dx[i, j]
            elif i == height - 1:
                a = dx[i, j] - dx[i - 1, j]
            else:
                a = dx[i+1, j] - dx[i-1, j]
            if j == 0:
                b = dy[i, j+1] - dy[i, j]
            elif j == width - 1:
                b = dy[i, j] - dy[i, j - 1]
            else:
                b = dy[i, j + 1] - dy[i, j - 1]
            magnitude = numpy.sqrt(a**2 + b**2)
            theta = numpy.rad2deg(numpy.arctan2(b, a))
            index = int(theta/10)
            theta_list[index] = theta_list[index] + magnitude
    pos = numpy.where(theta_list == max(theta_list))
    return fit_parabola(theta_list, pos[0], 10)


# find orientation histogram for each cell(4x4)
def get_hist_subregion(magnitude_m, theta_m, dominant_ang):
    hist = numpy.zeros(8, numpy.float32)
    for magnitude, angle in zip(magnitude_m, theta_m):
        angle = (angle - dominant_ang) % 360
        binno = int(numpy.floor(angle)//45)
        hist_interp_weight = 1 - abs(angle - (binno * 45 + 22.5)) / (45 / 2)
        magnitude *= max(hist_interp_weight, 1e-6)
        gy, gx = numpy.unravel_index(1, (4, 4))
        x_interp_weight = max(1 - abs(gx - 1.5) / 1.5, 1e-6)
        y_interp_weight = max(1 - abs(gy - 1.5) / 1.5, 1e-6)
        magnitude *= x_interp_weight * y_interp_weight
        hist[binno] += magnitude
    return hist


# get 128 dimension of gradient direction for each feature point
def feature_description(img, p_list):
    dx = cv2.filter2D(img.astype(float), -1, 1 / 8 * sobelKernel_x)
    dy = cv2.filter2D(img.astype(float), -1, 1 / 8 * sobelKernel_y)
    descreptor = []
    gaussian_kernel = gaussian_filter(1, 16)
    kernel = gaussian_kernel
    for pixel in p_list:
        theta = gradient_orientation(img, pixel[1], pixel[0], dx, dy)
        pixel.append(theta[0])

        x1, y1 = max(0, pixel[1] - 8), max(0, pixel[0] - 8)
        x2, y2 = min(pixel[1] + 8, img.shape[0]-1), min(pixel[0] + 8, img.shape[1]-1)
        dx_w = dx[x1:x2, y1:y2]
        dy_w = dy[x1:x2, y1:y2]
        if dx_w.shape[0] < 17:
            if x1 == 0:
                kernel = kernel[gaussian_kernel.shape[0] - dx_w.shape[0]:]
            else:
                kernel = kernel[:dx_w.shape[0]]
        if dx_w.shape[1] < 17:
            if y1 == 0:
                kernel = kernel[:, 0:dx_w.shape[1]]
            else:
                kernel = kernel[:, :dx_w.shape[1]]

        dx_w = dx_w * kernel
        dy_w = dy_w * kernel
        m = numpy.sqrt(dx_w ** 2 + dy_w ** 2)
        thetaM = numpy.rad2deg(numpy.arctan2(dy_w, dx_w))
        vectors = numpy.zeros(128, numpy.float32)
        for i in range(0, 4):
            for j in range(0, 4):
                x1, y1 = i * 4, j * 4
                x2, y2 = (i + 1) * 4, (j + 1) * 4

                hist = get_hist_subregion(m[x1:x2, y1:y2].flatten(), thetaM[x1:x2, y1:y2].flatten(), pixel[2])
                vectors[i * 4 * 8 + j * 8:i * 4 * 8 + (j + 1) * 8] = hist.flatten()
        vectors[vectors > 0.2] = 0.2
        descreptor.append(vectors)
        kernel = gaussian_kernel

    return numpy.array(descreptor)


# to find match points
def match(img1, img2):
    keypoints1, pixels1 = find_keypoints(img1)
    descriptors1 = feature_description(img1, pixels1)
    keypoints2, pixels2 = find_keypoints(img2)
    descriptors2 = feature_description(img2, pixels2)

    # marchlist for ratio test
    matchlist1 = []
    matchlist2 = []
    # marchlist for ssd < threshold
    matchlist11 = []
    matchlist22 = []
    ssdList = []
    ssd2_list = []
    for i in range(0, len(descriptors1)):
        pos1 = 0
        temp2 = 1000
        temp1 = numpy.sum((descriptors1[0] - descriptors2[0])**2)
        for j in range(1, len(descriptors2)):
            ssd = numpy.sum((descriptors1[i] - descriptors2[j])**2)
            if temp1 > ssd:
                temp2 = temp1
                temp1 = ssd
                pos1 = j
            elif temp1 <= ssd and temp2 > ssd:
                temp2 = ssd
        ratio = temp1 / temp2
        if ratio < 0.4:
            matchlist1.append(keypoints1[i])
            matchlist2.append(keypoints2[pos1])
            ssdList.append(temp1)
        if temp1 < 0.0000000000023:
            matchlist11.append(keypoints1[i])
            matchlist22.append(keypoints2[pos1])
            ssd2_list.append(temp1)
    matches_1 = []
    matches_2 = []
    for i in range(len(matchlist1)):
        edge = cv2.DMatch(i, i, 0, ssdList[i])
        matches_1.append(edge)
    for i in range(len(matchlist11)):
        edge = cv2.DMatch(i, i, 0, ssd2_list[i])
        matches_2.append(edge)
    return matchlist1, matchlist2, matches_1, matchlist11, matchlist22, matches_2
    #return keypoints1, keypoints2, matches


# main() read images and find match points between them and show it
# image1 = cv2.imread("./Yosemite1.jpg")
# image2 = cv2.imread("./Yosemite2.jpg")         #350, 0.4
image1 = cv2.imread("./Rainier1.png")
image2 = cv2.imread("./Rainier2.png")
newImg1 = image1.copy()
newImg2 = image2.copy()
if len(image1.shape) == 3:
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
elif len(image1.shape) == 4:
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGBA2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGBA2GRAY)

keyPoints1, keyPoints2, matchesV1, keyPoints11, keyPoints22, matchesV2 = match(image1, image2)

cv2.drawKeypoints(newImg1, keyPoints1, newImg1, (0, 0, 255))
cv2.drawKeypoints(newImg2, keyPoints2, newImg2, (0, 0, 255))

newImg = numpy.concatenate((newImg1, newImg2), axis=1)
cv2.imshow("feature points", newImg)

cv2.imshow("feature matching by ratio test", cv2.drawMatches(newImg1, keyPoints1, newImg2, keyPoints2, matchesV1, None))
cv2.imshow("feature matching by SSD", cv2.drawMatches(newImg1, keyPoints11, newImg2, keyPoints22, matchesV2, None))
cv2.waitKey(0)

