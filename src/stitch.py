import sys
import os
import cv2
import numpy as np
from scipy.spatial import distance

def get_homography_matrix(src, dst):
    assert(src.shape[1]==dst.shape[1])
    assert(src.shape[0]==2)
    schema = np.zeros((2*src.shape[1],9))
    src = src.T
    dst = dst.T
    
    length = src.shape[0]
    for i in range(0,length):
        x,y = src[i,0], src[i,1]
        u,v = dst[i,0], dst[i,1]
        schema[i*2,:] = np.array([-x,-y,-1,0,0,0,x*u,y*u,u])
        schema[i*2+1,:] = np.array([0,0,0,-x,-y,-1,v*x,v*y,v])
    
    [Z,B] = np.linalg.eig(np.matmul(schema.T,schema))
    idx = np.argmin(Z)
    homography_matrix = np.reshape(B[:,idx], (3,3))
    return homography_matrix  

def self_Matcher(keypoints1, descriptors1, keypoints2, descriptors2):
    o = 0
    pairs = []
    r = (1 << np.arange(8))[:,None]
    for i in range(len(descriptors1)):
        min_value = 0
        for j in range(len(descriptors2)):
            total = 0
            for k in range(len(descriptors1[0])):
                x = descriptors1[i][k] & r
                y = descriptors2[j][k] & r
                diff_bits = np.count_nonzero(x !=y)
                total = total + diff_bits
            if(min_value == 0 or min_value > total):
                min_value = total
                o = j
        pairs.append([i, o, min_value])

    return pairs

def warp_image(H, img2, img1):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    lp1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
    temp = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)
    lp2 = cv2.perspectiveTransform(temp, H)
    lp = np.concatenate((lp1, lp2), axis=0)

    [x_min, y_min] = np.int32(lp.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(lp.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

    result = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    result[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img1
    return result

def get_inliner_from_ransac(source_points, destination_points, threshold = 1):
    max_inliers = 0
    best_inliers = []
    
    while True:
        for i in range(1000):
            inliers = []
            random_pts = np.random.choice(range(0, len(source_points) - 1), 4, replace=False)

            src_sample = []
            dst_sample = []
            for i in random_pts:
                src_sample.append(source_points[i])
                dst_sample.append(destination_points[i])

            h = get_homography_matrix(np.asarray(src_sample).T, np.asarray(dst_sample).T)
            
            count = 0
            for index in range(len(source_points)):
                src_pt = np.append(source_points[index], 1)
                dest_pt = np.dot(h, src_pt.T)
                dest_pt = np.true_divide(dest_pt, dest_pt[2])[0: 2]
                if distance.euclidean(destination_points[index], dest_pt) <= threshold:
                    count += 1
                    inliers.append(True)
                else:
                    inliers.append(False)
            if count < 0.4 * len(source_points) and count > max_inliers:
                max_inliers = count
                best_inliers = inliers
        
        if sum(best_inliers) > 0.1 * len(source_points) and sum(best_inliers) < 0.4 * len(source_points):       # control the over and underfitting
            break
        
        if sum(best_inliers) < 0.1 * len(source_points):
            threshold += 1
        else:
            threshold -= 1   

    return best_inliers

def stitch(img1, img2,img11,img22):
    orb = cv2.ORB_create(nfeatures = 1000)
    keypoints1, descriptors1 = orb.detectAndCompute(img1,None)
    orb = cv2.ORB_create(nfeatures = 1000)
    keypoints2, descriptors2 = orb.detectAndCompute(img2,None)

    # gives the pair which matches with each other
    pairs =  self_Matcher(keypoints1, descriptors1, keypoints2, descriptors2)
    src = []
    dst = []
    dis = []
    for pair in pairs:
        src.append(keypoints1[pair[0]].pt)
        dst.append(keypoints2[pair[1]].pt)
        dis.append(pair[2])
    
    if min(dis) > 15:
        return None
    src = np.asarray(src)
    dst = np.asarray(dst)
    inliers =  np.asarray(get_inliner_from_ransac(src,dst))

    homography_Matrix = get_homography_matrix(src.T[:,inliers],dst.T[:,inliers])   # pass inliers to compute homograpy
    panorama_image = warp_image(homography_Matrix, img11,img22)
    return panorama_image

def read_images(img, data_directory):
    img1 = cv2.imread(data_directory + "/" +img,0)
    img2 = cv2.imread(data_directory + "/" +img)
    return img1, img2, img1.shape

def write_panorama(img1_size,img2_size, result, data_directory):
    cv2.imwrite(data_directory+"/panorama.jpg", result)


def prepare_Stitching():
    data_directory = ""
    if len(sys.argv) < 2:
        raise ValueError("data directory argument not supplied")
    else:
        data_directory = "../" + sys.argv[1]
    
    if os.path.exists(data_directory) == False:
        raise ValueError("invalid data directory")
    
    files = os.listdir(data_directory)
    image_files = []
    for filename in files:
        if (filename.lower().endswith(".jpg") or filename.lower().endswith(".png")) and not filename.startswith("panorama"):
            image_files.append(filename)
    
    if len(image_files) < 2:
        raise ValueError("atleast 2 images needed for stitching")
    
    j = 1
    i = 0

    while i < len(image_files):
        img11, img12, size_1 = read_images(image_files[i], data_directory)
        img21, img22, size_2 = read_images(image_files[j], data_directory)
        result = stitch(img11, img21, img12, img22)
        if result is None:
            if  j < len(image_files) - 1:
                j = j + 1
            else:
                raise ValueError("Images provided can not stitched")
        else:
            write_panorama(size_1,size_2, result, data_directory)
            if j == 1:
                j = 2
                if j > len(image_files) - 1:
                    return
            else:
                j = 1  

            img11, img12, size_1 = read_images("panorama.jpg", data_directory)
            img21, img22, size_2 = read_images(image_files[j], data_directory)
            result = stitch(img11, img21, img12, img22)
            write_panorama(size_1,size_2, result, data_directory)
            break

if __name__ == '__main__':
    prepare_Stitching()
    

    

    