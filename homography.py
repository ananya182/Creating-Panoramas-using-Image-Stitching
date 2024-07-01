import random
import numpy as np

def get_perspective_transform(sourcePoints, destinationPoints):

    a = np.zeros((8, 8))
    b = np.zeros((8))
    for i in range(4):
        a[i][0] = a[i+4][3] = sourcePoints[i][0]
        a[i][1] = a[i+4][4] = sourcePoints[i][1]
        a[i][2] = a[i+4][5] = 1
        a[i][3] = a[i][4] = a[i][5] = 0
        a[i+4][0] = a[i+4][1] = a[i+4][2] = 0
        a[i][6] = -sourcePoints[i][0]*destinationPoints[i][0]
        a[i][7] = -sourcePoints[i][1]*destinationPoints[i][0]
        a[i+4][6] = -sourcePoints[i][0]*destinationPoints[i][1]
        a[i+4][7] = -sourcePoints[i][1]*destinationPoints[i][1]
        b[i] = destinationPoints[i][0]
        b[i+4] = destinationPoints[i][1]

    x = np.linalg.solve(a, b)
    x.resize((9,), refcheck=False)
    x[8] = 1 
    return x.reshape((3,3))

def compute_homography(src_pts, dst_pts):
  N = src_pts.shape[0]

  H=[]
  src_array = np.asarray(src_pts)
  dst_array = np.asarray(dst_pts)

  for n in range(N):
    src = src_array[n]
    H.append(-src[0])
    H.append(-src[1])
    H.append(-1)
    H.append(0)
    H.append(0)
    H.append(0)

  H = np.asarray(H)
  H1 = H.reshape(2*N,3)

  H2 = np.zeros([2*N, 3], dtype=int)
  for i in range(0,2*N,2):
    H2[i:i+2,0:i+3] = np.flip(H1[i:i+2,0:i+3], axis=0)

  H2 = np.asarray(H2)
  H3 = np.concatenate((H1, H2), axis=1)

  H4=[]
  for n in range(N):
    src = src_array[n]
    dst = dst_array[n]

    H4.append(src[0]*dst[0])
    H4.append(src[1]*dst[0])
    H4.append(dst[0])
    H4.append(src[0]*dst[1])
    H4.append(src[1]*dst[1])
    H4.append(dst[1])

  H4 = np.asarray(H4)
  H4 = H4.reshape(2*N,3)

  H5 = np.concatenate((H3, H4), axis=1)
  H8 = np.matmul(np.transpose(H5), H5)

  w, v = np.linalg.eig(H8)
  minimum = w.min()
  for i in range(len(w)):
    if w[i] == minimum:
      a = v[:, i]

  a = np.asarray(a)
  a = a.reshape(3,3)
  a = a/a[2,2] 

  return a


def apply_homography(test, H):
  dst_output = []
  N = test.shape[0]

  for row in test:
    input = np.matrix([row[0,0], row[0,1], 1])
    input = input.transpose()
    mapped_pts = np.matmul(H, input)
    dst_output.append(mapped_pts[0]/mapped_pts[2])
    dst_output.append(mapped_pts[1]/mapped_pts[2])

  dst_output = np.asarray(dst_output)
  dst_output = dst_output.reshape(N, 2)
  
  return dst_output


def RANSAC(Xs, Xd, max_iter, eps):

  H = np.zeros([3,3])
  

  inliers_ids = []
  inliers_counts = []

  n = Xs.shape[0]
  iter = 0
  while iter < max_iter:
    inliers_id = []
    pts_index = random.sample(range(0, n), 4)

    Xs_new = []
    Xd_new = []

    for pt in range(4):
      Xs_new.append(Xs[pts_index[pt]][:])
      Xd_new.append(Xd[pts_index[pt]][:])

    Xs_new = np.asarray(Xs_new)
    Xd_new = np.asarray(Xd_new)
    Xs_new = np.asmatrix(Xs_new)
    Xd_new = np.asmatrix(Xd_new)

    H = compute_homography(Xs_new, Xd_new)

    Xs = np.asmatrix(Xs)
    Xd_predicted = apply_homography(Xs, H)

    for i in range(n):
      SSD = ((np.round(Xd_predicted[i][0]) - int(Xd[i, 0]))**2 + (np.round(Xd_predicted[i][1]) - int(Xd[i, 1]))**2)

      if SSD < eps:
        if i not in inliers_id:
          inliers_id.append(i)
    
    inliers_ids.append(inliers_id)
    inliers_counts.append(len(inliers_id))
    
    iter += 1

  largest_count_index = inliers_counts.index(max(inliers_counts))
  best_inliers_id = inliers_ids[largest_count_index]

  Xs_inliers = []
  Xd_inliers = []
  for i in best_inliers_id:
    Xs_inliers.append(Xs[i][:])
    Xd_inliers.append(Xd[i][:])

  Xs_inliers = np.asarray(Xs_inliers)
  Xd_inliers = np.asarray(Xd_inliers)
  Xs_inliers = np.asmatrix(Xs_inliers)
  Xd_inliers = np.asmatrix(Xd_inliers)

  H = compute_homography(Xs_inliers, Xd_inliers)

  return best_inliers_id, H

def invoke_RANSAC(matches, BaseImage_kp, SecImage_kp):

    BaseImage_pts = []
    SecImage_pts = []
    for Match in matches:
        BaseImage_pts.append(BaseImage_kp[Match[0]].pt)
        SecImage_pts.append(SecImage_kp[Match[1]].pt)

    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

    _,HomographyMatrix=RANSAC(SecImage_pts, BaseImage_pts,1000,4)

    return HomographyMatrix

