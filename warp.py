import numpy as np

def warp_img(src_img, H, dst_img_size):
    dst_img = np.zeros([dst_img_size[0], dst_img_size[1], 3],dtype=np.uint8)

    M = dst_img.shape[0]
    N = dst_img.shape[1]

    for i in range(N):
        for j in range(M):
            coords = [i, j, 1]
            coords = np.asarray(coords)
            coords = coords.transpose()
            H_inv = np.linalg.inv(H)
            new_pts = np.matmul(H_inv, coords)

            src_x = round(new_pts[0]/new_pts[2])
            src_y = round(new_pts[1]/new_pts[2])

            if (src_y < 0 or src_x < 0) or (src_x > src_img.shape[1] or src_y > src_img.shape[0]):
                dst_img[j, i] = 0

            elif (src_y > 0 and src_x > 0) and (src_x < src_img.shape[1] and src_y < src_img.shape[0]):
                dst_img[j, i] = src_img[src_y, src_x]

    return dst_img[:M, :N]

