import cv2
import numpy as np
import matplotlib.pyplot as plt
import warp
import homography
import match

def conv_xy(x, y):
    global center, f

    xnew = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    ynew = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
    
    return xnew, ynew

def conv_cylinder(InitialImage):
    global w, h, center, f
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    f = 1050
    
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)
    
    AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = AllCoordinates_of_ti[:, 0]
    ti_y = AllCoordinates_of_ti[:, 1]
    
    ii_x, ii_y = conv_xy(ti_x, ti_y)

    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)

    GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w-2)) * \
                  (ii_tl_y >= 0) * (ii_tl_y <= (h-2))

    ti_x = ti_x[GoodIndices]
    ti_y = ti_y[GoodIndices]
    
    ii_x = ii_x[GoodIndices]
    ii_y = ii_y[GoodIndices]

    ii_tl_x = ii_tl_x[GoodIndices]
    ii_tl_y = ii_tl_y[GoodIndices]

    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx)       * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx)       * (dy)
    
    TransformedImage[ti_y, ti_x, :] = ( weight_tl[:, None] * InitialImage[ii_tl_y,     ii_tl_x,     :] ) + \
                                      ( weight_tr[:, None] * InitialImage[ii_tl_y,     ii_tl_x + 1, :] ) + \
                                      ( weight_bl[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x,     :] ) + \
                                      ( weight_br[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x + 1, :] )


    min_x = min(ti_x)

    TransformedImage = TransformedImage[:, min_x : -min_x, :]

    return TransformedImage, ti_x-min_x, ti_y

def compute_newsize(HomographyMatrix, Sec_ImageShape, Base_ImageShape):
    (Height, Width) = Sec_ImageShape
    
    InitialMatrix = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])
    
    FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)

    [x, y, c] = FinalMatrix
    x = np.divide(x, c)
    y = np.divide(y, c)

    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)
    
  
    if New_Width < Base_ImageShape[1] + Correction[0]:
        New_Width = Base_ImageShape[1] + Correction[0]
    if New_Height < Base_ImageShape[0] + Correction[1]:
        New_Height = Base_ImageShape[0] + Correction[1]

    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    OldInitialPoints = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    NewFinalPonts = np.float32(np.array([x, y]).transpose())
    HomographyMatrix = homography.get_perspective_transform(OldInitialPoints, NewFinalPonts)
    
    return [New_Height, New_Width], Correction, HomographyMatrix

def stitch(base_img,sec_img):

    sec_img_cyl, mask_x, mask_y = conv_cylinder(sec_img)
    sec_img_mask = np.zeros(sec_img_cyl.shape, dtype=np.uint8)
    sec_img_mask[mask_y, mask_x, :] = 255

    matches, base_img_kpts, sec_img_kpts = match.get_matches(base_img, sec_img_cyl)
    H = homography.invoke_RANSAC(matches, base_img_kpts, sec_img_kpts)
    
    new_size, C, H = compute_newsize(H, sec_img_cyl.shape[:2], base_img.shape[:2])
    sec_img_new = warp.warp_img(sec_img_cyl, H, (new_size[0], new_size[1]))
    sec_img_new_mask = warp.warp_img(sec_img_mask, H, (new_size[0], new_size[1]))
    base_img_new = np.zeros((new_size[0], new_size[1], 3), dtype=np.uint8)
    base_img_new[C[1]:C[1]+base_img.shape[0], C[0]:C[0]+base_img.shape[1]] = base_img
    output_img = sec_img_new | (base_img_new & (~sec_img_new_mask))

    return output_img

