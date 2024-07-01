import sys
from stitcher import stitch
import os
import cv2
import processvideo
import matplotlib.pyplot as plt

def part1(img_dir, output_path):
    image_paths = [os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith(('.jpg', '.png'))]
    images = [cv2.imread(path) for path in image_paths]
    panorama=images[0]
    for i in range(1,len(images)):
        panorama=stitch(panorama,images[i])
        # plt.imshow(cv2.cvtColor(panorama.astype("uint8"), cv2.COLOR_BGR2RGB))
        # plt.show()     
    cv2.imwrite(output_path, panorama)

def part2(video_path, output_path):
    images=processvideo.capture_frames_at_intervals(video_path)
    panorama=images[0]
    for i in range(1,len(images)):
        panorama=stitch(panorama,images[i])
        # plt.imshow(cv2.cvtColor(panorama.astype("uint8"), cv2.COLOR_BGR2RGB))
        # plt.show()
    cv2.imwrite(output_path, panorama)

if __name__ == "__main__":
    part_id = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    if part_id == 1:
        part1(input_path, output_path)
    elif part_id == 2:
        part2(input_path, output_path)
    
