
__author__ = "Kushal Narasimha"
__date__ = "2024-02-20"

import cv2
import imutils

import os
import re
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Filter out redundant frames")
    parser.add_argument("--path", dest="path", required=True, help="The path to the image folder")
    parser.add_argument("--store_non_ess_frames", action="store_true", help="The flag to store non-essential frames")
    args = parser.parse_args()
    return args

def sort_images(directory):
    '''
    This function sorts the images by camera_id and timestamp and returns the dict.

    Parameters:
    directory (str): The path to folder containing image files 

    Returns:
    dict: Contains the sorted image files by camera_id and timestamp
    '''
    sort_files_by_camera_id = defaultdict(list)
    
    for filename in directory:
        if filename.endswith(".png"):
            # Extract the camera_id from the filename
            match = re.search(r'c(\d+)-', filename)
            if match:
                camera_id = match.group(1)
                sort_files_by_camera_id[camera_id].append(filename)
            else:
                match = re.search(r'c(\d+)_', filename)
                if match:
                    camera_id = match.group(1)
                    sort_files_by_camera_id[camera_id].append(filename)

    for camera_id, frames in sort_files_by_camera_id.items():
        sorted_frames = sorted(frames, key=lambda frame: get_timestamp(frame))
        sort_files_by_camera_id[camera_id] = sorted_frames

    return sort_files_by_camera_id


def get_timestamp(filename):
    '''
    This function search for the timestamp in the img file name.

    Parameters:
    filename (str): Complete image file name 

    Returns:
    timestamp (datetime): datetime.datetime format timestamp
    '''
    match = re.search(r'(\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2})', filename)
    if match:
        timestamp_str = match.group(1).replace('_', '')
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    else:
        match = re.search(r'(\d{13})', filename)
        if match:
            timestamp = datetime.fromtimestamp(int(match.group(1))/1000)
        else:
            raise ValueError(f"Invalid filename format: {filename}")
    return timestamp

def get_black_mask(cam_id):
    '''
    This function provides the black mask based on camer id.

    Parameters:
    cam_id (str): Camera id  

    Returns:
    tuple: Stating the percentage of black mask in all 4 corners
    '''

    val1 = val2 = val3 = val4 = 0

    if cam_id == '10':
        val2 = 13
        return (val1, val2, val3, val4)
    elif cam_id == '20':
        val2 = 28
        return (val1, val2, val3, val4)
    elif cam_id == '21':
        val2 = 30
        return (val1, val2, val3, val4)
    elif cam_id == '23':
        val2 = 32
        return (val1, val2, val3, val4)
    else:
        val2 = 0
        return (val1, val2, val3, val4)

def draw_color_mask(img, borders, color=(0, 0, 0)):
    '''
    This function draws color mask on the image.

    Parameters:
    img (MatLike): OpenCV format image 
    borders (tuple): Contains the percentage of black mask borders in all 4 corners 
    color (tuple): (R, G, B) default color black -> (0, 0, 0)

    Returns:
    img (MatLike): Masked OpenCV formate image 
    '''
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(0, 10, 0, 0)):
    '''
    This function converts image to gray, apply Gaussian Blur and color mask.

    Parameters:
    img (MatLike): OpenCV format image
    gaussian_blur_radius_list (list): blur kernel radius 
    black_mask (tuple): Contains the percentage of black mask borders in all 4 corners

    Returns:
    gray (MatLike): Grayscale image after blur and masking.
    '''
    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    '''
    This function provides black mask based on camer id.

    Parameters:
    prev_frame (MatLike): Previous frame image 
    next_frame (MatLike): Current frame image 
    min_contour_area (int): Minimum countour area 

    Returns:
    score: Total sum of contour area
    res_cnts: Resultant contours
    thresh: MatLike thresholded image
    '''
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []

    for c in cnts:
        if ((cv2.contourArea(c) < min_contour_area)):
            continue

        res_cnts.append(c)
        score += cv2.contourArea(c)

    return score, res_cnts, thresh


def main():
    args = parse_args()
    img_files_fold_path = os.listdir(args.path)
    sort_img_files_dict = sort_images(img_files_fold_path)
    total_frames = sum(len(v) for v in sort_img_files_dict.values())

    # Initialize variables
    counter = 0
    ess_frames = 0
    non_ess_frames = 0
    prev_img = None
    prev_img_shape = None
    min_contour_area = 1500  # To Fliter out coutour area less than 1500
    gauss_blur_rad_lst = [5] # Chose blur kernel radius 5 

    print("Process to filter out non-essential frames started!\n")
    with tqdm(total=total_frames) as pbar:
        for camera_id, img_files in sort_img_files_dict.items():
            for img_file in img_files:
                img_path = os.path.join(args.path, img_file)
                current_img = cv2.imread(img_path)
                if current_img is None:
                    counter += 1 # Count Damaged images 
                    continue

                current_img_shape = current_img.shape

                if prev_img is not None and (current_img_shape == prev_img_shape):
                    mask = get_black_mask(camera_id)
                    gray_current_img = preprocess_image_change_detection(current_img, gauss_blur_rad_lst, black_mask=mask)
                    gray_previous_img = preprocess_image_change_detection(prev_img, gauss_blur_rad_lst, black_mask=mask)
                    score, res_cnts, thresh = compare_frames_change_detection(gray_current_img, gray_previous_img, min_contour_area)

                    if (score > 0): 
                        output_dir = os.path.join(args.path, str(camera_id))
                        os.makedirs(output_dir, exist_ok=True)
                        output_dir = os.path.join(output_dir, img_file)
                        cv2.imwrite(output_dir, current_img)
                        ess_frames += 1

                    elif (args.store_non_ess_frames):
                        output_dir = os.path.join(args.path, "store_non_ess_frames")
                        os.makedirs(output_dir, exist_ok=True)
                        output_dir = os.path.join(output_dir, img_file)
                        cv2.imwrite(output_dir, current_img)
                        non_ess_frames += 1
                else:
                    # Control jumps here for initial frames and when current_img_shape != prev_img_shape
                    output_dir = os.path.join(args.path, str(camera_id))
                    os.makedirs(output_dir, exist_ok=True)
                    output_dir = os.path.join(output_dir, img_file)
                    cv2.imwrite(output_dir, current_img)
                    ess_frames += 1

                prev_img = current_img
                prev_img_shape = current_img.shape
                pbar.update(1)

    if (not args.store_non_ess_frames):
        non_ess_frames = total_frames - ess_frames - counter
    print(f'Total number of frames processed: {total_frames}\n')
    print(f'Total number of non-essential frames filtered out: {non_ess_frames}\n')
    print(f'Total number of essential frames: {ess_frames}\n')
    print(f'Total number of damaged frames: {counter}\n')
    print(f'The essential frames can be found inside their corresponding camera_id subfolder in the directory: {args.path}\n')


if __name__ == "__main__":
    main()
