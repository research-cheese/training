import os
import random

from segment_anything import SamPredictor, sam_model_registry
from cv2 import imread
from PIL import Image
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

SAM_MODEL_TYPE = "vit_b"
SAM_MODEL_CHECKPOINT = "model_checkpoint/sam_vit_b_01ec64.pth"

MY_DATASET_PATH = "data/abandoned_park/test"

WHITE_COLOR = [(255, 255, 255)]

def show_black_and_white_image(image):
    plt.pcolor(image, cmap='gray', vmin=0, vmax=1)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def filter_colors(image_path, colors):
    """
    Returns 2D numpy array of 0s and 1s
    """
    ground_truth_pil = Image.open(image_path).convert("RGB")
    
    # Resize to 256x144
    ground_truth_pil = ground_truth_pil.resize((256, 144), Image.NEAREST)

    ground_truth_pil = np.asarray(ground_truth_pil)

    # Iterate through ground truth pil and if color is in ground_truth_colors, set to 1, else 0.
    # It is a 2x2 image/
    ground_truth_pil = np.array([[
        1 if tuple(pixel) in colors else 0
        for pixel in row
    ] for row in ground_truth_pil])
    
    return ground_truth_pil

def all_surrounding(image, x, y):
    left = max(x - 1, 0)
    right = min(x + 1, image.shape[0] - 1)
    top = max(y - 1, 0)
    bottom = min(y + 1, image.shape[1] - 1)
    return image[x, y] == 1 and image[left, y] == 1 and image[right, y] == 1 and image[x, top] == 1 and image[x, bottom] == 1

def get_all_points_equal_1(image):
    points = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 1:
                points.append([i, j, image[i, j]])
    return points

def get_bbbox(ground_truth_map: np.array) -> list:
  """
  Get the bounding box of the image with the ground truth mask
  
    Arguments:
        ground_truth_map: Take ground truth mask in array format

    Return:
        bbox: Bounding box of the mask [X, Y, X, Y]

  """
  # get bounding box from mask
  idx = np.where(ground_truth_map > 0)
  x_indices = idx[1]
  y_indices = idx[0]
  try:
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = ground_truth_map.shape
    x_min = max(0, x_min)
    x_max = min(W, x_max)
    y_min = max(0, y_min)
    y_max = min(H, y_max)
    bbox = [x_min, y_min, x_max, y_max]

  except:
    bbox = [0, 0, 0, 0]

  return bbox

def get_bounding_box(image, variable):

    bbbox = get_bbbox(image)
    min_x = bbbox[0] 
    min_y = bbbox[1]
    max_x = bbbox[2]
    max_y = bbbox[3]

    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         if image[i, j] == 1:
    #             min_x = min(min_x, i)
    #             max_x = max(max_x, i)
    #             min_y = min(min_y, j)
    #             max_y = max(max_y, j)

    # random.random() * variable

    min_x = min_x + variable * (random.random()-0.5)
    max_x = max_x + variable * (random.random()-0.5)
    min_y = min_y + variable * (random.random()-0.5)
    max_y = max_y + variable * (random.random()-0.5)

    box = [
        min(min_x, max_x), 
        min(min_y, max_y), 
        max(max_x, min_x), 
        max(max_y, min_y)
    ]

    return np.array(box).astype(int)

def generate_points(image, num_points):
    points = get_all_points_equal_1(image)
    random.shuffle(points)
    temp = points[:min(num_points, len(points))]

    input_points = np.array([(
        point[0],
        point[1]
    ) for point in temp])

    input_labels = np.array([point[2] for point in temp])

    return input_points, input_labels

def create_array_of_ones_same_length(points: np.array):
    return np.ones(points.shape[0])

def merge_masks(masks):
    mask = np.zeros(masks[0].shape)
    for m in masks:
        mask = np.logical_or(mask, m)
    return mask

def add_masks(masks):
    mask = np.zeros(masks[0].shape)
    for m in masks:
        mask = np.add(mask, m)
    return mask

def random_points(num_points):
    return np.array([(
        int(random.random() * 256), 
        int(random.random() * 144)
    ) for _ in range(num_points)])

def logits_to_sgmd(logits):
    return 1 / (1 + np.exp(-logits))

def save_black_and_white_image(image, path):
    # Set black and white color map
    plt.pcolor(image, cmap='gray', vmin=0, vmax=1)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_imageage(image, path):
    # Set black and white color map
    plt.pcolor(image, cmap='gray', vmin=0, vmax=4)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

metadata = pd.read_json("metadata/abandoned_park/test.jsonl", lines=True)

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_CHECKPOINT)
sam = sam.to("cuda")
predictor = SamPredictor(sam)

sorted_dir = os.listdir(MY_DATASET_PATH)
sorted_dir = sorted([int(x) for x in sorted_dir])
sorted_dir = [str(x) for x in sorted_dir]
for sample in sorted_dir:
    for model in ["dust", "fog", "maple_leaf"]:
        for cls in ["ferris_wheel", "tree", "carousel", "roller_coaster"]:
            final_path = f"output/abandoned_park/{sample}/{model}/{cls}/"
            final_image_path = f"images/abandoned_park/{sample}/{model}/{cls}"
            
            print(f"Processing {final_path}")
            if os.path.exists(final_path): continue

            image = imread(os.path.join(MY_DATASET_PATH, sample, "Scene.png"))
            cls_image = filter_colors(os.path.join(MY_DATASET_PATH, sample, f"{cls}.png"), WHITE_COLOR)

            variable = 100
            if (float(metadata[metadata["sample"] == int(sample)][model]) > 0.01): variable = 0
            box = get_bounding_box(cls_image, variable)
            predictor.set_image(image)
            masks, _, _ = predictor.predict(box=box, return_logits=True)
            sgmd_masks = [logits_to_sgmd(m) for m in masks]
            sgmd_mask = add_masks(sgmd_masks)

            binary_mask = sgmd_mask > 0.5

            # show_black_and_white_image(cls_image)
            # show_black_and_white_image(binary_mask)

            os.makedirs(f"{final_path}")
            os.makedirs(f"{final_image_path}")

            np.save(f"{final_path}/sgmd.npy", sgmd_mask)
            np.save(f"{final_path}/binary.npy", binary_mask)
            np.save(f"{final_path}/ground.npy", cls_image)

            save_black_and_white_image(sgmd_mask, f"{final_image_path}/sgmd.png")
            save_black_and_white_image(binary_mask, f"{final_image_path}/binary.png")
            save_black_and_white_image(cls_image, f"{final_image_path}/ground.png")
        
        final_binary_mask = add_masks([np.load(f"output/abandoned_park/{sample}/{model}/{cls}/binary.npy") for cls in ["ferris_wheel", "tree", "carousel", "roller_coaster"]])
        np.save(f"output/abandoned_park/{sample}/{model}/combined_binary.npy", final_binary_mask)
        save_imageage(final_binary_mask, f"images/abandoned_park/{sample}/{model}/combined_binary.png")