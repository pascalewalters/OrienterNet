import os
import torch
import matplotlib.pyplot as plt

from maploc.demo import ImageCalibrator
from maploc.utils.io import read_image
from maploc.data.image import rectify_image


IMAGE_DIR = "/home/pascale/Documents/VPS/Mappedin_VPS_Data/YYC_VPS/user_data/photos/"
OUTPUT_IMAGE_DIR = "/home/pascale/Documents/VPS/Mappedin_VPS_Data/YYC_VPS/user_data/photos_rectified/photos/"
OUTPUT_VALID_DIR = "/home/pascale/Documents/VPS/Mappedin_VPS_Data/YYC_VPS/user_data/photos_rectified/valid/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
calibrator = ImageCalibrator().to(device)

images = os.listdir(IMAGE_DIR)
for image_name in images:
    image_path = os.path.join(IMAGE_DIR, image_name)
    # print(image_path)

    # # Create a figure with two subplots side by side
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    image = read_image(image_path)

    gravity, camera = calibrator.run(image)
    # print(gravity)

    roll, pitch = gravity
    image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
    image, valid = rectify_image(
        image,
        camera.float(),
        roll=-roll,
        pitch=-pitch,
    )
    image_np = image.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
    valid_np = valid.mul(255).byte().cpu().numpy()

    # # Plot original image
    # ax1.imshow(image_np)
    # ax1.set_title("Original Image")
    # ax1.axis("off")  # Hide axes
    # # Plot rectified image
    # ax2.imshow(valid_np, cmap="gray")
    # ax2.set_title("Rectified Image")
    # ax2.axis("off")  # Hide axes

    # plt.tight_layout()  # Adjust spacing between subplots
    # plt.show()

    output_image_path = os.path.join(OUTPUT_IMAGE_DIR, image_name)
    plt.imsave(output_image_path, image_np)
    output_valid_path = os.path.join(OUTPUT_VALID_DIR, image_name)
    plt.imsave(output_valid_path, valid_np, cmap="gray")
