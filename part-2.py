import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# 2a ---------------------------------------------------------------------------------------

def analyze_image(input_image_path, output_folder_path):
    """
    Analyze an image and save outputs including grayscale and color histograms,
    gradient maps, and gradient intensity.

    Parameters:
    input_image_path (str): Path to the input image.
    output_folder_path (str): Path to the folder where outputs will be saved.
    """
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not load image from {input_image_path}. Check the file path.")
        return
    
    # Determine if the image is grayscale or color
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if the color channels are all identical (which means the image is grayscale)
        is_color = not np.array_equal(image[:,:,0], image[:,:,1]) or not np.array_equal(image[:,:,1], image[:,:,2])
    else:
        is_color = False  # The image is grayscale if it has only one channel

    # Ensure the output directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # 1. Convert to grayscale and save
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if is_color else image
    cv2.imwrite(os.path.join(output_folder_path, 'grayimg.png'), gray_image)

    # 2. Plot and save grayscale histogram
    plt.figure()
    plt.hist(gray_image.ravel(), 256, [0, 256], color='black')
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_folder_path, 'gray_histogram.png'))
    plt.close()

    # 3-5. Plot and save color histograms if the image is color
    if is_color:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            plt.figure()
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.title(f'{col.upper()} Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Pixel Frequency')
            plt.savefig(os.path.join(output_folder_path, f'{col}_histogram.png'))
            plt.close()
    
    # 6. Compute and save x-gradient (Sobel filter)
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    cv2.imwrite(os.path.join(output_folder_path, 'gradient_x.png'), np.abs(gradient_x).astype(np.uint8))
    
    # 7. Compute and save y-gradient (Sobel filter)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    cv2.imwrite(os.path.join(output_folder_path, 'gradient_y.png'), np.abs(gradient_y).astype(np.uint8))
    
    # 8. Compute and save gradient intensity
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
    cv2.imwrite(os.path.join(output_folder_path, 'gradient_intensity.png'), np.abs(gradient_magnitude).astype(np.uint8))

    print(f"All outputs saved in {output_folder_path}")



# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the image
input_image_path = os.path.join(script_dir, 'DIP_803.png')

# Create a generic "outputs-a" folder within the same directory as the script
output_folder_path = os.path.join(script_dir, "outputs-a")

# Check if the "outputs-a" folder exists; if not, create it
if not os.path.isdir(output_folder_path):
    os.makedirs(output_folder_path)

# Call the function
analyze_image(input_image_path, output_folder_path)


# 2b ---------------------------------------------------------------------------------------

def first_op(input_image_path, output_folder_path):
    """
    Perform a geometric transformation (rotation) on the grayscale image using OpenCV.
    
    Parameters:
    input_image_path (str): Path to the input grayscale image.
    output_folder_path (str): Path to the folder where the output image will be saved.
    """

    # Load the image and check if it’s color or grayscale
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not load image from {input_image_path}")
        return
    
    # Define a fixed angle of rotation. The default angle chosen is 15 degrees counter-clockwise.
    rotation_angle = -15  # Rotating by 15 degrees counter-clockwise

    # Get image dimensions
    (h, w) = img.shape[:2]

    # Define the rotation center and get the rotation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Perform the rotation with bilinear interpolation
    img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Save the rotated image
    rotated_image_path = f"{output_folder_path}/img_firstop.png"
    cv2.imwrite(rotated_image_path, img_rotated)

    return rotated_image_path


def second_op(input_image_path, output_folder_path):
    """
    Crop the image to remove black rectangles (central crop).
    
    Parameters:
    input_image_path (str): Path to the input image.
    output_image_path (str): Path to the cropped output image.
    """

    # Load the image and check if it’s color or grayscale
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not load image from {input_image_path}")
        return
    
    # Get the image dimensions
    height, width = img.shape[:2]

    # Define the crop box (cropping 15% from right and left, and 20% from above and below)
    left = int(width * 0.15)
    upper = int(height * 0.20)
    right = int(width * 0.85)
    lower = int(height * 0.80)

    # Crop the image
    img_cropped = img[upper:lower, left:right]

    # Save the cropped image
    cropped_image_path = f"{output_folder_path}/img_secondop.png"
    cv2.imwrite(cropped_image_path, img_cropped)

    return cropped_image_path


# def third_op(input_image_path, output_folder_path):
#     """
#     Apply a basic sharpening filter to the image to enhance the edges and save it as 'img_thirdop.png'.
    
#     Parameters:
#     input_image_path (str): Path to the input image.
#     output_folder_path (str): Path to the folder where the output image will be saved.
#     """

#     # Load the image and check if it’s color or grayscale
#     img = cv2.imread(input_image_path)
#     if img is None:
#         print(f"Error: Could not load image from {input_image_path}")
#         return
    

#     # Define a basic sharpening kernel
#     sharpening_kernel = np.array([[-1, -1, -1], 
#                                   [-1, 9, -1], 
#                                   [-1, -1, -1]])

#     # Apply the sharpening filter
#     img_sharpened = cv2.filter2D(img, -1, sharpening_kernel)

#     # Save the sharpened image
#     sharpened_image_path = f"{output_folder_path}/img_thirdop.png"
#     cv2.imwrite(sharpened_image_path, img_sharpened)

#     return sharpened_image_path



def third_op(input_image_path, output_folder_path):
    """
    Apply a basic sharpening filter to the image to enhance the edges and save it as 'img_thirdop.png'.
    
    Parameters:
    input_image_path (str): Path to the input image.
    output_folder_path (str): Path to the folder where the output image will be saved.
    """

    # Load the image and check if it’s color or grayscale
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Define the sharpening kernel with 5 at the center
    sharpening_kernel = np.array([[ 0, -1,  0], 
                                  [-1,  5, -1], 
                                  [ 0, -1,  0]])

    # Apply the sharpening filter
    img_sharpened = cv2.filter2D(img, -1, sharpening_kernel)

    # Save the sharpened image
    sharpened_image_path = f"{output_folder_path}/img_thirdop.png"
    cv2.imwrite(sharpened_image_path, img_sharpened)

    return sharpened_image_path


# Create a generic "outputs-b" folder within the same directory as the script
output_folder_path = os.path.join(script_dir, "outputs-b")

# Check if the "outputs-b" folder exists; if not, create it
if not os.path.isdir(output_folder_path):
    os.makedirs(output_folder_path)

# Construct the path to "img_grayimg.png" inside the "outputs-a" folder
input_image_path = os.path.join(script_dir, "outputs-a", "grayimg.png")
# # First operation: Rotation
first_op(input_image_path, output_folder_path)

# Construct the path to "img_firstop.png" inside the "outputs-b" folder
input_image_path = os.path.join(script_dir, "outputs-b", "img_firstop.png")
# # Second operation: Cropping
second_op(input_image_path, output_folder_path)


# Construct the path to "img_secondop.png" inside the "outputs-b" folder
input_image_path = os.path.join(script_dir, "outputs-b", "img_secondop.png")
# # Third operation: Sharpening
third_op(input_image_path, output_folder_path)


 # 2c ---------------------------------------------------------------------------------------

# Create a generic "outputs-c" folder within the same directory as the script
output_folder_path = os.path.join(script_dir, "outputs-c")

# Check if the "outputs-c" folder exists; if not, create it
if not os.path.isdir(output_folder_path):
    os.makedirs(output_folder_path)

# Construct the path to "img_thirdop.png" inside the "outputs-b" folder
input_image_path = os.path.join(script_dir, "outputs-b", "img_thirdop.png")
analyze_image(input_image_path, output_folder_path)

# 2d ---------------------------------------------------------------------------------------

# Construct the full path to the image
input_image_path = os.path.join(script_dir, 'DIP_803.png')

# Create a generic "outputs-d" folder within the same directory as the script
output_folder_path = os.path.join(script_dir, "outputs-d")

# Check if the "outputs" folder exists; if not, create it
if not os.path.isdir(output_folder_path):
    os.makedirs(output_folder_path)

# # First operation: Rotation
first_op(input_image_path, output_folder_path)

# Construct the path to "img_firstop.png" inside the "outputs-d" folder
input_image_path = os.path.join(script_dir, "outputs-d", "img_firstop.png")
# # Second operation: Cropping
second_op(input_image_path, output_folder_path)


# Construct the path to "img_secondop.png" inside the "outputs-d" folder
input_image_path = os.path.join(script_dir, "outputs-d", "img_secondop.png")
# # Third operation: Sharpening
third_op(input_image_path, output_folder_path)

# 2e ---------------------------------------------------------------------------------------

# Create a generic "outputs-e" folder within the same directory as the script
output_folder_path = os.path.join(script_dir, "outputs-e")

# Check if the "outputs-e" folder exists; if not, create it
if not os.path.isdir(output_folder_path):
    os.makedirs(output_folder_path)

# Construct the full path to the image-
input_image_path = os.path.join(script_dir, 'my_image.png')

# # First operation: Rotation
first_op(input_image_path, output_folder_path)

# Construct the path to "img_firstop.png" inside the "outputs-e" folder
input_image_path = os.path.join(script_dir, "outputs-e", "img_firstop.png")
# # Second operation: Cropping
second_op(input_image_path, output_folder_path)


# Construct the path to "img_secondop.png" inside the "outputs-e" folder
input_image_path = os.path.join(script_dir, "outputs-e", "img_secondop.png")
# # Third operation: Sharpening
third_op(input_image_path, output_folder_path)
