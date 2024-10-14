import cv2 
import numpy as np
import os
import matplotlib.pyplot as plt


def analyze_image(input_image_path: str, output_folder_path: str) -> None: 
    """
        Analyzes an input image and generates several outputs including grayscale conversion,
        histograms, and Sobel gradient calculations.

        Args:
        input_image_path (str): The path to the input image file.
        output_folder_path (str): The path to the folder where the analysis results will be saved.

        The function performs the following operations:
        1. Create a grayscale format (if not inputted alreadyas one)
        2. Create a grayscale histogram
        3. Create a RGB histograms (if the inputted image is RGB)
        4. Calculate gradients X, Y and intensity of the image
        
    """

    # Load the image without any changes
    img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)  
    if img is None:
        print(f"Error: Could not open or find the image.")
        return

    # Check if the image is grayscale or colored
    if len(img.shape) == 2:
        # print("The image is grayscale.")
        is_colored = False
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # print("The image is colored.")
        is_colored = True
    else:
        print("The image format is not recognized.")
        return


    if is_colored:
        # Convert the image to grayscale using cv2 if it's colored
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Explanation about that function is at the beginning of the Part 2 in the PDF file
        img_bgr = img
    else:
        # If the image is already grayscale, use the loaded image
        gray_image = img

    # Save the grayscale image
    cv2.imwrite(os.path.join(output_folder_path, 'grayimg.png'), gray_image)
        
    plt.clf()

    # Calculate the histogram of the image in grayscale format
    hist, bin_edges = np.histogram(gray_image, bins=np.arange(257))
    plt.figure()
    plt.bar(bin_edges[:-1], hist, width=1, edgecolor='black')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel Frequency')
    plt.title('Grayscale Histogram')

    # Save the histogram plot
    plt.savefig(os.path.join(output_folder_path, 'gray_histogram.png'))
    plt.close()

    if is_colored:
        # Split the BGR image into its red, green, and blue channels
        blue_channel, green_channel, red_channel = cv2.split(img_bgr)

        # Calculate and plot the histograms for each channel
        for channel, color, name in zip([blue_channel, green_channel, red_channel], ['blue', 'green', 'red'], ['Blue', 'Green', 'Red']):
            
            plt.clf()
            hist_channel, bin_edges_channel = np.histogram(channel, bins=np.arange(257))
            plt.figure()
            plt.bar(bin_edges_channel[:-1], hist_channel, width=1, color=color)
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Pixel Frequency')
            plt.title(f'Histogram of {name} Image')

            # Save the histogram plot
            plt.savefig(os.path.join(output_folder_path, f'{color}_histogram.png'))
            plt.close()

    # Calculate the gradient in the x direction using the Sobel operator
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3) # Explanation about that function is at the beginning of the Part 2 in the PDF file
    abs_gradient_x = cv2.convertScaleAbs(gradient_x) # Explanation about that function is at the beginning of the Part 2 in the PDF file
    cv2.imwrite(os.path.join(output_folder_path, 'gradient_x.png'), abs_gradient_x)

    # Calculate the gradient in the y direction using the Sobel operator
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    abs_gradient_y = cv2.convertScaleAbs(gradient_y)
    cv2.imwrite(os.path.join(output_folder_path, 'gradient_y.png'), abs_gradient_y)

    # Calculate the gradient magnitude (intensity)
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    abs_gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    cv2.imwrite(os.path.join(output_folder_path, 'gradient_intensity.png'), abs_gradient_magnitude)



def display_image(image: np.ndarray, window_name: str) -> str:
    """
    Displays an image in a resizable window with dimensions adjusted to fit 
    within a specified size range, maintaining the original aspect ratio.

    Args:
    image (np.ndarray): The input image to be displayed.
    window_name (str): The name of the display window.
    """

    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    # Set maximum and minimum size for the window
    max_size = 1300
    min_size = 800

    # Calculate the aspect ratio of the image
    aspect_ratio = img_width / img_height
    
    # Determine the new window width and height within the range of 800 to 1000
    if img_width > img_height:
        # For landscape or wider images
        new_width = min(max_size, img_width)
        new_height = int(new_width / aspect_ratio)
    else:
        # For portrait or taller images
        new_height = min(max_size, img_height)
        new_width = int(new_height * aspect_ratio)
    
    # Ensure neither width nor height goes below the minimum size
    if new_width < min_size:
        new_width = min_size
        new_height = int(new_width / aspect_ratio)
    if new_height < min_size:
        new_height = min_size
        new_width = int(new_height * aspect_ratio)
    
    # Create a resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Resize the window based on the calculated dimensions
    cv2.resizeWindow(window_name, new_width, new_height)
    
    # Display the original image in the resized window
    cv2.imshow(window_name, image)

    return window_name



def crop_image(image : np.ndarray, top_left = None, bottom_right = None):
    """
    Crops an image based on user-selected top-left and bottom-right coordinates or uses provided coordinates.

    Args:
    image (np.ndarray): The input image to be cropped.
    top_left (tuple, optional): Coordinates of the top-left corner for cropping (x, y). Defaults to None.
    bottom_right (tuple, optional): Coordinates of the bottom-right corner for cropping (x, y). Defaults to None.

    The function performs the following steps:
    1. If no coordinates are provided, displays the original image and allows the user to select 
       the top-left and bottom-right corners by clicking on the image.
    2. If coordinates are provided, uses them directly to crop the image.
    3. Crops the image according to the selected or provided coordinates.
    4. Returns the cropped image along with the coordinates of the selected top-left and bottom-right corners.
    """

    if top_left is None and bottom_right is None:

        window_name = display_image(image, "Original Image")

        # Mouse callback function for selecting points
        def select_points(event, x, y, flags, param):
            nonlocal top_left, bottom_right
            
            if event == cv2.EVENT_LBUTTONDOWN: # Explanation about that function is at the beginning of the Part 2 in the PDF file
                top_left = (x, y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                bottom_right = (x, y)
                cv2.destroyAllWindows()
        
        # Register the mouse callback
        cv2.setMouseCallback(window_name, select_points) # Explanation about that function is at the beginning of the Part 2 in the PDF file
        
        # Wait until the user selects the points
        while True:
            if top_left is not None and bottom_right is not None:
                break
            cv2.waitKey(1) # Explanation about that function is at the beginning of the Part 2 in the PDF file
    
    # else:
    #     print(f"Selected Top-left corner: {top_left}")
    #     print(f"Selected Bottom-right corner: {bottom_right}")


    # Crop the image
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # display_image(cropped_image, "Cropped Image")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    

    # Return the cropped image and the coordinates of the vertices
    return cropped_image, (top_left, bottom_right)


def first_op(input_image_path: str, output_folder_path: str) -> np.ndarray:
    """
    Loads an image from the specified path, performs a cropping operation using predefined or user-selected
    coordinates, and saves the cropped image to the specified output folder.

    Args:
    input_image_path (str): The file path to the input image.
    output_folder_path (str): The directory where the cropped image will be saved.
    """

    # Load the image
    img_bgr = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    
    if img_bgr is None:
        print(f"Error: Could not open or find the image.")
        return

    cropped_img, _ = crop_image(img_bgr, coordinates[0], coordinates[1])

    # Save the cropped image
    cv2.imwrite(os.path.join(output_folder_path, 'img_firstop.png'), cropped_img)
    
    return cropped_img



def unsharp_mask_single_channel(input_channel: np.ndarray, blur_strength=15, alpha=1.5):
    """
    Applies an unsharp mask to a single channel of an image to enhance its sharpness.

    Args:
    input_channel (np.ndarray): The input channel (grayscale or a single color channel) to be sharpened.
    blur_strength (int, optional): The strength of the Gaussian blur applied to create the mask. 
                                   Must be an odd number. Defaults to 15.
    alpha (float, optional): The scaling factor to control the sharpness. A higher value increases sharpness. 
                             Defaults to 1.5.
    """

    blurred_channel = cv2.GaussianBlur(input_channel, (blur_strength, blur_strength), 0) # Explanation about that function is at the beginning of the Part 2 in the PDF file
    sharpened_channel = cv2.addWeighted(input_channel, 1 + alpha, blurred_channel, -alpha, 0) # Explanation about that function is at the beginning of the Part 2 in the PDF file
    return sharpened_channel
    


def second_op(input_image_path: str, output_folder_path: str) -> np.ndarray:
    """
    Applies an unsharp mask to an image to enhance its sharpness, whether it is grayscale or RGB.

    Args:
    input_image_path (str): The file path to the input image.
    output_folder_path (str): The directory where the sharpened image will be saved.
    """

    img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not open or find the image.")
        return

    # Check if the image is grayscale or RGB
    if len(img.shape) == 2:  # Grayscale image
        sharpened_image = unsharp_mask_single_channel(img, blur_strength=5, alpha=1.5)
    else:  # RGB image
        # Apply unsharp mask to each channel (R, G, B) separately
        # Explanation about that function is at the beginning of the Part 2 in the PDF file
        channels = cv2.split(img)  # Split the image into R, G, B channels 
        sharpened_channels = []
        for channel in channels:
            sharpened_channels.append(unsharp_mask_single_channel(channel, blur_strength=5, alpha=1.5))
        # Explanation about that function is at the beginning of the Part 2 in the PDF file
        sharpened_image = cv2.merge(sharpened_channels)  # Merge the sharpened channels back together

    # Save the sharpened image
    output_image_path = os.path.join(output_folder_path, 'img_secondop.png')
    cv2.imwrite(output_image_path, sharpened_image)

    return sharpened_image



def apply_median_filter(input_image: np.ndarray, kernel_size=3):
    """
    Applies a median filter to an input image to reduce noise, handling both grayscale and RGB images.

    Args:
    input_image (np.ndarray): The input image, either in grayscale or RGB format.
    kernel_size (int, optional): The size of the kernel to be used for the median filter. 
                                 It must be an odd number. Defaults to 3.
    """
        
    if len(input_image.shape) == 2:  # Grayscale image
        denoised_image = cv2.medianBlur(input_image, kernel_size)
    else:  # RGB image
        channels = cv2.split(input_image)  # Split into R, G, B channels
        denoised_channels = []
        for channel in channels:
            denoised_channels.append(cv2.medianBlur(channel, kernel_size))  # Apply median filter to each channel
        denoised_image = cv2.merge(denoised_channels)  # Merge back the channels
    
    return denoised_image



def third_op(input_image_path: str, output_folder_path: str) -> np.ndarray:
    """
    Loads an image from the specified path, applies a median filter to reduce noise, and saves the result.

    Args:
    input_image_path (str): The file path to the input image.
    output_folder_path (str): The directory where the denoised image will be saved.
    """

    img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not open or find the image.")
        return

    denoised_image = apply_median_filter(img, kernel_size=3)

    # Save the denoised image
    output_image_path = os.path.join(output_folder_path, 'img_thirdop.png')
    cv2.imwrite(output_image_path, denoised_image)

    return denoised_image


def fix_image(image_to_fix_path: str, output_folder_path: str):
    """
    Performs a sequence of image operations to enhance and clean an image, saving the results of each step.

    Args:
    image_to_fix_path (str): The file path to the input image that needs fixing.
    output_folder_path (str): The directory where the results of each operation will be saved.
    """

    # Operation 1: Cropping the image
    img_firstop = first_op(image_to_fix_path, output_folder_path)

    # Operation 2: Applying Unsharp Mask filter to sharpen the image
    second_input_image_path = os.path.join(output_folder_path, 'img_firstop.png')
    img_secondop = second_op(second_input_image_path, output_folder_path)  # Unsharp Mask

    # Operation 3: Applying Median Filter to remove noise from the image
    third_input_image_path = os.path.join(output_folder_path, 'img_secondop.png')
    img_thirdop = third_op(third_input_image_path, output_folder_path)  # Median Filter

    return img_firstop, img_secondop, img_thirdop



def main():

    # Get current file location
    script_dir = os.path.dirname(__file__)

    # Create global variable for coordinates 
    global coordinates

    # Part A -----------------------------------------------------------------------------------------------
    # Initialize paths and namings
    image_path = os.path.join(script_dir, 'DIP_811.png')
    part_a_folder_name = "part-a"

    # Create the dedicated folder if missing
    output_folder_path = os.path.join(script_dir, 'output-images', part_a_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    analyze_image(image_path, output_folder_path)


    # Part B -----------------------------------------------------------------------------------------------
    # Initialize paths and namings
    image_to_fix_path = os.path.join(script_dir, 'output-images', part_a_folder_name, 'grayimg.png')
    part_b_folder_name = "part-b"

    # Create the dedicated folder if missing
    output_folder_path = os.path.join(script_dir, 'output-images' , part_b_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Set cropping coordinates
    coordinates = ((900, 160), (2460, 1650))

    # Fix the image
    img_firstop, img_secondop, img_thirdop = fix_image(image_to_fix_path, output_folder_path)


    # Part C -----------------------------------------------------------------------------------------------
    # Initialize paths and namings
    image_path = os.path.join(script_dir, 'output-images', part_b_folder_name, 'img_thirdop.png')
    part_c_folder_name = "part-c"

    # Create the dedicated folder if missing
    output_folder_path = os.path.join(script_dir, 'output-images', part_c_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    analyze_image(image_path, output_folder_path)


    # Part D -----------------------------------------------------------------------------------------------
    # Initialize paths and namings
    image_to_fix_path = os.path.join(script_dir, 'DIP_811.png')
    part_d_folder_name = "part-d"

    # Create the dedicated folder if missing
    output_folder_path = os.path.join(script_dir, 'output-images' , part_d_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Set cropping coordinates
    coordinates = ((1000, 175), (2530, 1660))

    # Fix the image
    img_firstop, img_secondop, img_thirdop = fix_image(image_to_fix_path, output_folder_path)


    # Part E -----------------------------------------------------------------------------------------------
    # Initialize paths and namings
    image_to_fix_path = os.path.join(script_dir, 'IMG_811.png')
    part_e_folder_name = "part-e"

    # Create the dedicated folder if missing
    output_folder_path = os.path.join(script_dir, 'output-images' , part_e_folder_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Set cropping coordinates
    coordinates = ((870, 1130), (2300, 3090))   

    # Fix the image
    img_firstop, img_secondop, img_thirdop = fix_image(image_to_fix_path, output_folder_path)


if __name__ == "__main__":
    main()