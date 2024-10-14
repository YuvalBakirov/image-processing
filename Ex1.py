
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.patches as patches

def create_histogram(image, title, bin_width = 5, xtick_fontsize = 7):
    """
    Creates and displays a histogram of pixel intensity values from an image.
    
    Args:
    - image (array-like): The input image (or pixel array) to calculate the histogram from.
    - title (str): The title of the histogram plot.
    - bin_width (int, optional): The width of each bar in the histogram. Default is 5.
    - xtick_fontsize (int, optional): Font size for the x-tick labels. Default is 7.
    """

    # Calculate the histogram of the image
    hist, bin_edges = np.histogram(image, bins=256, range=(0,255))

    # Plot the histogram
    plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0] + bin_width), edgecolor='black', color='gray')

    # Add frequency numbers above each column dynamically based on the height
    for i in range(len(hist)):
        if hist[i] > 0:
            # Dynamically place the red label above each column using the label_ratio
            plt.text(bin_edges[i], hist[i] + 1, str(hist[i]), ha='center', color='red', zorder=5, fontsize=6)

    # Only show tick labels for bins with non-zero values
    non_zero_bins = [bin_edges[i] for i in range(len(hist)) if hist[i] > 0]
    plt.xticks(non_zero_bins, labels=[f'{int(x)}' for x in non_zero_bins], rotation=45, fontsize=xtick_fontsize, color='black')
    plt.yticks(fontsize=8)

    # Add horizontal grid lines on y-axis ticks
    plt.grid(axis='y', linestyle='--', color='gray', linewidth=0.5)

    # Labels and title
    plt.title(title)
    plt.xlabel(f'Pixel Intensity', labelpad=15)
    plt.ylabel('Frequency')

    # Adjust the bottom margin for better readability
    plt.gcf().subplots_adjust(bottom=0.2)

    plt.show()


def display_image(pixel_array, title, fontsize=7):
    """
    Displays a grayscale image using a pixel array.

    Args:
    - pixel_array (array-like): The input 2D array representing pixel values of the image. 
                                Each value in the array corresponds to a pixel intensity (0 to 255).
    - title (str): The title of the image plot.
    - fontsize (int, optional): Font size for the x and y axis tick labels. Default is 7.
    """

    plt.imshow(pixel_array, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    plt.title(title)

    # Set ticks at each cell and label them with (0, 1, 2, 3, ...)
    plt.xticks(ticks=range(pixel_array.shape[1]), labels=range(pixel_array.shape[1]), fontsize=fontsize)
    plt.yticks(ticks=range(pixel_array.shape[0]), labels=range(pixel_array.shape[0]), fontsize=fontsize)

    plt.colorbar()
    plt.show()


def crop_image(pixel_array, start_x=5, start_y=3, end_x=12, end_y=10):
    """
    Crops a portion of the image based on the specified coordinates.

    Args:
    - pixel_array (array-like): The input 2D array representing the image (pixel intensities).
    - start_x (int, optional): The starting x-coordinate (column) for the crop. Default is 5.
    - start_y (int, optional): The starting y-coordinate (row) for the crop. Default is 3.
    - end_x (int, optional): The ending x-coordinate (column) for the crop. Default is 12.
    - end_y (int, optional): The ending y-coordinate (row) for the crop. Default is 10.
    """

    # Crop the image using the slicing technique
    cropped_image = pixel_array[start_y:end_y, start_x:end_x]
    
    # Convert the cropped image to uint8 (OpenCV requires uint8 format)
    cropped_image_uint8 = cropped_image.astype(np.uint8)
    return cropped_image_uint8


def apply_median_filter(image):
    """
    Applies a median filter to the given image to reduce noise.

    Args:
    - image (array-like): The input image (2D array or grayscale) to which the median filter will be applied.
    """

    # Apply median filter
    filtered_image = cv2.medianBlur(image, ksize=3) # Apply the filter

    return filtered_image


def apply_mean_filter(image):
    """
    Applies a mean (or averaging) filter to the given image to smooth it.

    Args:
    - image (array-like): The input image (2D array or grayscale) on which the mean filter will be applied.
    """

    # Apply mean filter
    kernel = np.ones((3, 3), np.float32) / 9
    filtered_image = cv2.filter2D(image, -1, kernel)  # Apply the filter

    return filtered_image


def plot_image_with_intensity(ax, image, title, fontsize=8, with_intensity=False, with_ticks=False):
    """
    Plots an image on a given axis and optionally displays pixel intensity values and axis ticks.

    Args:
    - ax (matplotlib.axes.Axes): The axis on which to plot the image.
    - image (array-like): The input image (2D array or grayscale) to be plotted.
    - title (str): The title for the image plot.
    - fontsize (int, optional): Font size for pixel intensity values and axis ticks. Default is 8.
    - with_intensity (bool, optional): If True, displays pixel intensity values on top of each pixel. Default is False.
    - with_ticks (bool, optional): If True, displays x and y axis ticks and labels. If False, removes the axes. Default is False.
    """

    ax.imshow(image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)

    if with_intensity:
        # Add pixel intensity values
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                ax.text(j, i, str(image[i, j]), ha='center', va='center', color=(1, 0.4, 0.4, 0.8), fontsize=fontsize)

    if with_ticks:
        # Set x and y ticks with axis labels
        ax.set_xticks(range(image.shape[1]))
        ax.set_yticks(range(image.shape[0]))
        ax.set_xticklabels(range(image.shape[1]), fontsize=fontsize)
        ax.set_yticklabels(range(image.shape[0]), fontsize=fontsize)
    
    else:
        ax.axis('off')

    # Set title and colorbar
    ax.set_title(title)
    

def display_images_side_by_side(general_title, original_image, filtered_image, filtered_image_title, with_rectangle=False, start_x=6, start_y=4, end_x=11, end_y=9, with_intensity=False, with_ticks=False):
    """
    Displays two images (original and filtered) side by side for comparison.

    Args:
    - general_title (str): The general title for the entire figure.
    - original_image (array-like): The original image (2D array or grayscale) to be displayed.
    - filtered_image (array-like): The filtered image (2D array or grayscale) to be displayed.
    - filtered_image_title (str): The title for the filtered image subplot.
    - with_rectangle (bool, optional): If True, draws a rectangle on both images. Default is False.
    - start_x (int, optional): The starting x-coordinate for the rectangle. Default is 6.
    - start_y (int, optional): The starting y-coordinate for the rectangle. Default is 4.
    - end_x (int, optional): The ending x-coordinate for the rectangle. Default is 11.
    - end_y (int, optional): The ending y-coordinate for the rectangle. Default is 9.
    - with_intensity (bool, optional): If True, displays pixel intensity values on the images. Default is False.
    - with_ticks (bool, optional): If True, displays x and y axis ticks and labels. If False, removes the axes. Default is False.
    """
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Add a general title to the figure
    plt.suptitle(general_title, fontsize=13)

    # Plot original cropped image with a rectangle
    ax1 = axes[0]
    if with_rectangle:
        rect = patches.Rectangle((start_x - 0.5, start_y - 0.5), end_x - start_x, end_y - start_y, linewidth=1, edgecolor='orange', facecolor='none')
        ax1.add_patch(rect)
    plot_image_with_intensity(ax1, original_image, 'Original Image', fontsize=5, with_intensity=with_intensity, with_ticks=with_ticks)

    # Plot filtered image with pixel values
    ax2 = axes[1]
    if with_rectangle:
        rect = patches.Rectangle((start_x - 0.5, start_y - 0.5), end_x - start_x, end_y - start_y, linewidth=1, edgecolor='orange', facecolor='none')
        ax2.add_patch(rect)
    plot_image_with_intensity(ax2, filtered_image, filtered_image_title, fontsize=5, with_intensity=with_intensity, with_ticks=with_ticks)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


# Function to display two images (original and filtered) side by side
def display_cropped_side_by_side(general_title, original_image, filtered_image, filtered_image_title, start_x, start_y, end_x, end_y):
    """
    Displays two cropped images (original and filtered) side by side, highlighting a region of interest.

    Args:
    - general_title (str): The general title for the entire figure.
    - original_image (array-like): The original image (2D array or grayscale) to be displayed.
    - filtered_image (array-like): The filtered image (2D array or grayscale) to be displayed.
    - filtered_image_title (str): The title for the filtered image subplot.
    - start_x (int): The starting x-coordinate of the cropping rectangle.
    - start_y (int): The starting y-coordinate of the cropping rectangle.
    - end_x (int): The ending x-coordinate of the cropping rectangle.
    - end_y (int): The ending y-coordinate of the cropping rectangle.
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Add a general title to the figure
    plt.suptitle(general_title, fontsize=13)

    # Plot original cropped image with a rectangle
    ax1 = axes[0]
    rect = patches.Rectangle((0.5, 0.5), end_x - start_x - 2, end_y - start_y - 2, linewidth=2, edgecolor='orange', facecolor='none')
    ax1.add_patch(rect)
    plot_image_with_intensity(ax1, original_image, 'Original Cropped Image', fontsize=8, with_intensity=True)

    # Plot filtered image with pixel values
    ax2 = axes[1]
    rect = patches.Rectangle((-0.5, -0.5), end_x - start_x - 2, end_y - start_y - 2, linewidth=5, edgecolor='orange', facecolor='none')
    ax2.add_patch(rect)
    plot_image_with_intensity(ax2, filtered_image, filtered_image_title, fontsize=8, with_intensity=True)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()



def main():
    
    # Define the 16x16 pixel intensity values from the image
    pixel_values = [
        [240, 240, 240, 70, 70, 70, 70, 70, 70, 240, 240, 210, 122, 122, 122, 240],
        [240, 240, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 122, 122, 240],
        [240, 240, 45, 45, 45, 45, 210, 210, 0, 122, 255, 70, 70, 70, 70, 240],
        [240, 0, 210, 70, 45, 210, 210, 210, 0, 210, 210, 210, 122, 70, 70, 240],
        [240, 0, 210, 70, 45, 45, 210, 210, 0, 70, 210, 210, 210, 122, 70, 240],
        [240, 0, 210, 70, 45, 45, 210, 210, 210, 0, 210, 210, 210, 210, 70, 240],
        [240, 0, 0, 122, 210, 210, 210, 210, 0, 0, 0, 0, 0, 70, 240, 240],
        [240, 240, 240, 240, 210, 210, 210, 210, 210, 210, 210, 70, 70, 70, 240, 240],
        [70, 70, 70, 70, 70, 45, 70, 70, 70, 45, 70, 70, 70, 240, 240, 45],
        [70, 70, 70, 70, 70, 70, 45, 70, 70, 70, 45, 240, 240, 240, 45, 45],
        [240, 70, 70, 70, 70, 70, 45, 45, 45, 45, 240, 70, 70, 70, 45, 45],
        [240, 240, 45, 45, 45, 45, 45, 255, 70, 70, 70, 70, 70, 70, 45, 45],
        [240, 240, 45, 45, 45, 45, 45, 255, 70, 70, 70, 70, 70, 70, 45, 45],
        [240, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45],
        [45, 45, 45, 45, 45, 45, 45, 45, 45, 240, 240, 240, 240, 240, 240, 240],
        [45, 45, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240]
    ]

    # Convert to NumPy array
    pixel_array = np.array(pixel_values)

    
    # Part A -----------------------------------------------------------------------------------------------
    # Display the original image
    display_image(pixel_array, 'The Original Image')

    # Create a histogram of the pixel values
    create_histogram(pixel_array, 'Part A - Histogram of Grayscale Image')


    # Part B -----------------------------------------------------------------------------------------------
    # Crop the image
    cropped_image = crop_image(pixel_array, 5, 3, 12, 10)

    # Apply median filter
    filtered_image = apply_median_filter(cropped_image)

    # Display the original cropped image and filtered image side by side
    display_cropped_side_by_side('Part B - Median Filter', cropped_image, filtered_image[1:-1, 1:-1], 'After Median Filter Image', 5, 3, 12, 10)
    
    updated_full_image = np.copy(pixel_array)
    updated_full_image[4:9, 6:11] = filtered_image[1:-1, 1:-1]  # Update with the filtered portion
    display_image(updated_full_image, 'Original Image After Median Filter on (4,6) to (9,11)')

    create_histogram(updated_full_image, 'Part B - Histogram of Grayscale Image after Median Filter')

    display_images_side_by_side('Part B - Median Filter', pixel_array, updated_full_image, 'After Median Filter Image', with_rectangle=False, with_intensity=False, with_ticks=True)
    display_images_side_by_side('Part B - Median Filter', pixel_array, updated_full_image, 'After Median Filter Image', with_rectangle=True, with_intensity=True, with_ticks=True)
    

    # Part C -----------------------------------------------------------------------------------------------
    # # Apply mean filter to the cropped image
    filtered_image = apply_mean_filter(cropped_image)

    # Display the original cropped image and filtered image side by side
    display_cropped_side_by_side('Part C - Mean Filter', cropped_image, filtered_image[1:-1, 1:-1], 'After Mean Filter Image', 5, 3, 12, 10)

    updated_full_image = np.copy(pixel_array)
    updated_full_image[4:9, 6:11] = filtered_image[1:-1, 1:-1]  # Update with the filtered portion
    display_image(updated_full_image, 'Original Image After Mean Filter on (4,6) to (9,11)')

    create_histogram(updated_full_image, 'Part C - Histogram of Grayscale Image after Mean Filter applied', 0.5, 6)

    display_images_side_by_side('Part C - Mean Filter', pixel_array, updated_full_image, 'After Mean Filter Image', with_rectangle=False, with_intensity=False, with_ticks=True)
    display_images_side_by_side('Part C - Mean Filter', pixel_array, updated_full_image, 'After Mean Filter Image', with_rectangle=True, with_intensity=True, with_ticks=True)
    

if __name__ == "__main__":
    main()
