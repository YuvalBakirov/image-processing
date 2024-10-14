import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.patches as patches


def calculate_patch_confidence(confidence, x, y, window_size):
    """
    Calculates the average confidence for a patch (window) centered at a given pixel in the confidence map.

    Args:
    - confidence (array-like): A 2D array representing the confidence map of the image.
    - x (int): The x-coordinate (column) of the center of the patch.
    - y (int): The y-coordinate (row) of the center of the patch.
    - window_size (int): The size of the square patch (window) centered at (x, y).
    """

    half_w = window_size // 2
    patch = confidence[max(0, x - half_w):x + half_w + 1, max(0, y - half_w):y + half_w + 1]
    return np.mean(patch)


def calculate_rmse(patch1, patch2):
    """
    Calculates the Root Mean Square Error (RMSE) between two image patches, considering only valid pixels.

    Args:
    - patch1 (array-like): The first image patch (2D array).
    - patch2 (array-like): The second image patch (2D array) to compare against the first patch.
    """
        
    valid_mask = (patch1 >= 0)  # Only compare known pixels
    if np.sum(valid_mask) == 0:
        return np.inf  # If there are no valid pixels, return a large error
    return np.sqrt(np.mean((patch1[valid_mask] - patch2[valid_mask])**2))



def find_best_match(image, selected_patch, window_size):
    """
    Finds the location of the best matching patch in the image based on the lowest RMSE (Root Mean Square Error).

    Args:
    - image (array-like): The input image (2D array) in which to search for the best matching patch.
    - selected_patch (array-like): The patch (2D array) to compare with patches in the image.
    - window_size (int): The size of the square patch (window) used for the search.
    """

    best_rmse = np.inf
    best_match_location = (-1, -1)
    half_w = window_size // 2
    image_h, image_w = image.shape

    # Search for patches in regions with known pixels
    for i in range(half_w, image_h - half_w):
        for j in range(half_w, image_w - half_w):
            patch = image[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1]
            if np.any(patch == -1):  # Skip patches with missing pixels
                continue
            rmse = calculate_rmse(selected_patch, patch)
            if rmse < best_rmse:
                best_rmse = rmse
                best_match_location = (i, j)

    return best_match_location


def display_matrix(pixel_array, title, with_ticks=False, with_values=False, vmin=0, vmax=255, fontsize=7):
    """
    Displays a 2D matrix as a grayscale image with optional tick marks and values for each pixel.

    Args:
    - pixel_array (array-like): The 2D array representing the pixel values of the image to be displayed.
    - title (str): The title for the image plot.
    - with_ticks (bool, optional): If True, displays x and y axis ticks and labels. Default is False.
    - with_values (bool, optional): If True, displays the pixel values within each cell of the matrix. Default is False.
    - vmin (int, optional): The minimum value for color scaling in the plot. Default is 0.
    - vmax (int, optional): The maximum value for color scaling in the plot. Default is 255.
    - fontsize (int, optional): Font size for pixel values and tick labels. Default is 7.
    """
        
    plt.imshow(pixel_array, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title(title)

    plt.colorbar()

    if with_values:
        # Annotate each cell with the confidence value
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                value = pixel_array[i, j]
                # If the value is an integer, display it without decimals, otherwise display with two decimal places
                if value.is_integer():
                    plt.text(j, i, f'{int(value)}', ha='center', va='center', fontsize=fontsize, color=(1, 0.4, 0.4, 0.8))
                else:
                    plt.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=fontsize, color=(1, 0.4, 0.4, 0.8))

    if with_ticks:
        # Set ticks at each cell and label them with (0, 1, 2, 3, ...)
        plt.xticks(ticks=range(pixel_array.shape[1]), labels=range(pixel_array.shape[1]), fontsize=fontsize)
        plt.yticks(ticks=range(pixel_array.shape[0]), labels=range(pixel_array.shape[0]), fontsize=fontsize)
    else:
        plt.xticks([])
        plt.yticks([])

    plt.show()


def patch_based_inpainting(original_image, window_size):
    """
    Performs patch-based inpainting on an image to fill missing pixels by finding and copying similar patches.

    Args:
    - original_image (array-like): The input image (2D array) with missing pixels marked as -1.
    - window_size (int): The size of the square patch (window) used for the inpainting process.
    """

    image = original_image.copy()

    # Initialize confidence matrix (1 for known pixels, 0 for missing pixels)
    confidence = np.where(image >= 0, 1.0, 0.0)

    half_w = window_size // 2
    missing_pixels = np.argwhere(image == -1)  # Find the missing pixels

    iteration_number = 1

    while len(missing_pixels) > 0:
        boundary_pixels = []

        # Find boundary pixels (missing pixels adjacent to known pixels)
        for (x, y) in missing_pixels:
            if np.any(confidence[max(0, x-half_w):x+half_w+1, max(0, y-half_w):y+half_w+1] > 0):
                boundary_pixels.append((x, y))

        if not boundary_pixels:
            break  # No boundary pixels to fill

        # Calculate the confidence for each of the boundry pixels
        confidence = np.where(image >= 0, 1.0, 0.0)
        previous_confidence = confidence.copy()
        for (i,j) in boundary_pixels:
            # Update confidence for newly filled pixel
            confidence[i, j] = calculate_patch_confidence(previous_confidence, i, j, window_size)

        display_matrix(confidence, f"Confidence Matrix - Iteration No{iteration_number}", with_ticks=True, with_values=True, vmin=0, vmax=1)

        # Find the boundary pixel with the highest confidence
        max_confidence = -np.inf
        selected_pixel = (-1,-1)
        for (i,j) in boundary_pixels:
            if confidence[(i,j)] > max_confidence:
                selected_pixel = (i,j)
                max_confidence = confidence[(i,j)]

        # Extract the patch around the selected pixel
        selected_patch = image[selected_pixel[0] - half_w:selected_pixel[0] + half_w + 1,
                               selected_pixel[1] - half_w:selected_pixel[1] + half_w + 1]
        
        # Find the best matching patch in the known region
        best_match_location = find_best_match(image, selected_patch, window_size)

        # Get the best matching patch
        best_patch = image[best_match_location[0] - half_w:best_match_location[0] + half_w + 1,
                           best_match_location[1] - half_w:best_match_location[1] + half_w + 1]
        
        # display_patch(selected_patch, "patch")       
        # display_patch(best_patch, "best patch")

        display_image(image, f"Iteration No{iteration_number} - Before Update", window_size, best_match_location, selected_pixel)
        display_image(image, f"Iteration No{iteration_number} - Before Update", window_size, best_match_location, selected_pixel, with_intensity=True)

        # Fill in the missing pixels in the selected patch
        missing_mask_in_patch = (selected_patch == -1)
        selected_patch[missing_mask_in_patch] = best_patch[missing_mask_in_patch]

        # Update the confidence matrix (1 in each filled cell)
        confidence[selected_pixel[0] - half_w:selected_pixel[0] + half_w + 1,
              selected_pixel[1] - half_w:selected_pixel[1] + half_w + 1] = 1

        # Update the image with the selected patch
        image[selected_pixel[0] - half_w:selected_pixel[0] + half_w + 1,
              selected_pixel[1] - half_w:selected_pixel[1] + half_w + 1] = selected_patch
        
        display_image(image, f"Iteration No{iteration_number} - After Update", window_size, best_match_location, selected_pixel)
        display_image(image, f"Iteration No{iteration_number} - After Update", window_size, best_match_location, selected_pixel,with_intensity=True)
        iteration_number+=1

        # Recompute the missing pixels
        missing_pixels = np.argwhere(image == -1)

    return image



def display_image(image, title, window_size=3, best_match_location=None, selected_pixel_location=None, with_intensity=False):
    """
    Displays an image with missing pixels highlighted and optional pixel intensity values.

    Args:
    - image (array-like): The input image (2D array) with pixel values, where missing pixels are marked as -1.
    - title (str): The title for the image plot.
    - window_size (int, optional): The size of the patch window used for matching (default is 3).
    - best_match_location (tuple, optional): The (row, column) location of the best matching patch. Default is None.
    - selected_pixel_location (tuple, optional): The (row, column) location of the selected pixel patch. Default is None.
    - with_intensity (bool, optional): If True, displays the pixel intensity values on the image. Default is False.
    """

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the grayscale image
    img = ax.imshow(image, cmap='gray', interpolation='nearest', vmin=0, vmax=255)

    if with_intensity:
        # Add pixel intensity values
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                ax.text(j, i, str(image[i, j]), ha='center', va='center', color=(1, 0.4, 0.4, 0.8), fontsize=7)

    # Set x and y ticks with axis labels
    ax.set_xticks(range(image.shape[1]))
    ax.set_yticks(range(image.shape[0]))
    ax.set_xticklabels(range(image.shape[1]), fontsize=8)
    ax.set_yticklabels(range(image.shape[0]), fontsize=8)


    # Overlay the missing pixels (-1) in red
    missing_mask = np.ma.masked_where(image != -1, image)  # Mask everything except -1
    ax.imshow(missing_mask, cmap='autumn', interpolation='nearest')  # 'autumn' is used to show red for missing

    if best_match_location is not None:

        half_w = window_size // 2

        best_match_rect = patches.Rectangle((best_match_location[1] - half_w - 0.5, best_match_location[0] - half_w - 0.5), window_size, window_size, linewidth=1, edgecolor='orange', facecolor='none')
        ax.add_patch(best_match_rect)

        selected_pixel_rect = patches.Rectangle((selected_pixel_location[1] - half_w - 0.5, selected_pixel_location[0] - half_w - 0.5), window_size, window_size, linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(selected_pixel_rect)

    # Add colorbar for the grayscale image
    cbar = fig.colorbar(img, ax=ax)
    # cbar.set_label('Pixel Intensity')

    # Add title
    ax.set_title(title)

    # Show the plot
    plt.show()



def plot_image_with_intensity(ax, image, title, fontsize=8, with_intensity=False, with_ticks=False):
    """
    Plots an image with optional pixel intensity values and axis ticks.

    Args:
    - ax (matplotlib.axes.Axes): The axis on which to plot the image.
    - image (array-like): The input image (2D array or grayscale) to be plotted.
    - title (str): The title for the image plot.
    - fontsize (int, optional): Font size for pixel intensity values and axis ticks. Default is 8.
    - with_intensity (bool, optional): If True, displays pixel intensity values inside each cell. Default is False.
    - with_ticks (bool, optional): If True, displays x and y axis ticks and labels. Default is False.
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

    # Set title
    ax.set_title(title)


def display_images_side_by_side(general_title, original_image, original_image_title, filtered_image, filtered_image_title, with_intensity=False, with_ticks=False):
    """
    Displays two images (original and filtered) side by side for comparison, with optional pixel intensity values and axis ticks.

    Args:
    - general_title (str): The general title for the entire figure.
    - original_image (array-like): The original image (2D array or grayscale) to be displayed.
    - original_image_title (str): The title for the original image subplot.
    - filtered_image (array-like): The filtered image (2D array or grayscale) to be displayed.
    - filtered_image_title (str): The title for the filtered image subplot.
    - with_intensity (bool, optional): If True, displays pixel intensity values inside each cell for both images. Default is False.
    - with_ticks (bool, optional): If True, displays x and y axis ticks and labels for both images. Default is False.
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Add a general title to the figure
    plt.suptitle(general_title, fontsize=13)

    # Plot original cropped image with a rectangle
    ax1 = axes[0]

    missing_mask = np.ma.masked_where(original_image != -1, original_image)
    plot_image_with_intensity(ax1, original_image, original_image_title, fontsize=5, with_intensity=with_intensity, with_ticks=with_ticks)
    ax1.imshow(missing_mask, cmap='autumn', interpolation='nearest')
    
    # Plot filtered image with pixel values
    ax2 = axes[1]
    plot_image_with_intensity(ax2, filtered_image, filtered_image_title, fontsize=5, with_intensity=with_intensity, with_ticks=with_ticks)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def main():

    # Define the 16x16 pixel intensity values from the image
    image = np.array([
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
            [240, 240, 45, 45, 45, -1, -1, -1, 70, 70, 70, 70, 70, 70, 45, 45],
            [240, 240, 45, 45, 45, -1, -1, -1, 70, 70, 70, 70, 70, 70, 45, 45],
            [240, 45, 45, 45, 45, 45, -1, -1, -1, -1, 45, 45, 45, 45, 45, 45],
            [45, 45, 45, 45, 45, 45, -1, -1, -1, -1, 240, 240, 240, 240, 240, 240],
            [45, 45, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240]
        ], dtype=np.int32)

  
    # Display results
    display_image(image, "Image with missing pixels")

    # Perform patch-based inpainting
    window_size = 3
    filled_image = patch_based_inpainting(image, window_size)

    display_images_side_by_side("Before and After Patch-Based Inpainting", image, 'Before', filled_image, 'After', with_ticks=True)
    display_image(image, "Initial Image", with_intensity=True)
    display_image(filled_image, "Final Image", with_intensity=True)
    

if __name__ == "__main__":
    main()