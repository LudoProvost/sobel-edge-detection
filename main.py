import numpy as np
from PIL import Image
from convolution import convolution

image = Image.open('images/engine.png').convert('L')

image_data = np.asarray(image)

kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

sobel_image_x = convolution(image_data, kernel_x)
sobel_image_y = convolution(image_data, kernel_y)

sobel_gradient_magnitude = np.sqrt(np.square(sobel_image_x) + np.square(sobel_image_y))
sobel_gradient_magnitude *= 255.0 / sobel_gradient_magnitude.max()

processed_image = Image.fromarray(sobel_gradient_magnitude).convert('L')

processed_image.save('processed_images/engine.png')

processed_image.show()

