import cv2
import numpy as np
class metamorphic_relations:
    def rotate_90_clockwise(image, boxes):
        import cv2
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        new_boxes = []
        for box in boxes:
            # Unpack first to define 'y'
            c, x, y, w, h = box
            new_x = 1.0 - y
            new_y = x
            new_w = h
            new_h = w
            new_boxes.append([c, new_x, new_y, new_w, new_h])
        return rotated_image, new_boxes

    def horizontal_mirror(image, boxes):
        transformed_image = cv2.flip(image, 1)
        transformed_boxes = []
        for box in boxes:
            class_id, x, y, bw, bh = box
            new_x = 1 - x
            transformed_boxes.append([class_id, new_x, y, bw, bh])
        return transformed_image, transformed_boxes

    def vertical_mirror(image, boxes):
        transformed_image = cv2.flip(image, 0)
        transformed_boxes = []
        for box in boxes:
            class_id, x, y, bw, bh = box
            new_y = 1 - y
            transformed_boxes.append([class_id, x, new_y, bw, bh])
        return transformed_image, transformed_boxes


    def color_inversion(image, boxes):
        transformed_image = cv2.bitwise_not(image)
        return transformed_image, boxes
    
    def noise_addition_gaussian(image, boxes, sigma=15):
        # Create floating point noise to avoid uint8 wrapping/clipping issues during generation
        noise = np.random.normal(0, sigma, image.shape)
        # Add noise and clip to valid 0-255 range
        transformed_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return transformed_image, boxes
    
    def gamma_correction(image, boxes, gamma=1.5):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        transformed_image = cv2.LUT(image, table)
        return transformed_image, boxes

    def noise_addition_salt_and_pepper(image, boxes, amount=0.05):
        transformed_image = image.copy()
        num_pixels = image.shape[0] * image.shape[1]
        
        num_salt = int(num_pixels * amount / 2)
        num_pepper = int(num_pixels * amount / 2)
        
        salt_y = np.random.randint(0, image.shape[0], num_salt)
        salt_x = np.random.randint(0, image.shape[1], num_salt)
        transformed_image[salt_y, salt_x] = 255
        
        pepper_y = np.random.randint(0, image.shape[0], num_pepper)
        pepper_x = np.random.randint(0, image.shape[1], num_pepper)
        transformed_image[pepper_y, pepper_x] = 0
        
        return transformed_image, boxes

    def brightness_adjustment(image, boxes, beta=50):
        transformed_image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        return transformed_image, boxes

    def contrast_adjustment(image, boxes, alpha=1.5):
        transformed_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        return transformed_image, boxes


    def blur(image, boxes, kernel_size=5):
        if kernel_size % 2 == 0:
            kernel_size += 1
        transformed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return transformed_image, boxes

    def rotations(image, boxes):
        return metamorphic_relations.rotate_90_clockwise(image, boxes)