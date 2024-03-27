import numpy as np
import cv2 as cv


def createGaussianNoise(image, level=0.7):
    gauss = np.random.normal(0, level, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
    result = cv.add(image, gauss)

    return result


def contraharmonicMeanFilter(img, ks, QValue):
    def imageFilter(image, kernel_size, Q):
        r = kernel_size // 2
        result = np.zeros_like(image, dtype=np.float32)
        padded_image = cv.copyMakeBorder(image, r, r, r, r, cv.BORDER_REFLECT)

        for i in range(r, image.shape[0] + r):
            for j in range(r, image.shape[1] + r):
                window = padded_image[i - r:i + r + 1, j - r:j + r + 1].astype(np.float32)
                numerator = np.sum(np.power(window, Q + 1))
                denominator = np.sum(np.power(window, Q))
                result[i - r, j - r] = numerator / denominator if denominator != 0 else 0

        result = np.uint8(result)
        return result

    blue, green, red = cv.split(img)

    filtered_b = imageFilter(blue, ks, QValue)
    filtered_g = imageFilter(green, ks, QValue)
    filtered_r = imageFilter(red, ks, QValue)

    filtered_image = cv.merge([filtered_b, filtered_g, filtered_r])
    return filtered_image


def medianFilter(image, kernel_size):
    return cv.medianBlur(image, kernel_size)


def weightedMedianFilter(img, ks):
    def imageFilter(image, kernel_size):
        weights = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        result = np.zeros_like(image)
        padded_image = cv.copyMakeBorder(image, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2,
                                         cv.BORDER_REFLECT)

        for i in range(kernel_size // 2, image.shape[0] + kernel_size // 2):
            for j in range(kernel_size // 2, image.shape[1] + kernel_size // 2):
                window = padded_image[i - kernel_size // 2:i + kernel_size // 2 + 1,
                         j - kernel_size // 2:j + kernel_size // 2 + 1].astype(np.float32)
                window = np.multiply(window, weights)
                result[i - kernel_size // 2, j - kernel_size // 2] = np.median(window)

        return np.uint8(result)

    blue, green, red = cv.split(img)

    filtered_b = imageFilter(blue, ks)
    filtered_g = imageFilter(green, ks)
    filtered_r = imageFilter(red, ks)

    filtered_image = cv.merge([filtered_b, filtered_g, filtered_r])
    return filtered_image


def rankFilter(img, ks, r):
    def imageFilter(image, kernel_size, rank):
        result = np.zeros_like(image)
        padded_image = cv.copyMakeBorder(image, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2,
                                         cv.BORDER_REFLECT)

        for i in range(kernel_size // 2, image.shape[0] + kernel_size // 2):
            for j in range(kernel_size // 2, image.shape[1] + kernel_size // 2):
                window = padded_image[i - kernel_size // 2:i + kernel_size // 2 + 1,
                         j - kernel_size // 2:j + kernel_size // 2 + 1].astype(np.float32)
                result[i - kernel_size // 2, j - kernel_size // 2] = np.sort(window.flatten())[rank]

        return np.uint8(result)

    blue, green, red = cv.split(img)

    filtered_b = imageFilter(blue, ks, r)
    filtered_g = imageFilter(green, ks, r)
    filtered_r = imageFilter(red, ks, r)

    filtered_image = cv.merge([filtered_b, filtered_g, filtered_r])
    return filtered_image


def weinerFilter(img, ks, nv):
    def imageFilter(image, kernel_size, noise_var):
        noise_var = float(noise_var)
        result = np.zeros_like(image, dtype=np.float32)
        padded_image = cv.copyMakeBorder(image, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2,
                                         cv.BORDER_REFLECT)

        for i in range(kernel_size // 2, image.shape[0] + kernel_size // 2):
            for j in range(kernel_size // 2, image.shape[1] + kernel_size // 2):
                window = padded_image[i - kernel_size // 2:i + kernel_size // 2 + 1,
                         j - kernel_size // 2:j + kernel_size // 2 + 1].astype(np.float32)
                filter_weight = np.sum(np.square(window)) / (kernel_size ** 2 * noise_var + np.sum(np.square(window)))
                result[i - kernel_size // 2, j - kernel_size // 2] = filter_weight * np.mean(window)

        return np.uint8(result)

    blue, green, red = cv.split(img)

    filtered_b = imageFilter(blue, ks, nv)
    filtered_g = imageFilter(green, ks, nv)
    filtered_r = imageFilter(red, ks, nv)

    filtered_image = cv.merge([filtered_b, filtered_g, filtered_r])
    return filtered_image


def adaptiveMedianFilter(img, ks):
    def imageFilter(image, max_kernel_size):
        result = np.zeros_like(image, dtype=np.float32)
        padded_image = cv.copyMakeBorder(image, max_kernel_size // 2, max_kernel_size // 2, max_kernel_size // 2,
                                         max_kernel_size // 2, cv.BORDER_REFLECT)

        for i in range(max_kernel_size // 2, image.shape[0] + max_kernel_size // 2):
            for j in range(max_kernel_size // 2, image.shape[1] + max_kernel_size // 2):
                current_kernel_size = 3
                while current_kernel_size <= max_kernel_size:
                    window = padded_image[i - current_kernel_size // 2:i + current_kernel_size // 2 + 1,
                             j - current_kernel_size // 2:j + current_kernel_size // 2 + 1].astype(np.float32)
                    median_val = np.median(window)
                    min_val = np.min(window)
                    max_val = np.max(window)
                    A1 = median_val - min_val
                    A2 = median_val - max_val
                    if A1 > 0 and A2 < 0:
                        B1 = padded_image[i, j] - min_val
                        B2 = padded_image[i, j] - max_val
                        if B1.all() > 0 and B2.all() < 0:
                            result[i - max_kernel_size // 2, j - max_kernel_size // 2] = padded_image[i, j]
                            break
                    current_kernel_size += 2
                if current_kernel_size > max_kernel_size:
                    result[i - max_kernel_size // 2, j - max_kernel_size // 2] = median_val

        return np.uint8(result)

    blue, green, red = cv.split(img)

    filtered_b = imageFilter(blue, ks)
    filtered_g = imageFilter(green, ks)
    filtered_r = imageFilter(red, ks)

    filtered_image = cv.merge([filtered_b, filtered_g, filtered_r])
    return filtered_image
