import cv2
import numpy as np
from typing import Tuple


"""
classificationSpoof ~ end to end function

input
    img             : np.ndarray / iamge
    p_min (optional): float / threshold for classification
    
output (tuple)
    label           : str / prediction ('Spoof' or 'Real')
    p_value
"""


def classificationSpoof(img:np.ndarray, p_min:float = 0.1) -> Tuple[str, float]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = dog_filtering(img, k=2, sigma=0.9)
    img = fourier_transforms(img)
    threholded, p_value = pixelwise_threshding(img)

    label = 'Spoof' if isSpoof(p_value, p_min) else 'Real'

    return label, p_value


def fourier_transforms(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    spectrum = np.log(np.abs(fshift))
    spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
    spectrum = spectrum.astype(int)

    return spectrum


def dog_filtering(img, k, sigma):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma1 = sigma
    sigma2 = sigma * np.sqrt(k)

    blur1 = cv2.GaussianBlur(img, (5, 5), sigma1)
    blur2 = cv2.GaussianBlur(img, (5, 5), sigma2)

    dog = blur1 - blur2

    return dog



def pixelwise_threshding(dog_filtered_img):
    ERROR_CORRECTION = 2**4
    best_coef, best_threshold = _get_t_value(dog_filtered_img)

    result = np.where(dog_filtered_img <= best_threshold + ERROR_CORRECTION, 0, 1)

    p_value = np.sum(result) / (result.shape[0] * result.shape[1])

    return result, p_value


def _get_coef(threshold, prob_g, std_of_a, mean_of_a):
    LEFT_prob_sum, RIGHT_prob_sum = 0, 0
    LEFT_probxg_sum, RIGHT_probxg_sum = 0, 0

    for g, prob in prob_g.items():
        if g <= threshold:
            LEFT_prob_sum += prob
            LEFT_probxg_sum += prob * g

        else:
            RIGHT_prob_sum += prob
            RIGHT_probxg_sum += prob * g

    mu0 = LEFT_probxg_sum / LEFT_prob_sum
    mu1 = RIGHT_probxg_sum / RIGHT_prob_sum

    EB = (mu0 * LEFT_prob_sum) + (mu1 * RIGHT_prob_sum)
    EBB = ((mu0 ** 2) * LEFT_prob_sum) + ((mu1 ** 2) * RIGHT_prob_sum)
    EAB = (mu0 * LEFT_probxg_sum) + (mu1 * RIGHT_probxg_sum)

    EA = mean_of_a

    coef_numerator = EAB - (EA * EB)
    coef_denominator = std_of_a * np.sqrt(EBB - (EB ** 2))

    coef = coef_numerator / coef_denominator

    return coef


def _get_t_value(input_):
    num_of_pixel = input_.shape[0] * input_.shape[1]
    area_floor, area_ceil = np.min(input_), np.max(input_)

    std_of_a = np.std(input_)  # np.sqrt(EAA - (EA ** 2))
    mean_of_a = np.mean(input_)  # EA

    unique, counts = np.unique(input_, return_counts=True)
    prob_g = dict(zip(unique, counts / num_of_pixel))

    best_threshold, best_coef = 0, 0

    # for threshold in np.linspace(area_floor, area_ceil, num=50):
    for threshold in range(area_floor, area_ceil):

        # get_coef for each threshold
        try:
            coef = _get_coef(threshold=threshold,
                             prob_g=prob_g,
                             std_of_a=std_of_a,
                             mean_of_a=mean_of_a)

            if best_coef < coef:
                best_coef = coef
                best_threshold = threshold
            # print(f'threshold={threshold}, coef={coef}')

        except ZeroDivisionError:
            print(f'zero division except: {threshold}')
            pass

    return best_coef, best_threshold


def isSpoof(p, p_min=0.1):
    p_min = p_min

    if p >= p_min: return False  # Negative ~ Real Image
    else: return True  # Positive ~ Spoof Image


if __name__ == "__main__":
    image = cv2.imread('content/sample_original.jpeg')
    prediction, p = classificationSpoof(image, 0.1)

    print(f'prediction: {prediction}')
    print(f'p-value: {p}')
