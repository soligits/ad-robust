import cv2
import numpy as np
from scipy.signal import convolve2d


def calculate_err_ms(
    input_image, reconstructed_image, s=3, scale_scheme=[1, 0.5, 0.25, 0.125]
):
    err_ms = np.zeros(input_image.shape[:2])
    for scale in scale_scheme:
        resized_input = cv2.resize(input_image, (0, 0), fx=scale, fy=scale)
        resized_reconstruction = cv2.resize(
            reconstructed_image, (0, 0), fx=scale, fy=scale
        )
        err = cv2.resize(
            np.mean(np.square(resized_input - resized_reconstruction), axis=2),
            (0, 0),
            fx=np.round(1 / scale),
            fy=np.round(1 / scale),
        )
        err_ms += err
    err_ms /= len(scale_scheme)
    mean_filter = np.ones((s, s)) / (s * s)
    return convolve2d(err_ms, mean_filter, mode="same")


def anomaly_score(
    test_image,
    reconstructed_image,
    training_err_ms,
    s=3,
    scale_scheme=[1, 1 / 2, 1 / 4, 1 / 8],
):
    err_ms = calculate_err_ms(test_image, reconstructed_image, s, scale_scheme)
    return np.max(abs(err_ms - training_err_ms))


def reconstruct_image(model, test_image, k, reconstruction_type):
    if reconstruction_type == "one_shot":
        return model.reconstruct(test_image, k)
    elif reconstruction_type == "iterative":
        return model.reconstruct_iterative(test_image, k)
    else:
        raise ValueError("Unknown reconstruction type: {}".format(reconstruction_type))
    

