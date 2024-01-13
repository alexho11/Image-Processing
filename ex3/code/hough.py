import numpy as np
import cv2 as cv
def hough_lines_accumulator(edges):
    # Implement a Hough transform that takes in an edge image and outputs a Hough
    height, width = edges.shape
    max_d = int(np.sqrt(height**2 + width**2))
    d = np.arange(-max_d, max_d, 1)
    alpha = np.deg2rad(np.arange(-90,90,1))

    accumulator = np.zeros((len(d), len(alpha)), dtype=int)
    y_idxs, x_idxs = np.nonzero(edges)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(alpha)):
            d = int(x * np.cos(alpha[j]) + y * np.sin(alpha[j]))  + max_d
            accumulator[d, j] += 1

    return accumulator, alpha, d

def hough_peaks(accumulator, num_peaks, threshold=100):
    # Sort the accumulator to find the maxima
    peaks = []
    for _ in range(num_peaks):
        idx = np.argmax(accumulator)
        d_idx, alpha_idx = np.unravel_index(idx, accumulator.shape)
        if accumulator[d_idx, alpha_idx] > threshold:
            peaks.append((d_idx, alpha_idx))
            accumulator[d_idx, alpha_idx] = 0
        else:
            break
    return peaks