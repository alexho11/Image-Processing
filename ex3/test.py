import cv2 as cv
import numpy as np

def hough_lines_accumulator(edges, rho_resolution=1, theta_resolution=1):
    height, width = edges.shape
    max_rho = int(np.sqrt(height**2 + width**2))
    rhos = np.arange(-max_rho, max_rho, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)
    y_idxs, x_idxs = np.nonzero(edges)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)):
            rho = int((x * np.cos(thetas[j]) + y * np.sin(thetas[j])) / rho_resolution) + max_rho
            accumulator[rho, j] += 1

    return accumulator, thetas, rhos

def hough_peaks(accumulator, num_peaks, threshold=100):
    peaks = []
    for _ in range(num_peaks):
        idx = np.argmax(accumulator)
        rho_idx, theta_idx = np.unravel_index(idx, accumulator.shape)
        if accumulator[rho_idx, theta_idx] > threshold:
            peaks.append((rho_idx, theta_idx))
            accumulator[rho_idx, theta_idx] = 0
        else:
            break
    return peaks

# Load image, convert to grayscale, and perform edge detection
img = cv.imread('images/building.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# Hough Transform
accumulator, thetas, rhos = hough_lines_accumulator(edges)
peaks = hough_peaks(accumulator, num_peaks=10, threshold=100)

# Draw lines
for peak in peaks:
    rho = rhos[peak[0]]
    theta = thetas[peak[1]]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the result
cv.imwrite('houghlines_custom.jpg', img)
