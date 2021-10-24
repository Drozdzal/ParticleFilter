from line import Line
from shape import Shape
import cv2
import numpy as np


def dilate(img, width):
    kernel = np.ones((width, width), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def detect_circle(img: np.array, camera_transformation: np.array, f: int, circle_size):
    width, height = img.shape
    control_shape = Shape(lines=[
        Line([1, 1, 0, 1], [1, -1, 0, 1], (255, 255, 255)),
        Line([1, -1, 0, 1], [-1, -1, 0, 1], (255, 255, 255)),
        Line([-1, -1, 0, 1], [-1, 1, 0, 1], (255, 255, 255)),
        Line([-1, 1, 0, 1], [1, 1, 0, 1], (255, 255, 255)),
    ])
    # camera_direction =



def prepare_observation(observation_shape: Shape, target_res=(200, 200)):
    img = np.zeros((target_res[0], target_res[1], 3), dtype=np.uint8)
    observation_shape.draw(img, dir2color=True)
    return img


def blur(img, size):
    return cv2.GaussianBlur(img, size, cv2.BORDER_DEFAULT)


def prepare_observation2(observation_image: Shape, target_res=(200, 200)):
    lower_lines = np.array([0, 0, 208])  # od 8 juz opornie
    upper_lines = np.array([179, 255, 255])

    green_lower = np.array([50, 122, 104])
    green_upper = np.array([70, 162, 255])
    # return observation_image
    imgHSV = cv2.cvtColor(observation_image, cv2.COLOR_BGR2HSV)  # do hsv
    green_mask = cv2.inRange(imgHSV, green_lower, green_upper)
    kernel = np.ones((15, 15), np.uint8)
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)
    green_mask = cv2.erode(green_mask, kernel, iterations=1)
    gate_mask = cv2.inRange(imgHSV, lower_lines, upper_lines)
    gate_mask = cv2.bitwise_and(gate_mask, green_mask)
    edges = cv2.Canny(gate_mask, 150, 300)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi * 0.5 / 180, threshold=10, minLineLength=20,
                            maxLineGap=10)
    img = gate_mask * 0
    shape_lines = []
    for line in lines:
        [[x1, y1, x2, y2]] = line
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        shape_lines.append(Line([x1, y1], [x2, y2], (255, 255, 255)))
    shape = Shape(shape_lines)
    scale_transformation = np.eye(2) * target_res[0]/img.shape[0]
    shape.transform(scale_transformation)
    img2 = np.zeros((target_res[0], target_res[1], 3), dtype=np.uint8)
    shape.draw_raw(img2, True)
    img2 = blur(img2, (15, 15))
    img2 = img2 / img2.max() * 255
    img2 = np.array(img2, dtype=np.uint8)
    goal = detect_goal(observation_image)
    # return goal
    goal_mask = cv2.inRange(goal, 46, 999)

    kernel = np.ones((55, 55), np.uint8)
    goal_mask = cv2.dilate(goal_mask, kernel, iterations=4)
    goal_mask = cv2.erode(goal_mask, kernel, iterations=4)
    # return goal_mask
    goals_colored = np.zeros((target_res[0], target_res[1], 3), dtype=np.uint8)
    goals_colored[:, :, 2] = cv2.resize(goal_mask, target_res)
    return img2 + goals_colored


def detect_goal(image, kernel_size=5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = -np.ones((kernel_size, kernel_size), dtype=np.float32)
    positive_elements_count = 2*kernel_size - 1
    positive_value = (kernel_size ** 2 - positive_elements_count)/positive_elements_count
    for i in range(kernel_size):
        kernel[(kernel_size-1) // 2, i] = positive_value
        kernel[i, (kernel_size - 1) // 2] = positive_value
    return cv2.filter2D(image, -1, kernel/kernel_size/kernel_size)
