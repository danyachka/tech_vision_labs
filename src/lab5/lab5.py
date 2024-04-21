import src.lab5.utils as utils
import numpy as np
import cv2


def findShortestAndLongest(lines, tag="") -> tuple[int, int]:
    longest = 0
    longestId = 0

    shortest = float("inf")
    shortestId = 0

    for i in range(0, len(lines)):
        points: tuple = lines[i][0]

        length = ((points[0] - points[2])**2 + (points[1] - points[3])**2)**0.5

        if length > longest:
            longest = length
            longestId = i

        if length < shortest:
            shortest = length
            shortestId = i

    print(f"Самая длинная прямая ({tag}) = {longest}")
    print(f"Самая короткая прямая ({tag}) = {shortest}")
    print(f"Количество прямых ({tag}) = {len(lines)}")

    return shortestId, longestId


def drawLines(image, lines, tag="") -> np.ndarray:
    def drawLine(p, color):
        cv2.line(image, (p[0], p[1]), (p[2], p[3]), color, 5, cv2.LINE_AA)
        cv2.circle(image, (p[0], p[1]), radius=4, color=(0, 255, 0), thickness=-1)
        cv2.circle(image, (p[2], p[3]), radius=4, color=(0, 255, 0), thickness=-1)

    shortest, longest = findShortestAndLongest(lines, tag)

    for i in range(0, len(lines)):
        points: tuple = lines[i][0]

        drawLine(points, (0, 0, 255))


    # Нарисовать поверх
    points = lines[shortest][0]
    drawLine(points, (0, 255, 255))

    points = lines[longest][0]
    drawLine(points, (255, 0, 255))

    return image


def findLines(image, minLineLength) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 100, None, minLineLength, 20)

    image = drawLines(image, lines, "Обычный")

    return image, gray


def findLinesWithDerivative(image, minLineLength) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 100, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, minLineLength, 10)

    image = drawLines(image, lines, "с Кэнни")

    return image, edges


def findSmallestAndBiggest(circles, tag="") -> tuple[int, int]:
    smallest = float('inf')
    smallestId = -1

    biggest = 0
    biggestId = -1

    for i in range(0, len(circles)):
        _, _, radius = circles[0][i]

        if radius < smallest:
            smallest = radius
            smallestId = i

        if radius > biggest:
            biggest = radius
            biggestId = i

    print(f"Самая большая окружность ({tag}) = {biggest}")
    print(f"Самая маленькая окружность ({tag}) = {smallest}")
    print(f"Количество окружностей ({tag}) = {len(circles)}")

    return smallestId, biggestId


def findCircles(image, radiusRange, useCanny, tag):
    def drawCircle(x, y, r, color):
        print(f"x={x}, y={y}")
        x = int(x)
        y = int(y)
        r = int(r)
        cv2.circle(image, (x, y), radius=r, color=color, thickness=12)

    edges = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if useCanny:
        edges = cv2.Canny(edges, 100, 200, apertureSize=3)

    rows = edges.shape[0]
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=55, param2=40, minRadius=radiusRange[0], maxRadius=radiusRange[1])
    circles = np.uint16(np.around(circles))

    smallestId, biggestId = findSmallestAndBiggest(circles, tag)

    for i in range(len(circles)):
        x, y, radius = circles[0, i]

        if i == smallestId or i == biggestId:
            continue

        drawCircle(x, y, radius, (0, 0, 255))

    if smallestId != -1:
        x, y, radius = circles[0][smallestId]
        drawCircle(x, y, radius, (0, 255, 255))

    if biggestId != -1:
        x, y, radius = circles[0][biggestId]
        drawCircle(x, y, radius, (255, 0, 255))

    return image, edges


def firstTask():
    images = [utils.loadLinesImage()]

    for image in images:
        utils.showImage(image, "Оригинал")
        res, _ = findLines(image.copy(), 80)
        utils.showImage(res, "Результат без операторов")

        res1, edges1 = findLinesWithDerivative(image.copy(), 80)
        utils.showImage(res1, "Результат с Кэнни")
        utils.showImage(edges1, "Грани с Кэнни")


def secondTask():
    image = utils.loadCirclesImage()

    utils.showImage(image, "Оригинал")
    res, _ = findCircles(image.copy(), (50, 180), False, "без операторов")
    utils.showImage(res, "Результат без операторов")

    res1, edges1 = findCircles(image.copy(), (50, 180), True, "с Кэнни")
    utils.showImage(res1, "Результат с Кэнни")
    utils.showImage(edges1, "Грани с Кэнни")


def main():
    secondTask()
