from controller import Robot,Camera
from ImageProcessing import Vision
from Ruchy import Move
import cv2

robot = Robot()
timestep = int(robot.getBasicTimeStep())
camera=Vision(robot.getDevice("camera"), timestep)
ruchy=Move(robot)

while robot.step(timestep) != -1:
    img=camera.CleanImg()
    circles=camera.GetBall()
    lines=camera.GetLines()
    try:
        for j in circles[0, :]:
                cv2.circle(img, (j[0], j[1]), j[2], (0, 255, 0), 3)
        for line in lines:
                line = line[0]
                cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)
             
    except:
        print("czegos nie widzi narazie")            
    cv2.imshow("wsjo",img)    
    cv2.waitKey(200)
    