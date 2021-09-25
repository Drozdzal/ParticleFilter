import cv2
import numpy as np

class Vision:
	def __init__(self,camera_device,timestep):
		self.camera=camera_device
		self.timestep=timestep
		self.camera.enable(timestep)

		# parametry do maski na boisko
		self.lower_green = np.array([44, 100, 0])
		self.upper_green = np.array([80, 255, 255])

	def CleanImg(self):
		img = self.camera.getImageArray()
		img = np.asarray(img, dtype=np.uint8)
		return img

	def MaskedImg(self):
		img = self.camera.getImageArray()
		img = np.asarray(img, dtype=np.uint8)
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # do gray
		field_mask = cv2.inRange(imgHSV, self.lower_green, self.upper_green)
		field_mask = cv2.GaussianBlur(field_mask, (11, 11), 0)
		return field_mask

	def GetBall(self):
		field_mask = self.MaskedImg()
		circles = cv2.HoughCircles(field_mask, cv2.HOUGH_GRADIENT, 1.5, 160, param1=80, param2=30, minRadius=0, maxRadius=30)
		try:
			for j in circles[0, :]:
				odleglosc = 0.05/((j[2]/160)*np.tan(0.56/2))
				print("Pilka jest w odleglosi "+str(2*odleglosc)+"m")
			
		except:
			print("Nie znaleziono pileczki")
			return 0,0


	def GetLines(self):
		field_mask = self.MaskedImg()
		edges = cv2.Canny(field_mask, 150, 300)
		lines = cv2.HoughLinesP(edges, rho=1.0, theta=np.pi / 180, threshold=20, minLineLength=50, maxLineGap=5)
		try:
			return lines
		except:
			print("Lines not found")

        #def DistanceToBall(self):
                #distance = ...
                #return distance
