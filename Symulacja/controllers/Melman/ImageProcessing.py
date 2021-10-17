import cv2
import numpy as np

class Vision:
	def __init__(self,camera_device,timestep):
		self.camera=camera_device
		self.timestep=timestep
		self.camera.enable(timestep)

		# parametry do maski na murawe
		self.lower_green = np.array([44, 100, 0])
		self.upper_green = np.array([80, 255, 255])
		#parametry do maski na bramke
		self.lower_gate = np.array([7,71,91])# od 8 juz opornie
		self.upper_gate = np.array([20,86,97])
		#parametry do maski na bramke oraz linie boiska
		self.lower_lines= np.array([0,0,208])# od 8 juz opornie
		self.upper_lines = np.array([179,255,255])

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

	def MaskedGate(self):
		img = self.camera.getImageArray()
		img = np.asarray(img, dtype=np.uint8)
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # do hsv
		gate_mask = cv2.inRange(imgHSV, self.lower_gate, self.upper_gate)
		return gate_mask

	def MaskedLines(self):
		img = self.camera.getImageArray()
		img = np.asarray(img, dtype=np.uint8)
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # do hsv
		gate_mask = cv2.inRange(imgHSV, self.lower_lines, self.upper_lines)
		gate_mask=cv2.dilate(gate_mask, np.ones((5,5),np.uint8), iterations=3)
		#gate_mask=cv2.erode(gate_mask, np.ones((9,9),np.uint8))
		return gate_mask

	def GetGate(self):
		gate_mask = self.MaskedGate()
		contours, hierarchy = cv2.findContours(gate_mask,
											   cv2.RETR_CCOMP,
											   cv2.CHAIN_APPROX_SIMPLE)
		max_pixels_val = 0
		try:
			for cnt in contours:
				rect = cv2.boundingRect(cnt)
				x, y, w, h = rect
				if w>0.4*h:
					pixels_val = np.sum(img[y:y + h, x:x + w])  # licze wartosc pixeli w prostokacie
					if pixels_val > max_pixels_val:
						biggest_rectangle = rect
						max_pixels_val = pixels_val
			x, y, w, h = biggest_rectangle
			'''cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
			cv2.imshow("obraz", img)
			cv2.imshow("mask", gate_mask)
			cv2.waitKey(100) '''                                        #jak chcesz sprawdzic czy wykrywa odkomentuj
		except:
			print("nie znalazl")
		return w ### mozesz wyznaczyc odleglosc do bramki tak samo jak do pilki zwraca ci wysokosc bramki w pikselach

	def GetBall(self):
		field_mask = self.MaskedImg()
		circles = cv2.HoughCircles(field_mask, cv2.HOUGH_GRADIENT, 1.5, 160, param1=80, param2=30, minRadius=0, maxRadius=30)
		try:
			for j in circles[0, :]:
				odleglosc = 0.05/((j[2]/160)*np.tan(0.56/2))
				print("Pilka jest w odleglosi "+str(2*odleglosc)+"m")
				xBall = j[0]
			return 2*odleglosc, xBall
			
		except:
			print("Nie znaleziono pileczki")
			return 0,0

	def VelocityOfBall(self,pos1,pos2):
        	#pos1 = [x1,d1,t1], pos2 = [x2,d2,t2]
        	
        	w = 320                             #frame width (szerokość ramki)
        	FOV = 0.56                          #u nas FieldOfView = 0.56
        	
        	x1 = pos1[0]
        	d1 = pos1[1]
        	t1 = pos1[2]
        	
        	x2 = pos2[0]
        	d2 = pos2[1]
        	t2 = pos2[2]
        	
        	d = (d1+d2)/2                         #srednia odleglosc piłki od zawodnika
        	d_d = d2-d1				#roznica odleglosci piłki od zawodnika
        	d_x = x2-x1				#roznica polozenia pilki w poziomie
        	d_t = t2-t1				#roznica czasu dwoch polozen
        
        	p1 = np.sqrt( np.square(d1) + np.square(x1/w*FOV*d1) )
        	p2 = np.sqrt( np.square(d2) + np.square(x2/w*FOV*d2) )
        	velocity = (p2-p1)/d_t
        
        	#velocity = ( np.sqrt( np.square(d_d) + np.square(d_x/w*FOV*d) ) ) / d_t
        	if np.abs(velocity) > 20: velocity = 0
        	return velocity


	def GetLines(self):
		field_mask = self.MaskedLines()
		edges = cv2.Canny(field_mask, 150, 300)
		lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi *0.5/ 180, threshold=10, minLineLength=1, maxLineGap=5)# z tego powodu iz robione na masce to threshould 1 jest najdokladniejszy
		try:
			print("Lines found")
			return lines
		except:
			print("Lines not found")
			return None

			
