import cv2 # Computer vision library
import numpy as np
import serial
ser = serial.Serial('COM6', 115200)

# Read the color image
image = cv2.imread("D:\\DTCB\\code\\anhxoa.jpg")
images = image 
# Make a copy
new_image = image.copy()
 
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

 
# Convert the grayscale image to binary

#ret, binary = cv2.threshold(blur, 0, 255,   cv2.THRESH_OTSU)
thresh = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY_INV)[1]
thresh =~thresh
kernel = np.ones((5,5), dtype= np.uint8)
binary = cv2.erode(thresh , kernel= kernel, iterations= 1)
 

# Find the contours on the inverted binary image, and store them in a list
# Contours are drawn around white blobs.
# hierarchy variable contains info on the relationship between the contours
contours, hierarchy = cv2.findContours(thresh,
  cv2.RETR_TREE,
  cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key= cv2.contourArea, reverse= True)[:1]

x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 1)
length = (x+w)*20/800
injection = round((length*0.3349 + 2.0606), 2)*10
cv2.putText(image, f'length= {length} cm and injection point= {injection} cm', (int((x+w)/8), y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
# mask = np.zeros((image.shape[:2]), np.uint8)   
# draw = cv2.drawContours(mask, contours, 0, (255,255,255), -1) 
# cvt = np.stack((thresh,)*3, axis= -1)
# merge = np.hstack((cvt, image))

# cv2.imshow('All contours with bounding box', image)
# injection = 10
# cv2.waitKey(0)
# cv2.destroyAllWindows()
point = 50
dir = 1
value = bytes('{} {}\n'.format(dir, point), encoding= 'utf8')
ser.write(value)

# ser.write(b'12345\n')
# import serial

# ser = serial.Serial(
#     port='/dev/ttyUSB1',
#     baudrate=38400,
#     parity=serial.PARITY_NONE,
#     stopbits=serial.STOPBITS_ONE,
#     bytesize=serial.EIGHTBITS
# )

# import serial

# ser = serial.Serial('COM6', 115200) #check that port on your device from Device Manager

# # ser = serial.Serial(
# #     port='/dev/ttyUSB1',
# #     baudrate=38400,
# #     parity=serial.PARITY_NONE,
# #     stopbits=serial.STOPBITS_ONE,
# #     bytesize=serial.EIGHTBITS
# # )