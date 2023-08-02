import serial
ser = serial.Serial('COM3', 115200)

point = 20
dir = 1
value = bytes('{} {}\n'.format(dir, point), encoding= 'utf8')
print(point, dir)
ser.write(value)
