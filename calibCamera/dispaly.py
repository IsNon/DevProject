import cv2 as cv
import glob

images = glob.glob('a/*.jpg')

a_number_of_images = 1

for fname in images:

    # print(fname)
    # imagePath = 'D:\Project\images{}.png'.format(n)

    img = cv.imread(fname)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow('img', img)
    cv.waitKey(1000)

cv.destroyAllWindows()