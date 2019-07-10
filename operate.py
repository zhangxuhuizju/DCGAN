import cv2
import sys
import os
from glob import glob
def detectFace(img, cvmodel='lbpcascade_animeface.xml'):
    if not os.path.isfile(cvmodel):
        raise RuntimeError("%s is not exist!!!" %cvmodel)
    
    cv = cv2.CascadeClassifier(cvmodel)
    image = cv2.imread(img)
    
    convert_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    convert_image = cv2.equalizeHist(convert_image)
    
    detected_faces = cv.detectMultiScale(convert_image, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    
    for i, (x, y, w, h) in enumerate(detected_faces):
        detected_face = image[y:y+h, x:x+w, :]
        detected_face = cv2.resize(detected_face, (96, 96))
        filename = '%s.jpg' % (os.path.basename(img).split('.')[0])
        cv2.imwrite('cartoonFaces/'+filename, detected_face)
        
        
if __name__ == '__main__':
    if not os.path.exists('cartoonFaces'):
        os.makedirs('cartoonFaces')
    imgs = glob('image_raw/*.jpg')
    for img in imgs:
        detectFace(img)
        print('ok')