import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os



try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

def detect(image):
    '''
    Function to detect faces/eyes and smiles in the image passed to this function
    '''

    
    image = np.array(image.convert('RGB'))
    
    
    
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        
        
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        
        roi = image[y:y+h, x:x+w]
        
        
        eyes = eye_cascade.detectMultiScale(roi)
        
        
        smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)
        
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)

    return image, faces


def about():
	st.write(
		'''
		**Haar Cascade** is an object detection algorithm.
		It can be used to detect objects in images or videos. 
		The algorithm has four stages:
			1. Haar Feature Selection 
			2. Creating  Integral Images
			3. Training
			4. Cascading Classifiers
Read more :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid
		''')


def main():
    st.title("Face Detection App By Indu :sunglasses: ")
    st.write("**Using the Haar cascade Classifiers**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":

    	st.write("Go to the About section from the sidebar to learn more about it.")
        
        
    	image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

    	if image_file is not None:

    		image = Image.open(image_file)

    		if st.button("Process"):
                
                
    			result_img, result_faces = detect(image=image)
    			st.image(result_img, use_column_width = True)
    			st.success("Found {} faces\n".format(len(result_faces)))

    elif choice == "About":
    	about()




if __name__ == "__main__":
    main()
