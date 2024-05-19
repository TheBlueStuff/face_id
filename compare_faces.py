import dlib
import numpy as np
from scipy.spatial import distance
import sys
import os

def get_face_descriptor(image_path, predictor_path, face_rec_model_path):
    # Load the models
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    
    # Load the image
    img = dlib.load_rgb_image(image_path)
    
    # Detect faces
    dets = detector(img, 1)
    if len(dets) == 0:
        raise ValueError("No faces found in the image {}".format(image_path))
    
    # Get the landmarks and face descriptor for the first detected face
    shape = sp(img, dets[0])
    
    # Align the face
    face_chip = dlib.get_face_chip(img, shape)
    
    # Compute the face descriptor from the aligned face
    #face_descriptor = facerec.compute_face_descriptor(face_chip, 10)
    face_descriptor = facerec.compute_face_descriptor(img, shape, 10, 0.25)
    
    return np.array(face_descriptor)

def compare_faces(face1_path, face2_path, predictor_path, face_rec_model_path, threshold=0.6):
    face_descriptor1 = get_face_descriptor(face1_path, predictor_path, face_rec_model_path)
    face_descriptor2 = get_face_descriptor(face2_path, predictor_path, face_rec_model_path)
    
    # Calculate Euclidean distance between the two face descriptors
    dist = distance.euclidean(face_descriptor1, face_descriptor2)
    
    return dist < threshold, dist

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage:\n"
            "   python compare_faces.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat face1.jpg face2.jpg\n"
        )
        sys.exit(1)
    
    predictor_path = "./shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
    face1_path = "./mario_0.jpg"
    face2_path = ""
    
    result, dist = compare_faces(face1_path, face2_path, predictor_path, face_rec_model_path)
    print("The two faces are the same person: {}".format(result))
    print("Dist: {}".format(dist))
