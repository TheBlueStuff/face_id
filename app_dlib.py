import os
import cv2
import uuid
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import dlib
from scipy.spatial import distance

# Paths to the dlib models
PREDICTOR_PATH = "./shape_predictor_5_face_landmarks.dat"
FACE_REC_MODEL_PATH = "./dlib_face_recognition_resnet_model_v1.dat"

# Initialize dlib's models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)

class App:
    def __init__(self, window, window_title, video_source=1):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.vid = MyVideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=1600, height=800)
        self.canvas.pack()

        self.btn_snapshot = tk.Button(window, text="Capture Photo", command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo_path = f"./captured_{uuid.uuid4()}.jpg"
            cv2.imwrite(self.photo_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Photo saved to {self.photo_path}")

            self.verify_face(self.photo_path)

    def update(self):
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def get_face_descriptor(self, image_path):
        img = dlib.load_rgb_image(image_path)
        dets = detector(img, 1)

        if len(dets) == 0:
            raise ValueError(f"No faces found in the image {image_path}")

        shape = sp(img, dets[0])
        face_chip = dlib.get_face_chip(img, shape)
        face_descriptor = facerec.compute_face_descriptor(face_chip)

        return np.array(face_descriptor)

    def compare_faces(self, face1_path, face2_path, threshold=0.6):
        face_descriptor1 = self.get_face_descriptor(face1_path)
        face_descriptor2 = self.get_face_descriptor(face2_path)
        dist = distance.euclidean(face_descriptor1, face_descriptor2)
        return dist < threshold, dist

    def verify_face(self, test_image_path):
        reference_image_path = "./mario_0.jpg"
        try:
            result, dist = self.compare_faces(reference_image_path, test_image_path)
            print(f"The two faces are the same person: {result}")
            print(f"Distance: {dist}")
        except ValueError as e:
            print(e)
        finally:
            # Remove the captured photo
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                print(f"Removed {test_image_path}")

class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (False, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__ == "__main__":
    App(tk.Tk(), "Webcam Photo Capture")
