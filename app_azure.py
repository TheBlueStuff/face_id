import os
import cv2
import uuid
import tkinter as tk
from PIL import Image, ImageTk
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import QualityForRecognition

# Set your Azure Face API key and endpoint
KEY = os.environ["VISION_KEY"]
ENDPOINT = os.environ["VISION_ENDPOINT"]

# Set your person group ID and person ID
PERSON_GROUP_ID = '4e152da4-6eda-4257-bf35-6790c6bc3ea6'
PERSON_ID = '6b74177a-1159-4b6b-adc0-32ddac9204f7'

# Create a FaceClient
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

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

    def verify_face(self, photo_path):
        with open(photo_path, "rb") as image:
            test_faces = face_client.face.detect_with_stream(
                image=image,
                detection_model='detection_03',
                recognition_model='recognition_04',
                return_face_attributes=['qualityForRecognition']
            )

        if not test_faces:
            print("No faces detected in the image.")
            os.remove(photo_path)
            return

        test_face = test_faces[0]
        if test_face.face_attributes.quality_for_recognition in [QualityForRecognition.high, QualityForRecognition.medium]:
            test_face_id = test_face.face_id
        else:
            print("Calidad de imagen insuficiente para reconocimiento")
            os.remove(photo_path)
            return

        verify_result = face_client.face.verify_face_to_person(test_face_id, PERSON_ID, PERSON_GROUP_ID)
        print('Verification result: {}. Confidence: {}'.format(verify_result.is_identical, verify_result.confidence))
        if os.path.exists(photo_path):
            os.remove(photo_path)
            print(f"Removed {photo_path}")

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
