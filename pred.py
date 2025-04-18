import numpy as np
import cv2
import tensorflow as tf
import librosa
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class Video:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path='modelv0.2b.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @staticmethod
    def extract_frames(video_path, num_frames=10, resize=(128, 128)):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        cap = cv2.VideoCapture(video_path)
        frames = []
        face_detected = False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(frame_count // num_frames, 1)
        
        for i in range(0, frame_count, step):
            if len(frames) < num_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(faces) > 0:
                        face_detected = True
                    frame = cv2.resize(frame, resize)
                    frames.append(frame)
                else:
                    break
        cap.release()
        return np.array(frames), face_detected

    def predict(self, path):
        frames, face_detected = self.extract_frames(path)
        
        if not face_detected:
            return [0, "Face Not Detected"]

        frames = frames.astype('float32') / 255.0
        frames = frames.reshape((-1, 128, 128, 3))

        predictions = []
        for frame in frames:
            self.interpreter.set_tensor(self.input_details[0]['index'], [frame])
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(output_data[0])

        average_prediction = np.mean(predictions, axis=0)
        class_names = ['Real', 'Fake']
        
        video_class = 1 if average_prediction[1] > 0.5 else 0
        return class_names[video_class]

class Audio:
    def __init__(self):
        self.model = load_model('audiomodel.h5')

    @staticmethod
    def prepare_data(X, window_size=5):
        data = []
        for i in range(len(X)):
            row = X.iloc[i].values
            row_data = []
            for j in range(len(row) - window_size):
                window = row[j: j + window_size]
                row_data.append(window)
            data.append(row_data)
        return np.array(data)

    @staticmethod
    def extract_features(file_path):
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs

    def predict(self, file_path):
        features = []

        features.append(np.array([-319.13507  ,  118.986855 ,   -6.759981 ,   21.32388  ,
        -11.489506 ,    8.984695 ,    0.9885412,   -9.283413 ,
         -6.9272733,   -4.9711795,    2.425839 ,   -9.179202 ,
         -0.7424884], dtype=np.float32))

        feature_vector = self.extract_features(file_path)
        features.append(feature_vector)

        df = pd.DataFrame(features)
        df = MinMaxScaler().fit_transform(df)
        df = self.prepare_data(pd.DataFrame(df), window_size=5)

        predictions = self.model.predict(df)
        preds = list(np.round(predictions.flatten()).astype(int))

        types = ["Fake", "Real"]
        return types[preds[-1]]

class Image:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path='modelv0.2b.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @staticmethod
    def preprocess_frame(frame, resize=(128, 128)):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_detected = len(faces) > 0
        frame_resized = cv2.resize(frame, resize)
        frame_resized = frame_resized.astype('float32') / 255.0
        return frame_resized, face_detected

    def predictor(self, file_path):
        if isinstance(file_path, str):
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return [0, "Error Occured!"]
        else:
            frame = file_path

        frame, face_detected = self.preprocess_frame(frame)
        
        if not face_detected:
            return [0, "Face Not Detected"]

        frame = frame.reshape((1, 128, 128, 3))
        self.interpreter.set_tensor(self.input_details[0]['index'], frame)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        class_names = ['Real', 'Fake']
        threshold = 0.5000000596046448
        prediction = output_data[0][1]
        video_class = int(prediction > threshold)
        return class_names[video_class]
