import rclpy
from rclpy.node import Node

import sounddevice as sd
import numpy as np
import time
from collections import deque, Counter
from queue import Queue
import librosa
from tensorflow import keras
import joblib
from usb_4_mic_array.tuning import Tuning
import usb.core

# ROS2 Messages
from speech_recognition_msgs.msg import (
    CommandDetected,
    WakeWordDetected,
    AudioRecorded
)


# Konfiguration
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.125
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
QUEUE_SIZE = int(2 / CHUNK_DURATION)

RMS_THRESHOLD = 0.01
SILENCE_SECONDS = 0.8


class SpeechRecognitionNode(Node):

    def __init__(self):
        super().__init__('speech_recognition_node')

        self.get_logger().info("Speech Recognition Node gestartet")

        # Publisher
        self.command_pub = self.create_publisher(
            CommandDetected,
            'command_detected',
            10
        )

        self.wakeword_pub = self.create_publisher(
            WakeWordDetected,
            'wake_word_detected',
            10
        )

        self.audio_pub = self.create_publisher(
            AudioRecorded,
            'audio_recorded',
            10
        )

        # Modell laden
        self.scaler = joblib.load("MinMaxScaler.pkl")
        self.model = keras.models.load_model("DS-CNN-97.6.keras")

        
        # DOA Setup
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if not dev:
            raise RuntimeError("ReSpeaker USB Mic Array nicht gefunden!")
        
        self.mic = Tuning(dev)

        # Audio Device finden
        devices = sd.query_devices()
        self.device_index = None

        for i, d in enumerate(devices):
            if "respeaker" in d["name"].lower():
                self.device_index = i
                break

        if self.device_index is None:
            raise RuntimeError("Kein ReSpeaker-Gerät gefunden!")

        # Queues
        self.rolling_queue = deque(maxlen=QUEUE_SIZE)
        self.snippet_queue = Queue()
        self.label_queue = deque(maxlen=QUEUE_SIZE)

        # Listen State
        self.listen_active = False
        self.listen_state = "idle"
        self.listen_buffer = []
        self.listen_result = None
        self.listen_timeout = 10
        self.listen_phrase_limit = 10
        self.listen_start_time = 0
        self.listen_silence_counter = 0

        # Audio Stream starten
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SAMPLES,
            channels=1,
            dtype="float32",
            device=self.device_index,
            callback=self.audio_callback
        )
        self.stream.start()

        # Timer für Inferenz Loop
        self.create_timer(0.05, self.process_audio)

    # Hilfsfunktionen
    def most_frequent(self, dq, min_count=4):

        if len(dq) == 0:
            return 10, 1.0

        left_values = [elem[0] for elem in dq]
        counter = Counter(left_values)
        value, count = counter.most_common(1)[0]

        if count < min_count:
            return 10, 1.0

        indices = [i for i, v in enumerate(left_values) if v == value]
        right_values = [dq[i][1] for i in indices]
        mean_confidence = sum(right_values) / len(right_values)

        return value, mean_confidence

    # Audio Callback
    def audio_callback(self, indata, frames, time_info, status):

        chunk = indata[:, 0].copy()
        self.rolling_queue.append(chunk)

        if len(self.rolling_queue) == QUEUE_SIZE:
            snippet = np.concatenate(self.rolling_queue)
            self.snippet_queue.put(snippet)

        # Listen Logik (Wakeword Folgeaufnahme)
        if self.listen_active:

            rms = np.sqrt(np.mean(chunk ** 2))

            if self.listen_state == "waiting":

                if self.listen_start_time == 0:
                    self.listen_start_time = time.time()

                if rms > RMS_THRESHOLD:
                    self.listen_state = "recording"
                    self.listen_buffer = list(chunk)
                    self.listen_start_time = time.time()

                elif time.time() - self.listen_start_time > self.listen_timeout:
                    self.listen_active = False
                    self.listen_state = "idle"
                    self.listen_start_time = 0

            elif self.listen_state == "recording":

                self.listen_buffer.extend(chunk)

                if rms < RMS_THRESHOLD:
                    self.listen_silence_counter += 1
                else:
                    self.listen_silence_counter = 0

                silence_time = self.listen_silence_counter * CHUNK_DURATION
                total_time = time.time() - self.listen_start_time

                if silence_time > SILENCE_SECONDS or total_time > self.listen_phrase_limit:

                    audio = np.array(self.listen_buffer, dtype=np.float32)

                    self.publish_audio(audio)

                    self.listen_active = False
                    self.listen_state = "idle"
                    self.listen_start_time = 0
                    self.listen_silence_counter = 0
                    self.listen_buffer = []

    # Audio Verarbeitung (Timer Loop)
    def process_audio(self):

        while not self.snippet_queue.empty():

            snippet = self.snippet_queue.get()

            spec = librosa.feature.melspectrogram(
                y=snippet,
                sr=SAMPLE_RATE,
                n_mels=128,
                hop_length=251
            )

            spec_dB = librosa.power_to_db(spec, ref=np.max)

            X_flat = spec_dB.reshape(1, -1)
            X_scaled = self.scaler.transform(X_flat)
            X_scaled = X_scaled.reshape(1, 128, 128, 1)

            y_pred = self.model.predict(X_scaled, verbose=0)
            class_detected = np.argmax(y_pred, axis=1)[0]
            confidence = float(np.max(y_pred))

            self.label_queue.append([class_detected, confidence])

            probable_class, mean_confidence = self.most_frequent(
                self.label_queue,
                min_count=8
            )

            if probable_class < 10:

                doa = float(self.mic.direction)

                if 1 <= probable_class <= 9:
                    self.publish_command(mean_confidence)

                elif probable_class == 0:
                    self.publish_wakeword(mean_confidence, doa)
                    self.listen_active = True
                    self.listen_state = "waiting"

                self.rolling_queue.clear()
                self.label_queue.clear()

    # Publisher Funktionen
    def publish_command(self, mean_confidence):

        msg = CommandDetected()
        msg.detected = True
        msg.mean_confidence = float(mean_confidence)

        self.command_pub.publish(msg)
        self.get_logger().info("CommandDetected gesendet")

    def publish_wakeword(self, mean_confidence, doa):

        msg = WakeWordDetected()
        msg.detected = True
        msg.mean_confidence = float(mean_confidence)
        msg.doa = float(doa)

        self.wakeword_pub.publish(msg)
        self.get_logger().info("WakeWordDetected gesendet")

    def publish_audio(self, audio):

        msg = AudioRecorded()
        msg.audio_data = audio.tolist()

        self.audio_pub.publish(msg)
        self.get_logger().info("AudioRecorded gesendet")

# main
def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognitionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stream.stop()
    node.stream.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
