"""Driving server for the installed Udacity beta simulator WebSocket protocol."""

import eventlet

eventlet.monkey_patch()

import base64
import io
import json
import os

import cv2
import numpy as np
from eventlet import websocket
from PIL import Image
import tensorflow as tf
from flask import Flask

from utils import preprocess_image


TARGET_SPEED = 15.0
PORT = 4567
MODEL_PATH = "model.keras"


class SimplePIController:
    def __init__(self, kp: float, ki: float) -> None:
        self.kp = kp
        self.ki = ki
        self.target_speed = 0.0
        self.integral = 0.0

    def set_desired(self, speed: float) -> None:
        self.target_speed = speed

    def update(self, speed: float) -> float:
        error = self.target_speed - speed
        self.integral += error
        return self.kp * error + self.ki * self.integral


controller = SimplePIController(0.1, 0.002)
controller.set_desired(TARGET_SPEED)
app = Flask(__name__)
model = None


def send_control(ws, steering_angle: float, throttle: float) -> None:
    """Send the beta simulator's Socket.IO-style ``steer`` event frame."""
    payload = [
        "steer",
        {
            "steering_angle": str(float(steering_angle)),
            "throttle": str(float(throttle)),
        },
    ]
    ws.send("42" + json.dumps(payload, separators=(",", ":")))


def telemetry(data: dict) -> tuple[float, float] | None:
    """Turn one simulator camera frame into steering and throttle commands."""
    try:
        speed = float(data["speed"])
        image_bytes = base64.b64decode(data["image"])

        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb_image = np.asarray(image.convert("RGB"), dtype=np.uint8)

        # Keep preprocessing identical to training: utils.py expects BGR input.
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        processed_image = preprocess_image(bgr_image)
        image_tensor = np.expand_dims(processed_image, axis=0).astype(np.float32)

        steering_angle = float(model.predict(image_tensor, verbose=0)[0][0])
        throttle = max(0.1, min(0.35, controller.update(speed)))
        print(
            f"Speed: {speed:5.2f} | Steering: {steering_angle: .4f} | "
            f"Throttle: {throttle: .2f}",
            flush=True,
        )
        return steering_angle, throttle
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"Invalid telemetry received: {exc}", flush=True)
        return None


def connect() -> None:
    print("Simulator Connected!", flush=True)


def disconnect() -> None:
    print("Simulator Disconnected", flush=True)


@websocket.WebSocketWSGI
def simulator_socket(ws) -> None:
    """Serve the beta simulator's direct WebSocket Socket.IO event framing.

    The installed beta simulator opens ``/socket.io/?EIO=4&transport=websocket``
    directly but does not send the Socket.IO namespace-connect packet expected by
    python-socketio.  It expects the server to start the Socket.IO event stream.
    """
    connect()
    ws.send("40")

    try:
        while True:
            message = ws.wait()
            if message is None:
                break
            if isinstance(message, bytes):
                message = message.decode("utf-8")

            # Engine.IO heartbeat used by the beta simulator.
            if message == "2":
                ws.send("3")
                continue

            # Re-acknowledge a namespace open if the client sends one.
            if message == "40":
                ws.send("40")
                continue

            if not message.startswith("42"):
                continue

            try:
                event, data = json.loads(message[2:])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

            if event == "telemetry" and isinstance(data, dict):
                control = telemetry(data)
                if control is not None:
                    send_control(ws, *control)
            elif event == "manual":
                ws.send('42["manual",{}]')
    finally:
        disconnect()


def application(environ, start_response):
    """Route the simulator WebSocket upgrade; retain Flask for ordinary HTTP."""
    if (
        environ.get("PATH_INFO") == "/socket.io/"
        and environ.get("HTTP_UPGRADE", "").lower() == "websocket"
    ):
        return simulator_socket(environ, start_response)
    return app(environ, start_response)


def main() -> None:
    global model

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' was not found.")

    print("Loading model.keras...", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded.", flush=True)
    print(f"Listening on port {PORT}...", flush=True)
    eventlet.wsgi.server(eventlet.listen(("", PORT)), application)


if __name__ == "__main__":
    main()
