"""
Script used to create a dataset of hand positions
"""

import cv2
import time
import os
import json
from utils import points_to_tensor
from Webcam import Webcam


def save_points(data, label, fname):
    """
    Save data points to json file with label

    Args:
        data (list): List of data points
        label (str): Label of data points
    """
    file = f"./data/raw/{fname}"

    if not os.path.exists(file):
        with open(file, "w") as f:
            json.dump([], f)

    with open(file, "r") as f:
        try:
            prev = json.load(f)
        except json.decoder.JSONDecodeError:
            prev = []

    for d in data:
        prev.append({"data": d.tolist(), "label": label})

    with open(file, "w") as f:
        json.dump(prev, f)


def harvest(label: str, num_hands: int, num_frames: int, fname: str):
    """
    Runs the harvest script to collect num_frames of data
    for the given label and number of hands

    Args:
        label (str): Label of data points
        num_hands (int): Number of hands to track
    """
    webcam = Webcam()

    now = time.time()

    # For each frame save location
    data_points = []

    while True:
        points, hands, frame = webcam.process_next()

        # Display countdown (5 seconds to get ready)
        curr_time = time.time()
        time_left = int(5 - (curr_time - now))
        if time_left > 0:
            # Display countdown
            cv2.putText(
                frame,
                str(time_left),
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
        elif len(data_points) < num_frames:  # Save only hand positions in new image
            # Display remaining frames
            remaining_frames_text = f"Remaining frames: {num_frames - len(data_points)}"
            cv2.putText(
                frame,
                str(remaining_frames_text),
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
            )

            num_detected_hands = len(hands)

            # Extract landmarks and multi hands
            tensor = points_to_tensor(points, hands)

            # Store if hand detected
            if num_detected_hands == num_hands:
                data_points.append(tensor)
                print(
                    f"[{len(data_points)}/{num_frames}] ({len(data_points) / num_frames * 100:.2f}%)"
                )
            else:
                print("Not enough or too many hands detected")
        else:
            save_points(data_points, label, fname)
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return True


if __name__ == "__main__":
    labels = []
    frames_per_label = 200
    fname = input("Enter filename ('data/raw/{fname}'): ")

    while True:
        label = input("Enter label (or 'exit'): ")
        if label == "exit":
            break

        labels.append(label)
        num_hands = int(input("Enter number of hands to track: "))
        exited = harvest(label, num_hands, frames_per_label, fname)

        if exited:
            break

    print(f"Saved {len(labels)} labels: {labels}")
