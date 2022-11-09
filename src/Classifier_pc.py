import pickle
import torch
import cv2
from typing import Literal, Tuple

from src.utils import normalize_hand, create_sentence
from pointnet.PointNet import PointNet, eval_model, eval_model_scores

from src.AssemblyAI import AssemblyAI
from src.Webcam import Webcam

class Classifier:
    def __init__(self, model: Literal["abc", "conversation"] = "conversation", noise_lvl : int = 4) -> None:
        # Config constants
        self.DEVICE = "mps"
        self.CONFIDENCE_SCORE = 50 # Percent
        self.NOISE_LEVEL = noise_lvl # Number of frames considered noise
        self.MODE = model
        self.assembly_ai = AssemblyAI()

        self.WEBCAM = Webcam()
        with open(f"./model/{model}_id2label.pkl", "rb") as f:
            self.id2label = pickle.load(f)
        self.model = PointNet(classes=len(self.id2label), device=self.DEVICE)
        self.model.load_from_pth(f"./model/{model}.pth")

    def classify_snapshot(self, tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Classifies a single frame of data

        Args:
            tensor (torch.Tensor): The tensor representing the hands

        Returns:
            Tuple[str, float]: The label and confidence score
        """

        scores = eval_model_scores(
            self.model,
            tensor,
            self.id2label,
            device=self.DEVICE
        )

        return scores

    def points_to_tensor(self, points, hands) -> torch.Tensor:
        """
        Converts the points to a tensor

        Args:
            points (List[List[Point]]): The points
            hands (List[Hand]): The hands

        Returns:
            torch.Tensor: The tensor
        """
        left = right = None

        num_hands = len(hands)
        for idx in range(num_hands):
            points_arr = points[idx]
            hand = hands[idx]

            if hand == "Left":
                left = points_arr
            elif hand == "Right":
                right = points_arr

        left_tensor = torch.tensor(left) if left else torch.zeros((21, 3))
        right_tensor = torch.tensor(right) if right else torch.zeros((21, 3))

        left_tensor = normalize_hand(left_tensor)
        right_tensor = normalize_hand(right_tensor)

        data = torch.cat((left_tensor, right_tensor), dim=0).to(self.DEVICE)
        return data if (left or right) else None

    def capture_points(self):
        """
        Captures the points from webcam or socket and returns them

        Returns:
            points, hands (Tuple[List[List[Point]], List[Hand]]): The points and hands
        """

        return self.WEBCAM.process_next()

    def log(self, msg: str):
        print(f"[Classifier] {msg}")

    def run(self):
        """
        Main entry point for the classifier
        """
        # For sentence creation (instead of individual words)
        sentences, curr_sentence, curr_nones = [], [], 0
        frame_idx = 0

        # For abc only

        self.log("Starting live classification...")
        scores = []

        while True:
            # Capture points from camera or socket
            points, hands, frame = self.capture_points()

            # Transform data into tensor
            if hands and frame_idx % 5 == 0:
                curr_nones = 0
                tensor = self.points_to_tensor(points, hands)

                # Classify
                scores = self.classify_snapshot(tensor)
            else:
                curr_nones += 1

                # If curr_nones is greater than NOISE_LEVEL, then we have a sentence
                if curr_nones > self.NOISE_LEVEL:
                    scores = []

            # Display
            self.WEBCAM.display_text(scores, frame)

            # Convert to BGR for display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Live Classification", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                self.CAMERA.release()
                cv2.destroyAllWindows()
                exit()
            elif key & 0xFF == ord("r"):
                self.assembly_ai.record_audio()

            frame_idx += 1
