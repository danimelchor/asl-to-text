import pickle
import torch
import cv2
from typing import Literal

from src.utils import normalize_hand
from pointnet.PointNet import PointNet, eval_model_scores

from src.Webcam import Webcam


class Classifier:
    def __init__(
        self, model: Literal["abc", "conversation"] = "conversation", noise_lvl: int = 4
    ) -> None:
        # Config constants
        self.DEVICE = "mps"
        self.CONFIDENCE_SCORE = 50  # Percent
        self.MODE = model

        self.WEBCAM = Webcam()
        with open(f"./model/{model}_id2label.pkl", "rb") as f:
            self.id2label = pickle.load(f)
        self.model = PointNet(classes=len(self.id2label), device=self.DEVICE)
        self.model.load_from_pth(f"./model/{model}.pth")

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

        # Convert to tensors
        left_tensor = torch.tensor(left) if left else torch.zeros((21, 3))
        right_tensor = torch.tensor(right) if right else torch.zeros((21, 3))

        # Normalize
        left_tensor = normalize_hand(left_tensor)
        right_tensor = normalize_hand(right_tensor)

        # Concatenate
        data = torch.cat((left_tensor, right_tensor), dim=0).to(self.DEVICE)
        return data if (left or right) else None

    def run(self):
        """
        Main entry point for the classifier
        """
        frame_idx = 0
        frames_no_hands = 0
        scores = None

        while True:
            # Capture points from camera or socket
            points, hands, frame = self.WEBCAM.process_next()

            # Transform data into tensor
            if hands and frame_idx % 5 == 0:
                tensor = self.points_to_tensor(points, hands)

                # Classify
                scores = eval_model_scores(
                    self.model, tensor, self.id2label, device=self.DEVICE
                )

                frames_no_hands = 0
            else:
                frames_no_hands += 1

            # Display
            if frames_no_hands < 5:
                self.WEBCAM.display_text(scores, frame)

            # Convert to BGR for display
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Live Classification", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                self.WEBCAM.release()
                cv2.destroyAllWindows()
                exit()

            frame_idx += 1
