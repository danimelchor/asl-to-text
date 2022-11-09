import pickle
import torch
import cv2
from typing import Literal

from src.utils import normalize_hand
from src.PointNet import PointNet

from src.Webcam import Webcam


class Classifier:
    def __init__(self, model: Literal["abc", "conversation"]) -> None:
        # Config constants
        self.DEVICE = "mps"
        self.MODE = model
        self.WEBCAM = Webcam()

        # Load the model
        with open(f"./data/model/{model}_id2label.pkl", "rb") as f:
            self.id2label = pickle.load(f)
        self.model = PointNet(classes=len(self.id2label), device=self.DEVICE)
        self.model.load_from_pth(f"./data/model/{model}.pth")

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

    def predict(self, tensor: torch.Tensor) -> str:
        """
        Predicts the label of the given tensor

        Args:
            tensor (torch.Tensor): The tensor

        Returns:
            str: The label
        """
        self.model.eval()
        with torch.no_grad():
            # Convert from 42x3 to 1x42x3
            tensor = tensor.unsqueeze(0).to(self.DEVICE)
            outputs, __, __ = self.model(tensor.transpose(1, 2))

            # Softmax over logits
            probs = torch.exp(outputs) * 100

            # Create tuples (word, probability)
            scores = [
                (self.id2label[i], probs[0][i].item())
                for i in range(len(self.id2label))
            ]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores

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
                scores = self.predict(tensor)

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
