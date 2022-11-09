import pickle
import torch
import cv2
from typing import Literal

from src.utils import points_to_tensor
from src.PointNet import PointNet

from src.Webcam import Webcam


class Classifier:
    def __init__(self, model: Literal["abc", "conversation"], interval: int) -> None:
        # Config constants
        self.DEVICE = "mps"
        self.MODE = model
        self.INTERVAL = interval
        self.WEBCAM = Webcam()

        # Load the model
        with open(f"./data/model/{model}_id2label.pkl", "rb") as f:
            self.id2label = pickle.load(f)
        self.model = PointNet(classes=len(self.id2label), device=self.DEVICE)
        self.model.load_from_pth(f"./data/model/{model}.pth")

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
            if hands and frame_idx % self.INTERVAL == 0:
                tensor = points_to_tensor(points, hands, self.DEVICE)

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
