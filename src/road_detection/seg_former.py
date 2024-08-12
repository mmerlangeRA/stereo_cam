import cv2
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Load the feature extractor and model
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")

def seg_segment_image(cv_image: np.ndarray, verbose=False) -> Tuple[np.ndarray, np.ndarray]:
    # Convert OpenCV image (BGR) to PIL image (RGB)
    original_height, original_width = cv_image.shape[:2]
    pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # Preprocess the image and pass it through the model
    inputs = feature_extractor(images=pil_image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # Resize logits to the original image size
    logits_resized = F.interpolate(logits, size=(original_height, original_width), mode="bilinear", align_corners=False)

    # Get the predicted labels
    pred_labels = torch.argmax(logits_resized, dim=1)  # shape (batch_size, height, width)
    pred_labels = pred_labels.squeeze().cpu().numpy()  # Convert to numpy array and remove batch dimension

    # Define a color map (cityscapes has 19 classes)
    colors = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                       (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
                       (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                       (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)])

    # Create an empty array to hold the segmented image
    segmented_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Map each label to the corresponding color
    for label in range(len(colors)):
        segmented_image[pred_labels == label] = colors[label]

    return segmented_image, pred_labels