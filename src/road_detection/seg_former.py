# Documentation is available here : https://huggingface.co/docs/transformers/model_doc/segformer
import cv2
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import time


class SegFormerEvaluater:
    use_1024: bool
    processor: SegformerImageProcessor
    seg_former_model: SegformerForSemanticSegmentation
    road_label: int
    def __init__(self, use_1024: bool=False) -> None:
        self.use_1024 = use_1024
        self.load_model()

    def load_model(self) -> None:
        if self.use_1024:
            self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            self.seg_former_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            self.road_label = 0
        else:
            self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            self.seg_former_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            self.road_label = 6
    def segment_image(self, cv_image: np.ndarray, use_1024:bool) -> Tuple[np.ndarray, np.ndarray]:
        if use_1024 != self.use_1024:
            self.use_1024 = use_1024
            self.load_model()
        original_height, original_width = cv_image.shape[:2]
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_image, return_tensors="pt")
        outputs = self.seg_former_model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        # Resize logits to the original image size
        logits_resized = F.interpolate(logits, size=(original_height, original_width), mode="bilinear", align_corners=False)
        return logits_resized, self.road_label

segFormerEvaluater = SegFormerEvaluater(use_1024=False)

def seg_segment_image(cv_image: np.ndarray, use_1024=False,verbose=False) -> Tuple[np.ndarray, np.ndarray]:
    original_height, original_width = cv_image.shape[:2]
    start_time = time.time()
    logits_resized, road_label = segFormerEvaluater.segment_image(cv_image=cv_image, use_1024=use_1024)
    end_model_time = time.time()
    if verbose:
        print(f"Model inference time: {end_model_time - start_time}")

    # Get the predicted labels
    pred_labels = torch.argmax(logits_resized, dim=1)  # shape (batch_size, height, width)
    pred_labels = pred_labels.squeeze().cpu().numpy()  # Convert to numpy array and remove batch dimension

    end_resize_time = time.time()
    if verbose:
        print(f"Resize time: {end_resize_time - end_model_time}")
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

    return segmented_image, pred_labels, road_label