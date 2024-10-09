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
    device:str

    def __init__(self, use_1024: bool=False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_1024 = use_1024
        self.load_model()
        

    def load_model(self) -> None:
        self.processor = SegformerImageProcessor(do_resize=False)
        if self.use_1024:
           # self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            self.seg_former_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            self.road_label = 0
        else:
           # self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            self.seg_former_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            self.road_label = 6
        
        self.seg_former_model.to(self.device)
    
    def segment_image(self, cv_image: np.ndarray, use_1024:bool) -> Tuple[np.ndarray, np.ndarray]:
        if use_1024 != self.use_1024:
            self.use_1024 = use_1024
            self.load_model()

        original_height, original_width = cv_image.shape[:2]
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self.seg_former_model(pixel_values)
            logits = outputs.logits
        # Resize logits to the original image size
        logits_resized = F.interpolate(logits, size=(original_height, original_width), mode="bilinear", align_corners=False)
        return logits_resized, self.road_label
