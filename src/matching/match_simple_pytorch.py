import os
import pickle
import cv2
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

from src.utils.image_processing import crop_transparent_borders, get_transparency_mask

# Define the image preprocessing transformations
""" 
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
]) 
"""

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])

class FeatureExtractor:
    _instance = None  # Class variable to store the singleton instance

    def __new__(cls):
        if cls._instance is None:
            # Create the singleton instance
            cls._instance = super(FeatureExtractor, cls).__new__(cls)
            # Initialize the instance
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load the pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Remove the classifier layers to get features from the convolutional layers
        self.feature_extractor = nn.Sequential(*list(self.vgg16.features.children()))
        self.feature_extractor.eval()
        # Move the model to GPU if available
        self.feature_extractor.to(self.device)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Preprocess the image
        input_tensor = preprocess(image_rgb)
        # Add a batch dimension
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(input_batch)
        # Flatten the features
        features = torch.flatten(features, 1)
        # Squeeze to remove the batch dimension
        features = features.squeeze(0)  # Shape: (feature_size,)
        # Move features to CPU and convert to NumPy array
        features = features.cpu().numpy()
        return features

    def compute_feature_distance(self, features1, features2) -> float:
        # Convert features to tensors if they are numpy arrays
        if isinstance(features1, np.ndarray):
            features1 = torch.tensor(features1, dtype=torch.float32)
        if isinstance(features2, np.ndarray):
            features2 = torch.tensor(features2, dtype=torch.float32)

        # Move features to the same device
        device = self.device
        features1 = features1.to(device)
        features2 = features2.to(device)

        # Add batch dimensions if needed
        if features1.dim() == 1:
            features1 = features1.unsqueeze(0)
        if features2.dim() == 1:
            features2 = features2.unsqueeze(0)

        # Ensure the features are normalized
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)

        # Compute cosine similarity
        similarity = F.cosine_similarity(features1, features2)

        # Convert similarity to distance
        distance = 1 - similarity.item()  # Since cosine similarity ranges from -1 to 1
        return distance


class VGGMatcher:
    reference_folder_path:str
    reference_image_paths:List[str] = []
    save_path = "reference_data.pk"
    reference_features:List[torch.Tensor] = []
    force_regenerate: bool = True

    def __init__(self, reference_folder_path,save_name = "reference_data.pk", force_regenerate=True):
        self.reference_folder_path = reference_folder_path
        self.save_path = os.path.join(self.reference_folder_path, save_name)
        self.force_regenerate = force_regenerate
        self.setup_references()

    def compute_all_reference_image_paths(self)->None:
        if not os.path.exists(self.reference_folder_path):
            raise ValueError(f"Folder {self.reference_folder_path} does not exist.")
        
        self.reference_image_paths = []
        for filename in os.listdir(self.reference_folder_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    reference_image_path = os.path.join(self.reference_folder_path, filename)
                    self.reference_image_paths.append(reference_image_path)

    def crop_all_images(self) -> None:
        """
        Crops the transparent borders of all images.
        """
        self.compute_all_reference_image_paths()
        for image_path in self.reference_image_paths:
            image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
            cropped = crop_transparent_borders(image)
            h,w=image.shape[:2]
            c_h,c_w = cropped.shape[:2]
            if h!=c_h or w!=c_w:
                cv2.imwrite(image_path, cropped)

    def compute_shape_corners(self,img:np.array)->np.array:
        mask = get_transparency_mask(img)
        mask_inv = cv2.bitwise_not(mask)
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        print(approx)
        return approx

    def setup_references(self)->None:        
        print("setup_references")
        if os.path.exists(self.save_path) and not self.force_regenerate:
            print(f'Loading pickle from {self.save_path}')
            with open(self.save_path, 'rb') as f:
                self.reference_features, self.reference_image_paths  = pickle.load(f)
        else:
            featureExtractor=FeatureExtractor()
            self.compute_all_reference_image_paths()
            self.reference_features=[]
            for reference_image_path in self.reference_image_paths:
                image = cv2.imread(reference_image_path)
                features = featureExtractor.extract_features(image)
                self.reference_features.append(features)

            with open(self.save_path, 'wb') as f:
                print(f'saving pickle at {self.save_path}')
                pickle.dump((self.reference_features, self.reference_image_paths ), f)
        return 

    def find_matching(self, query_image: np.ndarray, verbose=False)->Tuple[int,float]:
        # Extract features for the query image
        feature_extractor = FeatureExtractor()
        query_features = feature_extractor.extract_features(query_image)  # NumPy array of shape (N,)

        # Convert query features to Tensor and add a batch dimension
        query_features = torch.tensor(query_features, dtype=torch.float32).unsqueeze(0)  # Shape: (1, N)

        # Convert reference features to Tensors
        reference_features = [torch.tensor(feat, dtype=torch.float32) for feat in self.reference_features]  # Each of shape (N,)

        # Stack reference features into a single Tensor
        reference_features = torch.stack(reference_features)  # Shape: (num_references, N)

        # Ensure features are on the same device
        device = feature_extractor.device
        query_features = query_features.to(device)            # Shape: (1, N)
        reference_features = reference_features.to(device)    # Shape: (num_references, N)

        # Normalize features for cosine similarity
        query_features = F.normalize(query_features, p=2, dim=1)       # Shape: (1, N)
        reference_features = F.normalize(reference_features, p=2, dim=1)  # Shape: (num_references, N)

        # Compute cosine similarities
        similarities = torch.mm(query_features, reference_features.t())  # Shape: (1, num_references)

        # Convert similarities to distances
        distances = 1 - similarities[0]  # Shape: (num_references,)

        # Move distances to CPU for NumPy operations
        distances_cpu = distances.cpu().numpy()

        if verbose:
            for index, distance in enumerate(distances_cpu):
                print(f'{self.reference_image_paths[index]}  {distance}')

        # Find the best match
        best_match_idx = np.argmin(distances_cpu)
        best_match_distance = distances_cpu[best_match_idx]
        if verbose:
            print(f'Best match: {self.reference_image_paths[best_match_idx]} with distance: {best_match_distance}')

        return best_match_idx, best_match_distance


    def get_image_path(self, index):
        return self.reference_image_paths[index]

