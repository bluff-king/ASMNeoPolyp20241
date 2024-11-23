import torch
import argparse
import os
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as T

# Function to preprocess the input image
def preprocess_image(image_path, resize_dim=(256, 256)):
    image = cv2.imread(image_path)  # Read as BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, resize_dim)  # Resize to target dimensions
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet standard
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to map model outputs to the original mask format
def postprocess_mask(output, original_size):
    # Convert model logits to class indices
    mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # Shape: (H, W)
    
    # Resize back to the original image dimensions
    mask_resized = cv2.resize(mask, original_size[::-1], interpolation=cv2.INTER_NEAREST)
    
    return mask_resized

# Function to map segmentation mask to RGB using a color dictionary
def mask_to_rgb(mask, color_dict):
    h, w = mask.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_dict.items():
        rgb_image[mask == class_id] = color
    return rgb_image

# Function to postprocess the output and save the result
def save_segmentation(output_mask, save_path, original_size, color_dict):
    mask_rgb = mask_to_rgb(output_mask, color_dict)
    cv2.imwrite(save_path, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))  # Save in BGR for OpenCV
    print(f"Segmented image saved to: {save_path}")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Run inference using a pretrained U-Net++ model.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()

    # Path to checkpoint and other configurations
    checkpoint_path = "code/model_step_change_3.pth"
    resize_dim = (256, 256)
    num_classes = 3
    color_dict = {
        0: (0, 0, 0),       # Background: Black
        1: (255, 0, 0),     # Class 1: Red
        2: (0, 255, 0),     # Class 2: Green
        # Add more classes and colors as needed
    }

    # Load the model
    print("Loading model...")
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,  # No pretrained encoder weights
        in_channels=3,  # Input channels (RGB)
        classes=num_classes  # Number of output classes
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Preprocess the input image
    print(f"Processing image: {args.image_path}")
    original_image = cv2.imread(args.image_path)
    original_size = original_image.shape[:2]  # (height, width)
    input_image = preprocess_image(args.image_path, resize_dim).to(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(input_image)  # Shape: (1, num_classes, H, W)

    # Postprocess and save the output
    mask_resized = postprocess_mask(output, original_size)
    output_filename = "segmented_" + os.path.basename(args.image_path)
    save_segmentation(mask_resized, output_filename, original_size, color_dict)
    print("Inference complete.")
