import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CLEVRERVideoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, clip_length=None):
        """
        Args:
            root_dir (str): Base directory of the CLEVRER videos (e.g., 'data/clevrer/videos').
            split (str): Which split to use ('train', 'validation', or 'test').
            transform: A torchvision.transforms transformation (applied per frame).
            clip_length (int): Fixed number of consecutive frames to sample from each video.
                               If None, returns the entire video.
        """
        self.split = split
        self.root_dir = os.path.join(root_dir, split)
        self.video_paths = []
        # Gather all .mp4 files
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.mp4'):
                    self.video_paths.append(os.path.join(root, file))
                    
        if not self.video_paths:
            raise RuntimeError(f"No video files found in {self.root_dir}.")
            
        self.transform = transform
        self.clip_length = clip_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            if self.transform:
                pil_img = self.transform(pil_img)
            else:
                pil_img = transforms.ToTensor()(pil_img)
            frames.append(pil_img)
        cap.release()
        video_tensor = torch.stack(frames)  # [T, C, H, W]
        
        # If clip_length is defined and video is longer, sample a random contiguous clip.
        if self.clip_length is not None and video_tensor.size(0) > self.clip_length:
            T = video_tensor.size(0)
            start_idx = torch.randint(0, T - self.clip_length + 1, (1,)).item()
            video_tensor = video_tensor[start_idx:start_idx+self.clip_length]
            
        return video_tensor, video_path

if __name__ == '__main__':
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    # For training, we use a fixed-length clip (e.g. 6 frames)
    dataset = CLEVRERVideoDataset(root_dir='data/clevrer/videos', split='train', transform=test_transform, clip_length=6)
    print(f"Number of video files: {len(dataset)}")
    
    video_tensor, path = dataset[0]
    print(f"Video path: {path}")
    print(f"Video tensor shape: {video_tensor.shape}")
    # Expected output: Video tensor shape: [6, 3, 128, 128] (if the video is long enough)
