import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import namedtuple
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths
import subprocess

Manifold = namedtuple('Manifold', ['features', 'radii'])
PrecisionAndRecall = namedtuple('PrecisionAndRecall', ['precision', 'recall'])

# Improved Precision and Recall (IPR) Class
class IPR:
    def __init__(self, device, batch_size=50, k=1, num_samples=5000):
        self.manifold_ref = None
        self.batch_size = batch_size
        self.k = k
        self.num_samples = num_samples
        self.device = device

        print("Loading VGG16 for Improved Precision and Recall...")
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:4]) 
        self.vgg16 = vgg16.to(device).eval()

    def compute_manifold_ref(self, real_images_path):
        self.manifold_ref = self.compute_manifold(real_images_path)

    def compute_manifold(self, images_path):
        # Extract features and compute pairwise distances for manifold
        features = self.extract_features_from_files(images_path)
        distances = compute_pairwise_distances(features)
        radii = distances2radii(distances, k=self.k)
        return Manifold(features, radii)

    def precision_and_recall(self, fake_images_path):
        assert self.manifold_ref is not None, "Compute the reference manifold with real images first."
        manifold_subject = self.compute_manifold(fake_images_path)
        precision = compute_metric(self.manifold_ref, manifold_subject.features, "Computing precision...")
        recall = compute_metric(manifold_subject, self.manifold_ref.features, "Computing recall...")
        return PrecisionAndRecall(precision, recall)

    def extract_features_from_files(self, path):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = ImageFolder(path, transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        features = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting features from {path}"):
                batch = batch.to(self.device)
                feature = self.vgg16(batch)
                features.append(feature.cpu().data.numpy())
        return np.concatenate(features, axis=0)

class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.fnames = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(".png") or fname.endswith(".jpg")]
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.fnames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.fnames)

def compute_pairwise_distances(X, Y=None):
    X = X.astype(np.float32)
    if Y is None:
        Y = X
    else:
        Y = Y.astype(np.float32)

    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-6

    X_norm_square = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm_square = np.sum(Y ** 2, axis=1, keepdims=True)
    XY = np.dot(X, Y.T)
    diff_square = X_norm_square - 2 * XY + Y_norm_square.T
    diff_square[diff_square < 0] = 0  # Ensure no negative distances
    return np.sqrt(diff_square)

def distances2radii(distances, k=1):
    radii = np.zeros(distances.shape[0], dtype=np.float16)
    for i in range(distances.shape[0]):
        radii[i] = get_kth_value(distances[i], k)
    return radii

def get_kth_value(arr, k):
    return np.partition(arr, k)[k]

def compute_metric(manifold_ref, feats_subject, desc=""):
    count = 0
    dist = compute_pairwise_distances(manifold_ref.features, feats_subject)
    for i in range(feats_subject.shape[0]):
        count += (dist[:, i] < manifold_ref.radii).any()
    return count / feats_subject.shape[0]

def compute_precision_recall(sample_file, device='cpu'):

    ipr = IPR(device=device, k=5, batch_size=64, num_samples=10000)
    ipr.compute_manifold_ref('data/png_test')

    metric = ipr.precision_and_recall(sample_file)
    
    return metric.precision, metric.recall#, fid_value

def compute_fid(sample_file, device='cpu'):
    subprocess.run(["python", "-m", "pytorch_fid", 'data/png_test', sample_file])
    
    #fid_value = calculate_fid_given_paths(['data/png_test', 'samples'],batch_size=64, device=device, dims=2048)
    #return fid_value