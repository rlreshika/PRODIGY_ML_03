import os
import numpy as np
from skimage import feature, io, transform
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Function to extract HOG features from an image
def extract_hog_features(image):
    hog_features = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    return hog_features

# Load and preprocess the dataset
def load_dataset(root_dir, max_images_per_class=None):
    images = []
    labels = []
    
    classes = ['cats', 'dogs']
    for class_index, class_name in enumerate(classes):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory {class_dir} does not exist.")
            continue
        
        file_list = os.listdir(class_dir)
        
        if max_images_per_class is not None:
            file_list = file_list[:max_images_per_class]
        
        for file_name in file_list:
            image_path = os.path.join(class_dir, file_name)
            try:
                image = io.imread(image_path, as_gray=True)
                image = transform.resize(image, (128, 128))  # Resize images to 128x128
                hog_features = extract_hog_features(image)
                images.append(hog_features)
                labels.append(class_index)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    
    return np.array(images), np.array(labels)

# Specify the root directory of your dataset
root_dir = '/kaggle/input/dogs-vs-cats/train'

# Load dataset
X, y = load_dataset(root_dir, max_images_per_class=1000)  # Limiting to 1000 images per class for demonstration

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVM classifier
svm = SVC(kernel='linear', random_state=42)

# Train SVM model
svm.fit(X_train, y_train)

# Predict on test set
y_pred = svm.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

# Plot some sample results
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

# Assuming 8100 features, reshape to (90, 90) for visualization
for i in range(10):
    img_index = np.random.randint(0, len(X_test))
    img_hog = X_test[img_index].reshape((90, 90))  # Adjust the reshape dimensions
    img_label = y_test[img_index]
    img_pred = y_pred[img_index]

    axes[i].imshow(img_hog, cmap='gray')
    axes[i].set_title(f"True: {['cat', 'dog'][img_label]}, Pred: {['cat', 'dog'][img_pred]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
