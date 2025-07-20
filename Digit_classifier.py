import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
import matplotlib.pyplot as plt

# Load the digit data from text file
def load_digit_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = np.array([list(map(float, line.strip().split())) for line in lines])
    X = data[:, 1:]  # 256 grayscale pixel values
    y = data[:, 0].astype(int)  # labels
    return X, y

# HOG feature extraction for all samples
def extract_hog_features(images):
    features = []
    for img in images:
        # Dividing each 16x16 image into a 4x4 cell which results in 16 cells per image 
        img_reshaped = img.reshape(16, 16)
        hog_feat = hog(img_reshaped, pixels_per_cell=(4, 4), cells_per_block=(1, 1), feature_vector=True)
        features.append(hog_feat)
    return np.array(features)

# Load data
X_train_raw, y_train = load_digit_data("train-data.txt")
X_test_raw, y_test = load_digit_data("test-data.txt")

# Extract HOG features
X_train = extract_hog_features(X_train_raw)
X_test = extract_hog_features(X_test_raw)

# Train SVM classifier
# rbf is the kernal that best works with SVM models with the gamma set to 1
svm_model = SVC(kernel='rbf', gamma='scale')
svm_model.fit(X_train, y_train)

# Predictions
y_pred = svm_model.predict(X_test)




# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = np.arange(10)

# Plot confusion matrix to better see the outputs of the matrix
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix', pad=20)
fig.colorbar(cax)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

plt.xlabel('Predicted')
plt.ylabel('Actual')

# Accuracy Score placed on the plot graph. 
acc = accuracy_score(y_test, y_pred)

plt.figure(figsize=(4, 6))
plt.bar(["Accuracy"], [acc], color='green')
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.ylabel("Accuracy Score")
plt.text(0, acc + 0.02, f"{acc:.2f}", ha='center')
plt.tight_layout()
plt.show()


# Get classification metrics
report = classification_report(y_test, y_pred, output_dict=True)

# Extract F1-scores per digit
f1_scores = [report[str(i)]['f1-score'] for i in range(10)]

plt.figure(figsize=(10, 5))
plt.bar(np.arange(10), f1_scores, tick_label=[str(i) for i in range(10)], color='skyblue')
plt.ylim(0, 1)
plt.title("F1 Score per Digit Class")
plt.xlabel("Digit")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.show()

# Annotate each cell with the count
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                va='center', ha='center', color='black')

plt.tight_layout()
plt.show()

# Evaluate the accuracy score and ouptput the confusion matrix. 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# To Visualize a few correct/incorrect predictions 
def plot_sample_predictions(X_raw, y_true, y_pred, num=10):
    correct = np.where(y_true == y_pred)[0][:num]
    incorrect = np.where(y_true != y_pred)[0][:num]
    
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(correct):
        plt.subplot(2, num, i + 1)
        plt.imshow(X_raw[idx].reshape(16, 16), cmap='gray')
        plt.title(f"✓ {y_pred[idx]}")
        plt.axis('off')

    for i, idx in enumerate(incorrect):
        plt.subplot(2, num, num + i + 1)
        plt.imshow(X_raw[idx].reshape(16, 16), cmap='gray')
        plt.title(f"x {y_pred[idx]} (T:{y_true[idx]})")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

#plot_sample_predictions(X_test_raw, y_test, y_pred)  # Uncomment to see the sampled predictions
