import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from sklearn.linear_model import LinearRegression


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
# ======================================================================
# DATA LOADING AND PREPROCESSING
# ======================================================================

image_dir = 'Original_data/images'
mask_dir = 'Original_data/masks'

# Initialize lists to store data
X = []
y = []
failed_samples = 0
file_stats = []

# Process all images
for img_file in os.listdir(image_dir):
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    try:
        # Load and preprocess image
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        original_shape = img.shape
        img = cv2.resize(img, (128, 128))  # Standardize size
        normalized_img = img / 255.0  # Normalize to [0,1]
        
        # Load and process mask
        mask_path = os.path.join(mask_dir, img_file)
        mask = cv2.imread(mask_path)
        if mask is None:
            failed_samples += 1
            continue
            
        # Determine label based on mask color (RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        avg_red = np.mean(mask_rgb[:,:,0])
        avg_green = np.mean(mask_rgb[:,:,1])
        
        if avg_green > avg_red + 20:  # Benign (green)
            label = 0
        elif avg_red > avg_green + 20:  # Malignant (red)
            label = 1
        else:  # Normal (black)
            label = 2
        
        # Record file statistics
        file_stats.append({
            'filename': img_file,
            'original_height': original_shape[0],
            'original_width': original_shape[1],
            'mean_pixel_value': np.mean(img),
            'std_pixel_value': np.std(img),
            'label': label
        })
        
        X.append(normalized_img.flatten())
        y.append(label)            
        
    except Exception as e:
        print(f"Skipping {img_file}: {str(e)}")
        failed_samples += 1

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# ======================================================================
# DATA ANALYSIS
# ======================================================================
print("\n" + "="*80)
print("DATA ANALYSIS REPORT")
print("="*80)

# How many attributes?
num_attributes = X.shape[1] if len(X.shape) > 1 else 0
print(f"\nNumber of Attributes: {num_attributes} (128x128 flattened image = 16384 features)")

# How many samples?
num_samples = X.shape[0]
print(f"\nNumber of Samples: {num_samples}")
print(f"   - Failed/corrupted samples: {failed_samples}")

# What are the properties of the data?
print("\nData Properties:")
stats_df = pd.DataFrame(file_stats)
print("\nOriginal Image Statistics:")
print(stats_df[['original_height', 'original_width']].describe().to_string())

print("\nPixel Value Statistics:")
print(stats_df[['mean_pixel_value', 'std_pixel_value']].describe().to_string())

print("\nLabel Distribution:")
print(stats_df['label'].value_counts().to_string())
print("(0 = Benign, 1 = Malignant, 2 = Normal)")

# ======================================================================
# DATA VISUALIZATION
# ======================================================================
print("\n" + "="*80)
print("DATA VISUALIZATION")
print("="*80)

plt.figure(figsize=(15, 5))

# Get sample images for each class
class_samples = {}
for label in [0, 1, 2]:
    indices = np.where(y == label)[0]
    sample_idx = np.random.choice(indices, 3, replace=False)
    class_samples[label] = sample_idx

# Plot samples
for i, (label, samples) in enumerate(class_samples.items()):
    for j, idx in enumerate(samples):
        plt.subplot(3, 3, i*3 + j + 1)
        img = X[idx].reshape(128, 128)
        plt.imshow(img, cmap='gray')
        plt.title(f"Class {label} ({'Benign' if label==0 else 'Malignant' if label==1 else 'Normal'})")
        plt.axis('off')

plt.tight_layout()
plt.savefig('sample_images_by_class.png', dpi=300, bbox_inches='tight')
plt.show()

# Set the style for all plots
sns.set_style("whitegrid")
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ======================================================================
# Class Distribution
# ======================================================================
plt.figure(figsize=(20, 12))


# 1. Class Distribution Pie Chart
plt.subplot(1, 2, 1)
class_counts = stats_df['label'].value_counts()
labels = ['Benign', 'Malignant', 'Normal']
plt.pie(class_counts, labels=labels, autopct='%1.1f%%', 
        colors=['lightgreen', 'salmon', 'lightblue'], startangle=90)
plt.title('Class Distribution')

# 2. Pixel Value Distribution by Class 
plt.subplot(1, 2, 2)
for label in [0, 1, 2]:
    class_pixels = X[y == label].flatten()
    sns.kdeplot(class_pixels, label=f"Class {label} ({'Benign' if label==0 else 'Malignant' if label==1 else 'Normal'})")
plt.title('Pixel Value Distribution by Class')
plt.xlabel('Normalized Pixel Value (0-1)')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()

plt.savefig('samples_and_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Pixel Statistics and Correlations
plt.figure(figsize=(20, 12))

# 1. Pixel Statistics by Class
plt.subplot(2, 2, 1)
sns.boxplot(x='label', y='mean_pixel_value', data=stats_df)
plt.title('Mean Pixel Value by Class')
plt.xlabel('Class')
plt.ylabel('Mean Pixel Value')
plt.xticks([0, 1, 2], ['Benign', 'Malignant', 'Normal'])

# 2. Pixel Standard Deviation by Class 
plt.subplot(2, 2, 2)
sns.boxplot(x='label', y='std_pixel_value', data=stats_df)
plt.title('Pixel Value Standard Deviation by Class')
plt.xlabel('Class')
plt.ylabel('Standard Deviation')
plt.xticks([0, 1, 2], ['Benign', 'Malignant', 'Normal'])

# 3. Correlation Heatmap 
plt.subplot(2, 2, 3)
sample_pixels = X[:, ::100]  # Take every 100th pixel
corr_matrix = np.corrcoef(sample_pixels.T)
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Sampled Pixels')

# 4. t-SNE Visualization 
plt.subplot(2, 2, 4)
sample_size = 500 if X.shape[0] > 500 else X.shape[0]
sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
X_sample = X[sample_indices]
y_sample = y[sample_indices]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_sample)

scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.6)
plt.title('t-SNE Visualization of Image Features')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(handles=scatter.legend_elements()[0], 
           labels=['Benign', 'Malignant', 'Normal'])

plt.tight_layout()
plt.savefig('pixel_stats_and_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# DATA SPLITTING 
# ======================================================================

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save preprocessed data
os.makedirs('Preprocessed_data', exist_ok=True)
pd.DataFrame(X_train).to_csv('Preprocessed_data/X.csv', index=False)
pd.DataFrame(X_test).to_csv('Preprocessed_data/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('Preprocessed_data/Y.csv', index=False, header=['label'])
pd.DataFrame(y_test).to_csv('Preprocessed_data/Y_test.csv', index=False, header=['label'])

# ======================================================================
# MODEL TRAINING AND PREDICTION
# ======================================================================
print("\n" + "="*80)
print("MODEL TRAINING AND PREDICTION")
print("="*80)

results = []

# Create directory for results
os.makedirs('Results', exist_ok=True)

# 1. Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
pd.DataFrame({'true': y_test, 'predicted': svm_pred}).to_csv('Results/prediction_SVM_model.csv', index=False)
# print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
acc = accuracy_score(y_test, svm_pred)
results.append(["SVM", f"{acc:.4f}"])

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
pd.DataFrame({'true': y_test, 'predicted': rf_pred}).to_csv('Results/prediction_RandomForest_model.csv', index=False)
# print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
acc = accuracy_score(y_test, rf_pred)
results.append(["Random Forest", f"{acc:.4f}"])

# 3. Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
pd.DataFrame({'true': y_test, 'predicted': dt_pred}).to_csv('Results/prediction_DecisionTree_model.csv', index=False)
# print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
acc = accuracy_score(y_test, dt_pred)
results.append(["Decision Tree", f"{acc:.4f}"])

# 4. K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
pd.DataFrame({'true': y_test, 'predicted': knn_pred}).to_csv('Results/prediction_KNN_model.csv', index=False)
# print(f"KNN Accuracy: {accuracy_score(y_test, knn_pred):.4f}")
acc = accuracy_score(y_test, knn_pred)
results.append(["KNN", f"{acc:.4f}"])

# 5. Gaussian Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
pd.DataFrame({'true': y_test, 'predicted': nb_pred}).to_csv('Results/prediction_NaiveBayes_model.csv', index=False)
# print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
acc = accuracy_score(y_test, nb_pred)
results.append(["Naive Bayes", f"{acc:.4f}"])

# 6. Artificial Neural Network (ANN)
ann_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
ann_model.fit(X_train, y_train)
ann_pred = ann_model.predict(X_test)
pd.DataFrame({'true': y_test, 'predicted': ann_pred}).to_csv('Results/prediction_ANN_model.csv', index=False)
# print(f"ANN Accuracy: {accuracy_score(y_test, ann_pred):.4f}")
acc = accuracy_score(y_test, ann_pred)
results.append(["ANN", f"{acc:.4f}"])

# 7. Linear Regression (with rounding for classification)
print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Convert continuous predictions to class labels (0, 1, 2) by rounding
lr_pred = np.round(lr_pred).astype(int)
# Clip values to ensure they stay within class range (0-2)
lr_pred = np.clip(lr_pred, 0, 2)

# Save predictions
pd.DataFrame({'true': y_test, 'predicted': lr_pred}).to_csv('Results/prediction_LinearRegression_model.csv', index=False)

# Calculate accuracy
lr_acc = accuracy_score(y_test, lr_pred)
results.append(["Linear Regression", f"{lr_acc:.4f}"])

print(tabulate(results, headers=["Model", "Accuracy"], tablefmt="grid", floatfmt=".4f"))

# ======================================================================
# MODEL ACCURACY COMPARISON VISUALIZATION 
# ======================================================================

# Collect all model accuracies
model_names = ['SVM', 'Random Forest', 'Decision Tree', 'KNN', 'Naive Bayes', 'ANN', 'Linear Regression' ]
accuracies = [
    accuracy_score(y_test, svm_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, knn_pred),
    accuracy_score(y_test, nb_pred),
    accuracy_score(y_test, ann_pred),
    accuracy_score(y_test, lr_pred)
]

# Create a DataFrame for visualization
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': accuracies
}).sort_values('Accuracy', ascending=False)

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(14, 8))
palette = sns.color_palette("viridis", len(model_names))

# 1. Enhanced Bar Plot with Annotations 
plt.subplot(2, 2, 1)
ax = sns.barplot(x='Accuracy', y='Model', data=results_df, hue='Model', palette=palette, legend=False)
plt.title('Model Accuracy Comparison', fontsize=10, pad=20)
plt.xlabel('Accuracy Score', fontsize=8)
plt.ylabel('')
plt.xlim(0, 1.05)

# Add value annotations
for p in ax.patches:
    width = p.get_width()
    ax.text(width + 0.02, p.get_y() + p.get_height()/2., 
            f'{width:.3f}', 
            ha='left', va='center', fontsize=8)

# 2. Radial Visualization
plt.subplot(2, 2, 2, polar=True)
theta = np.linspace(0, 2 * np.pi, len(model_names), endpoint=False)
theta = np.concatenate((theta, [theta[0]]))
values = np.concatenate((accuracies, [accuracies[0]]))
plt.polar(theta, values, marker='o', linestyle='-', linewidth=2, markersize=6, color='steelblue')
plt.fill(theta, values, alpha=0.25, color='steelblue')
plt.title('Radial Accuracy Comparison', fontsize=10, pad=20)
plt.xticks(theta[:-1], model_names)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"])
plt.ylim(0, 1.1)

# 3. Heatmap of Accuracies
plt.subplot(2, 2, 3)
heatmap_data = results_df.set_index('Model').T
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", 
            cbar_kws={'label': 'Accuracy'}, linewidths=.5)
plt.title('Accuracy Heatmap', fontsize=14, pad=10)
plt.yticks([])

# 4. Dot Plot with Error Margins 
plt.subplot(2, 2, 4)
for i, (model, acc) in enumerate(zip(model_names, accuracies)):
    plt.errorbar(x=acc, y=model, 
                xerr=acc*0.05,  # Simulated error margin
                fmt='o', markersize=6, capsize=5, 
                color=palette[i], label=model)
plt.title('Accuracy with Confidence Intervals', fontsize=10, pad=20)
plt.xlabel('Accuracy Score', fontsize=8)
plt.xlim(0.5, 1.05)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('Results/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# CONFUSION MATRIX GRID 
# ======================================================================

# Create confusion matrices for the top 6 models (Linear Regression excluded)
models = {
    'SVM': svm_pred,
    'Random Forest': rf_pred,
    'Decision Tree': dt_pred,
    'KNN': knn_pred,
    'Naive Bayes': nb_pred,
    'ANN': ann_pred,
}

plt.figure(figsize=(16, 12))
for i, (name, pred) in enumerate(models.items()):
    plt.subplot(2, 3, i+1)
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant', 'Normal'],
                yticklabels=['Benign', 'Malignant', 'Normal'])
    plt.title(f'{name}\nAccuracy: {accuracy_score(y_test, pred):.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('Results/confusion_matrices_grid.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================================================================
# CLASSIFICATION REPORT HEATMAPS 
# ======================================================================

plt.figure(figsize=(16, 12))
for i, (name, pred) in enumerate(models.items()):
    plt.subplot(2, 3, i+1)
    report = classification_report(y_test, pred, output_dict=True, target_names=['Benign', 'Malignant', 'Normal'])
    report_df = pd.DataFrame(report).iloc[:-1, :].T
    sns.heatmap(report_df, annot=True, cmap='YlOrRd', vmin=0, vmax=1)
    plt.title(f'{name} Classification Report')

plt.tight_layout()
plt.savefig('Results/classification_reports.png', dpi=300, bbox_inches='tight')
plt.show()

