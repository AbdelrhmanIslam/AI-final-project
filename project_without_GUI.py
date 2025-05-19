from ucimlrepo import fetch_ucirepo

data = fetch_ucirepo(id=17)
X = data.data.features
y = data.data.targets

print(data.metadata)  
print(data.variables) 

df = X.copy()
df['Diagnosis'] = y
df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

print(f"dimensions of the data is : {df.shape}")
df.info()

df.describe()

# print(f"Missing values per column: {df.isnull().sum()}")
print(f"Missing values in all columns: {df.isnull().sum().sum()}")
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

from sklearn.model_selection import train_test_split

X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

import matplotlib.pyplot as plt
import seaborn as sns

train_df = X_train.copy()
train_df['Diagnosis'] = y_train

# Histogram
features_to_plot = ['radius1', 'texture1', 'perimeter1', 'area1']
sns.set(style="whitegrid", palette="pastel")

for feature in features_to_plot:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=train_df, x=feature, hue='Diagnosis', kde=True, bins=30)
    plt.title(f'Distribution of {feature} by Diagnosis')
    plt.tight_layout()
    plt.show()

# Boxplot
for feature in features_to_plot:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_df, x='Diagnosis', y=feature)
    plt.title(f'Boxplot of {feature} by Diagnosis')
    plt.tight_layout()
    plt.show()

# Correlation heatmap
plt.figure(figsize=(14, 12))
corr = train_df.drop(columns='Diagnosis').corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
print("Sample of scaled training data:")
print(X_train_scaled.head())

def triangular_membership(x, a, b, c):
    """
    function to calculate triangular membership function values.
    x: input value
    a, b, c: membership function parameters
    a->start point
    b->peak point
    c->end point
    """
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)


import numpy as np

# نختار الخصائص اللي هنشتغل عليها
features = ['radius1', 'area1', 'texture1']

# هنشتغل على بيانات التدريب فقط
fuzzy_train = X_train_scaled.copy()
fuzzy_test = X_test_scaled.copy()

for feature in features:
    min_val = fuzzy_train[feature].min()
    max_val = fuzzy_train[feature].max()
    mid_val = fuzzy_train[feature].median()
    
    # Low
    fuzzy_train[f'{feature}_low'] = triangular_membership(fuzzy_train[feature], min_val, min_val, mid_val)
    fuzzy_test[f'{feature}_low'] = triangular_membership(fuzzy_test[feature], min_val, min_val, mid_val)
    
    # Medium
    fuzzy_train[f'{feature}_medium'] = triangular_membership(fuzzy_train[feature], min_val, mid_val, max_val)
    fuzzy_test[f'{feature}_medium'] = triangular_membership(fuzzy_test[feature], min_val, mid_val, max_val)
    
    # High
    fuzzy_train[f'{feature}_high'] = triangular_membership(fuzzy_train[feature], mid_val, max_val, max_val)
    fuzzy_test[f'{feature}_high'] = triangular_membership(fuzzy_test[feature], mid_val, max_val, max_val)

# عرض عينة من البيانات بعد إضافة الخصائص الجديدة
print(fuzzy_train[[f'{f}_low' for f in features]].head())
print(fuzzy_train[[f'{f}_medium' for f in features]].head())
print(fuzzy_train[[f'{f}_high' for f in features]].head())

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train_final = fuzzy_train.copy()
X_test_final = fuzzy_test.copy()

current_depth = 1
best_depth = current_depth
best_accuracy = 0
accuracy_list = []

while True:
    model = DecisionTreeClassifier(max_depth=current_depth, random_state=42)
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_train_final)
    acc = accuracy_score(y_train, y_pred)
    accuracy_list.append((current_depth, acc))
    
    print(f"Depth: {current_depth}, Accuracy: {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_depth = current_depth
        current_depth += 1
    else:
        print("No improvement — stopping search.")
        break

X_train_final = fuzzy_train.copy()
X_test_final = fuzzy_test.copy()

current_depth = 1
best_depth_test = current_depth
best_accuracy_test = 0
accuracy_list_test = []

while True:
    model = DecisionTreeClassifier(max_depth=current_depth, random_state=42)
    model.fit(X_train_final, y_train)
    
    y_pred_test = model.predict(X_test_final)
    acc_test = accuracy_score(y_test, y_pred_test)
    accuracy_list_test.append((current_depth, acc_test))
    
    print(f"[TEST] Depth: {current_depth}, Accuracy: {acc_test:.4f}")
    
    if acc_test > best_accuracy_test:
        best_accuracy_test = acc_test
        best_depth_test = current_depth
        current_depth += 1
    else:
        print("No improvement — stopping search")
        break

grid_results = []

for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_train_final)
    acc = accuracy_score(y_train, y_pred)
    grid_results.append((depth, acc))
    print(f"Grid Search - Depth: {depth}, Accuracy: {acc:.4f}")

best_grid_depth, best_grid_acc = max(grid_results, key=lambda x: x[1])

print("\n=== Comparison Summary ===")
print(f"Hill Climbing Best Depth: {best_depth}, Accuracy: {best_accuracy:.4f}")
print(f"Grid Search Best Depth: {best_grid_depth}, Accuracy: {best_grid_acc:.4f}")
print(f"Total Hill Climbing Evaluations: {len(accuracy_list)}")
print(f"Total Grid Search Evaluations: {len(grid_results)}")

grid_results = []

for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42) 
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred)
    grid_results.append((depth, acc))
    print(f"Grid Search - Depth: {depth}, Test Accuracy: {acc:.4f}")

best_grid_depth, best_grid_acc = max(grid_results, key=lambda x: x[1])
print("\n=== Comparison Summary ===")
print(f"Hill Climbing Best Depth: {best_depth}, Accuracy: {best_accuracy:.4f}")
print(f"Grid Search Best Depth: {best_grid_depth}, Test Accuracy: {best_grid_acc:.4f}")
print(f"Total Hill Climbing Evaluations: {len(accuracy_list)}")
print(f"Total Grid Search Evaluations: {len(grid_results)}")

best_accuracy = 1
best_grid_acc = 1

accuracies = [best_accuracy * 100, best_grid_acc * 100]
methods = ['Hill-Climbing', 'Grid Search']

plt.figure(figsize=(7, 5))  
plt.bar(methods, accuracies, color=['#36A2EB', '#FF6384'], edgecolor='black')

plt.title('Training Accuracy: Hill-Climbing vs Grid Search', fontsize=16)
plt.xlabel('Optimization Method', fontsize=14)
plt.ylabel('Training Accuracy (%)', fontsize=14)

plt.ylim(0, 100)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

hill_evaluations = 9  
grid_evaluations = 10  

evaluations = [hill_evaluations, grid_evaluations]
methods = ['Hill-Climbing', 'Grid Search']

plt.figure(figsize=(8, 6)) 
plt.bar(methods, evaluations, color=['#36A2EB', '#FF6384'], edgecolor='black', width=0.45)

plt.title('Number of Evaluations: Hill-Climbing vs Grid Search', fontsize=16)
plt.xlabel('Optimization Method', fontsize=14)
plt.ylabel('Number of Evaluations', fontsize=14)

for i, v in enumerate(evaluations):
    plt.text(i, v + 0.2, f'{v}', ha='center', fontsize=12)

plt.ylim(0, 12) 

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_model.fit(X_train_final, y_train)

plt.figure(figsize=(20, 10))
plot_tree(final_model, feature_names=X_train_final.columns, class_names=["Benign", "Malignant"],
          filled=True, rounded=True, fontsize=10)
plt.title(f"Decision Tree (max_depth={best_depth})")
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# توقعات النموذج النهائي
y_test_pred = final_model.predict(X_test_final)

# 1. Accuracy
acc = accuracy_score(y_test, y_test_pred)

# 2. Precision & Recall (لكل فئة)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)

# 4. تقرير مفصل
report = classification_report(y_test, y_test_pred, target_names=["Benign", "Malignant"])

print("=== Final Evaluation on Test Set ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (Malignant class): {prec:.4f}")
print(f"Recall (Malignant class): {rec:.4f}")
print(f"\nConfusion Matrix: \n{cm}")
print(f"\nDetailed Classification Report: \n{report}")

cm = np.array([[67, 5], [3, 39]])
plt.figure(figsize=(12, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

benign_pred = 72     #* 67 (صحيح) + 5 (خطأ)
malignant_pred = 42  #* 3 (خطأ) + 39 (صحيح)

labels = ['Benign', 'Malignant']
sizes = [benign_pred, malignant_pred]
colors = ['#36A2EB', '#FF6384'] 

plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', 
        shadow=True, startangle=90, textprops={'fontsize': 14})

plt.title('Predicted Class Distribution on Test Set', fontsize=16)

plt.axis('equal')

plt.show()