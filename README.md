# 🤖 AI Project– Fuzzy Logic & Decision Tree Classifier

<img src="/img/final.png"  width="700"/>

## 🧠 Project Description

This project presents a complete machine learning pipeline using **Decision Trees** enhanced with **Fuzzy Logic-based features**. The goal is to explore intelligent preprocessing and optimization techniques to improve classification performance. All steps are implemented in Python using libraries like `pandas`, `scikit-learn`, `matplotlib`, and more.

> 📊 **Dataset Used:** *(e.g., Wine Quality or Breast Cancer Wisconsin)*

---

## 📁 Project Structure

```
project-folder/
├── data/              # Raw dataset files
├── images/            # Plots and visualizations
├── notebooks/         # Jupyter notebooks for each step
├── models/            # Saved models (optional)
├── README.md          # This file
└── main.py            # Main script (optional)
```

---

## ✅ Phase-by-Phase Implementation

### 1. 📥 Load Dataset and Train/Test Split

* Used `pandas` to load the dataset.
* Chose a dataset with at least **10 features**.
* Split into **80% training** and **20% testing** using `train_test_split` from `sklearn.model_selection`.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 2. 🧹 Clean Missing Values and Remove Duplicates

* Checked for null values using `df.isnull().sum()` and applied **mean/median imputation**.
* Removed duplicated rows using `df.drop_duplicates()`.

---

### 3. 📊 Exploratory Data Analysis (EDA)

* Visualized **feature distributions**, **correlation heatmap**, and detected **outliers**.

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(X_train['feature1'], kde=True)
```
 <img src="/img/09.png"  width="700"/>

---

### 4. 📏 Feature Scaling

* Applied **Min-Max Scaling** using `MinMaxScaler` from `sklearn.preprocessing`.
* Fitted on training set, then used the same scaler on test set.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 5. 🌫️ Create Fuzzy Features

* Selected 2–3 important numeric features.
* Defined **triangular membership functions** for `low`, `medium`, and `high`.
* Used these functions to compute fuzzy membership degrees.

```python
def triangular(x, a, b, c):
    return max(min((x-a)/(b-a), (c-x)/(c-b)), 0)
```

* Appended new fuzzy features to `X_train` and `X_test`.

<img src="/img/15.png"  width="700"/>

<img src="/img/16.png"  width="700"/>

<img src="/img/17.png"  width="700"/>

---

### 6. 🌲 Train Decision Tree & Optimize with Hill-Climbing

* Trained a `DecisionTreeClassifier` using fuzzy-augmented data.
* Tuned the `max_depth` using **Hill-Climbing**:

  * Started from depth = 1
  * Iteratively tested neighboring depths and selected the best
  * Stopped when no improvement in validation accuracy

---

### 7. 🔍 Compare to Grid Search

* Exhaustively evaluated `max_depth` values from 1 to 10.
* Compared with Hill-Climbing in terms of:

  * 🔁 Search Path
  * ⚙️ Efficiency (number of models evaluated)
  * 🎯 Final Accuracy

 <img src="/img/11.png"  width="700"/>

---

### 8. 🌳 Visualize Final Tree

* Used `sklearn.tree.plot_tree()` to visualize the tuned model.

```python
from sklearn.tree import plot_tree
plot_tree(clf, feature_names=X.columns, class_names=['Class 0', 'Class 1'])
```

 <img src="/img/12.png"  width="700"/>

---

### 9. 🧪 Final Evaluation on Test Set

* Reported metrics:

  * ✅ Accuracy
  * 🎯 Precision & Recall
  * 📉 Confusion Matrix

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```

 <img src="/img/13.png"  width="700"/>    

---

## 📈 Results Summary

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.94  |
| Precision | 0.93  |
| Recall    | 0.92  |

> ✅ Model improved after fuzzy augmentation and hyperparameter tuning.

---

## 🙋‍♂️ About Me

Hi! I'm a passionate machine learning and AI student. I built this project as part of my coursework to deepen my understanding of fuzzy logic and tree-based models.

* 🔗 [LinkedIn](https://www.linkedin.com/in/abdelrhman-islam)
* 🐙 [GitHub](https://github.com/AbdelrhmanIslam)

Feel free to connect and check out my other projects!

---

## 📌 How to Run

1. Clone the repository
2. Install required libraries

```bash
pip install -r requirements.txt
```

3. Open Jupyter Notebook or run `main.py`

---

## 📚 References

* scikit-learn documentation
* matplotlib & seaborn guides
* Fuzzy logic theory: Zadeh, 1965

---

## 📷 Other photos 

<img src="/img/01.png"  width="700"/>  
<img src="/img/02.png"  width="700"/>  
<img src="/img/03.png"  width="700"/>  
<img src="/img/04.png"  width="700"/>  
<img src="/img/05.png"  width="700"/>  
<img src="/img/06.png"  width="700"/>  
<img src="/img/07.png"  width="700"/>  
<img src="/img/08.png"  width="700"/>  
<img src="/img/10.png"  width="700"/>  
<img src="/img/14.png"  width="700"/>  
