import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QTabWidget, QPushButton, QTableWidget, QTableWidgetItem, 
                             QLabel, QMessageBox, QComboBox, QSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

class BreastCancerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breast Cancer Classification")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #2E2E2E; color: #FFFFFF;")

        # Initialize data
        self.df = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.fuzzy_train = None
        self.fuzzy_test = None
        self.final_model = None
        self.best_depth = 8  
        self.accuracy_list = []
        self.grid_results = []

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; background: #3A3A3A; }
            QTabBar::tab { background: #3A3A3A; color: #FFF; padding: 10px; }
            QTabBar::tab:selected { background: #4CAF50; }
        """)
        main_layout.addWidget(self.tabs)

        # Tab 1: Data Exploration
        self.data_tab = QWidget()
        self.data_layout = QVBoxLayout(self.data_tab)
        self.tabs.addTab(self.data_tab, "Data Exploration")

        # Load Data Button
        load_btn = QPushButton("Load Dataset")
        load_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-size: 16px;")
        load_btn.clicked.connect(self.load_data)
        self.data_layout.addWidget(load_btn)

        # Data Info Label
        self.data_info = QLabel("Dataset not loaded.")
        self.data_info.setStyleSheet("font-size: 14px;")
        self.data_layout.addWidget(self.data_info)

        # Data Table
        self.data_table = QTableWidget()
        self.data_table.setStyleSheet("background-color: #3A3A3A; color: #FFF;")
        self.data_layout.addWidget(self.data_table)

        # Tab 2: Visualizations
        self.viz_tab = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_tab)
        self.tabs.addTab(self.viz_tab, "Visualizations")

        # Visualization Options
        viz_options = QHBoxLayout()
        self.viz_type = QComboBox()
        self.viz_type.addItems(["Histogram", "Boxplot", "Correlation Heatmap"])
        self.viz_type.setStyleSheet("background-color: #3A3A3A; color: #FFF; padding: 5px;")
        viz_options.addWidget(self.viz_type)

        self.feature_select = QComboBox()
        self.feature_select.addItems(["radius1", "texture1", "perimeter1", "area1"])
        self.feature_select.setStyleSheet("background-color: #3A3A3A; color: #FFF; padding: 5px;")
        viz_options.addWidget(self.feature_select)

        plot_btn = QPushButton("Plot")
        plot_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        plot_btn.clicked.connect(self.plot_visualization)
        viz_options.addWidget(plot_btn)
        self.viz_layout.addLayout(viz_options)

        # Matplotlib Canvas
        self.figure, self.ax = plt.subplots(figsize=(40, 20)) #TODO:
        self.canvas = FigureCanvas(self.figure)
        self.viz_layout.addWidget(self.canvas)

        # Tab 3: Model Training
        self.train_tab = QWidget()
        self.train_layout = QVBoxLayout(self.train_tab)
        self.tabs.addTab(self.train_tab, "Model Training")

        # Training Options
        train_options = QHBoxLayout()
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 20)
        self.depth_spin.setValue(self.best_depth)
        self.depth_spin.setStyleSheet("background-color: #3A3A3A; color: #FFF; padding: 5px;")
        train_options.addWidget(QLabel("Max Depth:"))
        train_options.addWidget(self.depth_spin)

        train_btn = QPushButton("Train Model")
        train_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        train_btn.clicked.connect(self.train_model)
        train_options.addWidget(train_btn)
        self.train_layout.addLayout(train_options)

        # Training Results
        self.train_results = QLabel("Model not trained.")
        self.train_results.setStyleSheet("font-size: 14px;")
        self.train_layout.addWidget(self.train_results)

        # Tab 4: Evaluation
        self.eval_tab = QWidget()
        self.eval_layout = QVBoxLayout(self.eval_tab)
        self.tabs.addTab(self.eval_tab, "Evaluation")

        # Evaluation Options
        eval_options = QHBoxLayout()
        self.eval_plot_type = QComboBox()
        self.eval_plot_type.addItems(["Confusion Matrix", "Pie Chart", "Decision Tree", 
                                     "Evaluations Bar", "Accuracy Bar"])
        self.eval_plot_type.setStyleSheet("background-color: #3A3A3A; color: #FFF; padding: 5px;")
        eval_options.addWidget(self.eval_plot_type)

        eval_btn = QPushButton("Evaluate & Plot")
        eval_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        eval_btn.clicked.connect(self.evaluate_model)
        eval_options.addWidget(eval_btn)
        self.eval_layout.addLayout(eval_options)

        # Evaluation Results
        self.eval_results = QLabel("Model not evaluated.")
        self.eval_results.setStyleSheet("font-size: 14px;")
        self.eval_layout.addWidget(self.eval_results)

        # Evaluation Canvas
        self.eval_figure, self.eval_ax = plt.subplots(figsize=(40, 20)) #TODO:
        self.eval_canvas = FigureCanvas(self.eval_figure)
        self.eval_layout.addWidget(self.eval_canvas)

    def load_data(self):
        try:
            # Fetch dataset
            data = fetch_ucirepo(id=17)
            X = data.data.features
            y = data.data.targets

            # Prepare DataFrame
            self.df = X.copy()
            self.df['Diagnosis'] = y
            self.df['Diagnosis'] = self.df['Diagnosis'].map({'M': 1, 'B': 0})

            # Display data info
            info = (f"Dimensions: {self.df.shape}\n"
                    f"Missing Values: {self.df.isnull().sum().sum()}\n"
                    f"Duplicates: {self.df.duplicated().sum()}")
            self.data_info.setText(info)

            # Populate table
            self.data_table.setRowCount(min(self.df.shape[0], 100))  # Limit rows for performance
            self.data_table.setColumnCount(self.df.shape[1])
            self.data_table.setHorizontalHeaderLabels(self.df.columns)
            for i in range(min(self.df.shape[0], 100)):
                for j in range(self.df.shape[1]):
                    self.data_table.setItem(i, j, QTableWidgetItem(str(self.df.iloc[i, j])))

            # Split and scale data
            X = self.df.drop(columns=['Diagnosis'])
            y = self.df['Diagnosis']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            scaler = MinMaxScaler()
            self.X_train_scaled = pd.DataFrame(scaler.fit_transform(self.X_train), 
                                              columns=self.X_train.columns, 
                                              index=self.X_train.index)
            self.X_test_scaled = pd.DataFrame(scaler.transform(self.X_test), 
                                             columns=self.X_test.columns, 
                                             index=self.X_test.index)

            # Fuzzy features
            self.fuzzy_train = self.X_train_scaled.copy()
            self.fuzzy_test = self.X_test_scaled.copy()
            features = ['radius1', 'area1', 'texture1']
            for feature in features:
                min_val = self.fuzzy_train[feature].min()
                max_val = self.fuzzy_train[feature].max()
                mid_val = self.fuzzy_train[feature].median()
                for dataset in [self.fuzzy_train, self.fuzzy_test]:
                    dataset[f'{feature}_low'] = self.triangular_membership(
                        dataset[feature], min_val, min_val, mid_val)
                    dataset[f'{feature}_medium'] = self.triangular_membership(
                        dataset[feature], min_val, mid_val, max_val)
                    dataset[f'{feature}_high'] = self.triangular_membership(
                        dataset[feature], mid_val, max_val, max_val)

            QMessageBox.information(self, "Success", "Dataset loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")

    def triangular_membership(self, x, a, b, c):
        return np.maximum(np.minimum((x - a) / (b - a + 1e-10), (c - x) / (c - b + 1e-10)), 0)

    def plot_visualization(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load the dataset first!")
            return

        self.ax.clear()
        train_df = self.X_train.copy()
        train_df['Diagnosis'] = self.y_train

        viz_type = self.viz_type.currentText()
        feature = self.feature_select.currentText()

        if viz_type == "Histogram":
            sns.histplot(data=train_df, x=feature, hue='Diagnosis', kde=True, bins=30, ax=self.ax)
            self.ax.set_title(f'Distribution of {feature} by Diagnosis')
        elif viz_type == "Boxplot":
            sns.boxplot(data=train_df, x='Diagnosis', y=feature, ax=self.ax)
            self.ax.set_title(f'Boxplot of {feature} by Diagnosis')
        else:  # Correlation Heatmap
            corr = train_df.drop(columns='Diagnosis').corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=self.ax)
            self.ax.set_title("Correlation Heatmap of Features")

        self.canvas.draw()

    def train_model(self):
        if self.fuzzy_train is None:
            QMessageBox.warning(self, "Warning", "Please load the dataset first!")
            return

        max_depth = self.depth_spin.value()
        self.final_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.final_model.fit(self.fuzzy_train, self.y_train)

        # Hill Climbing Search
        self.accuracy_list = []
        current_depth = 1
        best_accuracy = 0
        best_depth = current_depth
        while True:
            model = DecisionTreeClassifier(max_depth=current_depth, random_state=42)
            model.fit(self.fuzzy_train, self.y_train)
            y_pred = model.predict(self.fuzzy_train)
            acc = accuracy_score(self.y_train, y_pred)
            self.accuracy_list.append((current_depth, acc))
            if acc > best_accuracy:
                best_accuracy = acc
                best_depth = current_depth
                current_depth += 1
            else:
                break

        # Grid Search
        self.grid_results = []
        for depth in range(1, 11):
            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(self.fuzzy_train, self.y_train)
            y_pred = model.predict(self.fuzzy_train)
            acc = accuracy_score(self.y_train, y_pred)
            self.grid_results.append((depth, acc))

        # Evaluate on training set
        y_pred = self.final_model.predict(self.fuzzy_train)
        acc = accuracy_score(self.y_train, y_pred)

        self.train_results.setText(f"Training Accuracy: {acc:.4f}\n"
                                  f"Hill Climbing Best Depth: {best_depth}, Accuracy: {best_accuracy:.4f}\n"
                                  f"Grid Search Best Depth: {max(self.grid_results, key=lambda x: x[1])[0]}")
        QMessageBox.information(self, "Success", "Model trained successfully!")

    def evaluate_model(self):
        if self.final_model is None:
            QMessageBox.warning(self, "Warning", "Please train the model first!")
            return

        # Evaluate on test set
        y_test_pred = self.final_model.predict(self.fuzzy_test)
        acc = accuracy_score(self.y_test, y_test_pred)
        prec = precision_score(self.y_test, y_test_pred)
        rec = recall_score(self.y_test, y_test_pred)
        cm = confusion_matrix(self.y_test, y_test_pred)
        report = classification_report(self.y_test, y_test_pred, target_names=["Benign", "Malignant"])

        # Compute pie chart data
        benign_pred = np.sum(y_test_pred == 0)
        malignant_pred = np.sum(y_test_pred == 1)

        # Compute best accuracies
        best_accuracy = max([x[1] for x in self.accuracy_list]) if self.accuracy_list else 0
        best_grid_acc = max([x[1] for x in self.grid_results]) if self.grid_results else 0
        hill_evaluations = len(self.accuracy_list)
        grid_evaluations = len(self.grid_results)

        # Display results
        results = (f"Accuracy: {acc:.4f}\n"
                   f"Precision (Malignant): {prec:.4f}\n"
                   f"Recall (Malignant): {rec:.4f}\n\n"
                   f"Confusion Matrix:\n{cm}\n\n"
                   f"Classification Report:\n{report}")
        self.eval_results.setText(results)

        # Plot selected visualization
        self.eval_ax.clear()
        plot_type = self.eval_plot_type.currentText()

        if plot_type == "Confusion Matrix":
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Benign", "Malignant"], 
                        yticklabels=["Benign", "Malignant"], ax=self.eval_ax)
            self.eval_ax.set_title("Confusion Matrix")
            self.eval_ax.set_xlabel("Predicted")
            self.eval_ax.set_ylabel("Actual")
        elif plot_type == "Pie Chart":
            labels = ['Benign', 'Malignant']
            sizes = [benign_pred, malignant_pred]
            colors = ['#36A2EB', '#FF6384']
            self.eval_ax.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', 
                             shadow=True, startangle=90, textprops={'fontsize': 14})
            self.eval_ax.set_title('Predicted Class Distribution on Test Set')
            self.eval_ax.axis('equal')
        elif plot_type == "Decision Tree":
            plot_tree(self.final_model, feature_names=self.fuzzy_train.columns, 
                      class_names=["Benign", "Malignant"], filled=True, 
                      rounded=True, fontsize=8, ax=self.eval_ax)
            self.eval_ax.set_title(f"Decision Tree (max_depth={self.depth_spin.value()})")
        elif plot_type == "Evaluations Bar":
            evaluations = [hill_evaluations, grid_evaluations]
            methods = ['Hill-Climbing', 'Grid Search']
            bars = self.eval_ax.bar(methods, evaluations, color=['#36A2EB', '#FF6384'], 
                                    edgecolor='black', width=0.45)
            self.eval_ax.set_title('Number of Evaluations: Hill-Climbing vs Grid Search')
            self.eval_ax.set_xlabel('Optimization Method')
            self.eval_ax.set_ylabel('Number of Evaluations')
            self.eval_ax.set_ylim(0, 12)
            self.eval_ax.grid(axis='y', linestyle='--', alpha=0.7)
            for bar, v in zip(bars, evaluations):
                self.eval_ax.text(bar.get_x() + bar.get_width()/2, v + 0.2, f'{v}', 
                                  ha='center', fontsize=12)
        elif plot_type == "Accuracy Bar":
            accuracies = [best_accuracy * 100, best_grid_acc * 100]
            methods = ['Hill-Climbing', 'Grid Search']
            self.eval_ax.bar(methods, accuracies, color=['#36A2EB', '#FF6384'], 
                             edgecolor='black')
            self.eval_ax.set_title('Training Accuracy: Hill-Climbing vs Grid Search')
            self.eval_ax.set_xlabel('Optimization Method')
            self.eval_ax.set_ylabel('Training Accuracy (%)')
            self.eval_ax.set_ylim(0, 100)
            self.eval_ax.grid(axis='y', linestyle='--', alpha=0.7)

        self.eval_canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BreastCancerGUI()
    window.show()
    sys.exit(app.exec_())