# Heart Disease Classification

This project uses machine learning to classify whether a patient is likely to have heart disease from clinical measurements in the UCI Heart Disease dataset. The analysis is implemented in a Jupyter Notebook and compares three supervised learning models:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

The target is converted to a binary label:

- `0`: no diagnosed heart disease
- `1`: diagnosed heart disease

> **Important:** This project is for educational and research purposes only. It is not a medical diagnostic tool and must not be used to make healthcare decisions.

## Repository contents

| File | Description |
| --- | --- |
| `Classification_of_Heart_Disease_Using_the_Heart_Disease_Dataset.ipynb` | Notebook containing data preparation, model training, hyperparameter tuning, and evaluation |
| `ML_project.pdf` | Project report |
| `heart_disease_uci.csv` | Input dataset required by the notebook; it is not currently included in this repository |

## Dataset

The notebook expects a file named `heart_disease_uci.csv` in the project directory. It uses the Cleveland subset and the following 13 features:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalch`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`

The original target column is `num`. It is transformed so that `0` represents no disease and every non-zero value represents disease. The dataset can be obtained from the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

## How to run

1. Clone the repository:

   ```bash
   git clone https://github.com/Nubah3/ML_Project.git
   cd ML_Project
   ```

2. Install the Python dependencies:

   ```bash
   pip install pandas scikit-learn jupyter
   ```

3. Place `heart_disease_uci.csv` in the project directory.

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `Classification_of_Heart_Disease_Using_the_Heart_Disease_Dataset.ipynb` and run the cells in order.

## Methodology

The notebook:

1. Loads the dataset and selects the Cleveland records.
2. Selects the clinical features and converts the target into a binary label.
3. Fills missing numeric values with their column means and missing categorical values with their column modes.
4. Encodes categorical variables with `LabelEncoder`.
5. Splits the data into training and test sets using an 80/20 stratified split.
6. Standardizes the features with `StandardScaler`.
7. Trains Logistic Regression, Random Forest, and SVM classifiers.
8. Uses five-fold `GridSearchCV` tuning for the Random Forest and SVM models.
9. Compares the models using test-set accuracy.

## Recorded results

The results below are the outputs saved in the notebook. They may change if the dataset, preprocessing, library versions, or random seed changes.

| Model | Test accuracy |
| --- | ---: |
| Logistic Regression | 0.8852 |
| Random Forest | 0.9180 |
| SVM | 0.9344 |

In the recorded run, SVM achieved the highest test accuracy. The selected Random Forest parameters were `n_estimators=200`, `max_depth=20`, `min_samples_split=2`, and `class_weight=None`. The selected SVM parameters were `C=100`, `kernel=linear`, `gamma=scale`, and `class_weight=None`.

## Limitations

- The evaluation uses a relatively small subset of the full dataset.
- Accuracy alone does not capture false positives, false negatives, precision, recall, or clinical usefulness.
- Label encoding categorical variables imposes numeric ordering that may not represent the underlying medical categories.
- The dataset and model are not validated for use with real-world clinical data.
