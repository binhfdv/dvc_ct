import joblib
from pathlib import Path
import pandas as pd

# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

repo_path = Path(__file__).parent.parent
training_set = pd.read_csv(repo_path / 'data/prepared/training_set.csv')
training_set = training_set.fillna(0)
X = training_set[['Pclass','Age','SibSp']]
y = training_set['Survived']

clf = SVC()
clf.fit(X, y)
joblib.dump(clf, repo_path / "models/model.joblib")