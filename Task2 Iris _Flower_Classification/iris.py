import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


  # Load CSV file
data = pd.read_csv("iris.csv")
sns.pairplot(data, hue="species")
plt.show()

# Features and target
X = data.drop("species", axis=1)
y = data["species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

#Classification report
print("n\Classification Report:\n")
print(classification_report(y_test,y_pred))

# Predict one flower
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(sample)
print("Predicted Flower:", prediction[0])

import matplotlib.pyplot as plt

# Scatter plot
plt.scatter(
    X["sepal_length"],
    X["sepal_width"],
    c=pd.factorize(y)[0]
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset - Sepal Length vs Width")

plt.show()



plt.scatter(
    X["sepal_length"],
    X["sepal_width"],
    c=pd.factorize(y)[0]
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset - Sepal Length vs Width")

plt.show()