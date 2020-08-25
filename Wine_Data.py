import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


wine_data = pd.read_excel('C:/Users/mowab/Downloads/Wine Data.xlsx')
print(wine_data.head(6))

sns.countplot(wine_data['CLASS'])
plt.show()

sns.pairplot(wine_data, hue='CLASS')
plt.show()

x = wine_data.iloc[:, 1:14].values
y = wine_data.iloc[:, 0].values

print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', degree=3)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)
print("\n CONFUSION MATRIX ")
print(cm)

plt.title("HEATMAP OF THE CONFUSION MATRIX")
sns.heatmap(cm)
plt.show()

score = accuracy_score(y_test, y_pred)
print(score)
print('CLASSIFICATION REPORT')
clas_rep = classification_report(y_test, y_pred)
print(clas_rep)


