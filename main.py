#Using Logistic Regression since we need to predict 1 and 0
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

google_rating = pd.read_csv("bill_authentication.csv")
google_rating_top = google_rating.head()

# Printing the top 5 lines of the csv file
print(google_rating_top)

X = google_rating.drop(columns=['Class'])
y = google_rating['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

dtmodel = LogisticRegression()
dtmodel.fit(X_train, y_train)

predictions = dtmodel.predict(X_test)

# Outputting the score
score = accuracy_score(y_test, predictions)
print("Being right:{} ".format(score))

print("Being wrong:{} ".format(1-score))
print(dtmodel.predict([[1213, -121, 3, -5.4]]))
