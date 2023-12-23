import pandas as pd
from sklearn.model_selection import train_test_split as split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import f1_score,recall_score,precision_score
from matplotlib import pyplot as plt
import seaborn as sns

def run_model(model, alg_name, X_train, X_test, y_train, y_test):
   # build the model on training data
   model.fit(X_train, y_train)
 
   # make predictions for test data
   y_pred = model.predict(X_test)
   # calculate the accuracy score
   ps=precision_score(y_test,y_pred)
   rs=recall_score(y_test,y_pred)
   f1 = f1_score(y_test,y_pred)
   accuracy =  accuracy_score(y_test, y_pred)
   cm = confusion_matrix(y_test, y_pred)
   # cross-validation scores
   scoresDT3 = cross_val_score(model, X_test, y_test, cv=6)
   Cr = classification_report(y_test, y_pred)
   results.append((alg_name, accuracy, model))
   print("Model: ", alg_name)
   print("Accuracy on Test Set for {} = {:.2f}\n".format(alg_name, accuracy))
   print('f1_score:',f1)
   print('precision:',ps)
   print('recall:',rs)
   print(Cr)
   print("{}: CrossVal Accuracy Mean: {:.2f} and Standard Deviation: {:.2f} \n".format(alg_name, scoresDT3.mean(), scoresDT3.std()))
   #joblib.dump(model, f"final_model2.pkl")

   ft_importance_values = model.feature_importances_

   ft_series = pd.Series(ft_importance_values, index = X_train.columns)
   ft_top20 = ft_series.sort_values(ascending=False)[:20]

   plt.figure(figsize=(8,6))
   plt.xlabel('Importance')
   plt.ylabel('Feature_Name')
   plt.title('Feature Importance Top 20')
   sns.barplot(x=ft_top20, y=ft_top20.index)
   plt.show()


if __name__ == "__main__":

   scale=False
   df = pd.read_csv('final_dataset.csv')
   data = df.copy()
   y = data['popularity_level']
   X = data.drop(columns=['popularity_level','popularity'])
   X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=42)
   
   results = []

   if  scale:
      ctr = ColumnTransformer([('minmax', MinMaxScaler(), ['duration_mins','tempo'])], remainder='passthrough')
      X_train_scaled = ctr.fit_transform(X_train)
      X_test_scaled = ctr.transform(X_test)

      '''
      model = DecisionTreeClassifier()
      run_model(model, "Decision Tree", X_train_scaled, X_test_scaled, y_train, y_test)

      #model = SVC(kernel='poly', degree=3, C=1)
      #run_model(model, "SVM Classifier", X_train_scaled, X_test_scaled, y_train, y_test)

      model = KNeighborsClassifier()
      run_model(model, "Nearest Neighbors Classifier", X_train_scaled, X_test_scaled, y_train, y_test)

      model = LogisticRegression(multi_class='multinomial' , solver='lbfgs', max_iter=100)
      run_model(model, "Logistic Regression", X_train_scaled, X_test_scaled, y_train, y_test)

      dt_b = DecisionTreeClassifier(max_depth=1, random_state=42)
      model = AdaBoostClassifier(base_estimator=dt_b)
      run_model(model, "Adaboost Classifier", X_train_scaled, X_test_scaled, y_train, y_test)
      '''
      model = RandomForestClassifier(n_estimators=10)
      run_model(model, "Random Forest", X_train_scaled, X_test_scaled, y_train, y_test)

   else:
      '''
      model = DecisionTreeClassifier()
      run_model(model, "Decision Tree",X_train, X_test, y_train, y_test)
      
      model = SVC(kernel='poly', degree=3, C=1)
      run_model(model, "SVM Classifier",X_train, X_test, y_train, y_test)
      
      model = KNeighborsClassifier()
      run_model(model, "Nearest Neighbors Classifier",X_train, X_test, y_train, y_test)

      model = LogisticRegression(multi_class='multinomial' , solver='lbfgs', max_iter=100)
      run_model(model, "Logistic Regression",X_train, X_test, y_train, y_test)

      dt_b = DecisionTreeClassifier(max_depth=1, random_state=42)
      model = AdaBoostClassifier(base_estimator=dt_b)
      run_model(model, "Adaboost Classifier",X_train, X_test, y_train, y_test)
      '''
      model = RandomForestClassifier(n_estimators=10)
      run_model(model, "Random Forest",X_train, X_test, y_train, y_test)