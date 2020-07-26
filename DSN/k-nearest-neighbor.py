import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

accuracies = []

for i in range(25):

    # read the data
    df =  pd.read_csv('../breast_cancer/breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)  # replace all missing values
    df.drop(['id'], 1, inplace=True)  # drop the ID from the data

    X = np.array(df.drop(['class'], 1))  # features -> Drop only the class
    y = np.array(df['class'])  # labels

    # model selection
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # classify the model
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    # test the model
    accuracy = clf.score(X_test, y_test)
    # print(accuracy)

    # # example data
    # example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
    # example_measures = example_measures.reshape(len(example_measures), -1)

    # # make a prediction based on the example_measures data
    # prediction = clf.predict(example_measures)
    # print(prediction)

    accuracies.append(accuracy)
print(sum(accuracies)/len(accuracies))
