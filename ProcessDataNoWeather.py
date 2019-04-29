import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pydot
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


file = open("ScoresData.txt", 'r')
scores = json.load(file)
file.close()

file2 = open("Conferences.txt", 'r')
conferences = json.load(file2)
file2.close()

file3 = open("DecisionTreeNoWeather.txt", 'w')

file4 = open("ConfusionMatrixNoWeather.txt", 'w')


gameinfo = []
unique_conf = set()


for item in scores.values():  #iterate through years
    for oth in item.values(): #iterate through weeks
        for thing in oth:     #iterate through the games that week
                for i in range(0,2): #data for each competitor
                    try:
                        gameinfolst = []
                        name = thing['competitions'][0]['competitors'][i]['team']['displayName']
                        gameinfolst.append(name)
                        gameinfolst.append(conferences[name])
                        unique_conf.add(conferences[name])
                        gameinfolst.append(thing['competitions'][0]['attendance'])
                        gameinfolst.append(thing['competitions'][0]['competitors'][i]['score'])
                        if thing['competitions'][0]['competitors'][i]['homeAway'] == 'home':
                            gameinfolst.append(1)
                        else:
                            gameinfolst.append(0)
                        if thing['competitions'][0]['competitors'][i]['winner'] == 0:
                            thing = 'Lose'
                        else:
                            thing = 'Win'
                        gameinfolst.append(thing)
                        gameinfo.append(gameinfolst)
                    except:
                        continue


print('hi')


MAXDEPTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
NUMFOLDS = 5

#Decision Tree information

data = pd.DataFrame(np.array(gameinfo), columns=['name', 'conference', 'attendance', 'score', 'home',  'winner'])

Y = data['winner']

X = data.drop('winner', axis=1)
X = X.drop('name', axis=1)
X = X.drop('conference', axis=1)

col_names = X.columns.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=1)


obj = SimpleImputer( strategy='mean')
obj.fit(X_train)

X_train = obj.transform(X_train)
X_test = obj.transform(X_test)

validationAcc = np.zeros(len(MAXDEPTHS))
testAcc = np.zeros(len(MAXDEPTHS))
confMat = [[] for i in range(len(MAXDEPTHS))]

index = 0

for depth in MAXDEPTHS:
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    scores = cross_val_score(clf, X_train, Y_train, cv=NUMFOLDS)
    validationAcc[index] = np.mean(scores)

    clf = clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    tree_acc = accuracy_score(Y_test, Y_pred)
    testAcc[index] = tree_acc

    cm = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])

    file4.write("Accuracy: Ind=" + str(index) + "\n")
    file4.write(str(cm))
    file4.write('\n\n')

    dot_data = tree.export_graphviz(clf, out_file=("DecisionTrees/NoWeather/decision_tree" + str(index) + ".dot"), feature_names=list(col_names), class_names=['Win', 'Lose'])
    (graph,) = pydot.graph_from_dot_file("DecisionTrees/NoWeather/decision_tree" + str(index) + ".dot")
    graph.write_png('DecisionTrees/NoWeather/decision_tree' + str(index) + '.png')
    index+=1

minimum = min(round(validationAcc.min(0), 1), round(testAcc.min(0), 1)) - .1
maximum = max(round(validationAcc.max(0), 1), round(testAcc.max(0), 1)) + .1


#plot how accurate the data is
plt.plot(MAXDEPTHS, validationAcc, 'ro--', MAXDEPTHS, testAcc, 'kv-')
plt.xlabel('Maximum depth')
plt.ylabel('Accuracy')
plt.title('Decision tree accuracy')
plt.legend(['Validation', 'Testing'])
plt.ylim([minimum, maximum])
plt.savefig('DecisionTrees/NoWeather/decision_tree_accuracy.png')



bestHyperparam = np.argmax(validationAcc)
file3.write('All Values\n')
file3.write('Best hyperparameter, maxdepth = ' + str(MAXDEPTHS[bestHyperparam]) + '\n')
file3.write('Test Accuracy = ' + str(testAcc[bestHyperparam]) + '\n')
file3.write('Data Shape = ' + str(data.shape) + '\n')
file3.write('\n\n')



#Decision Tree information
for item in unique_conf:

    data = pd.DataFrame(np.array(gameinfo), columns=['name', 'conference', 'attendance', 'score', 'home',  'winner'])
    data_set = data[data['conference']==item]

    Y = data_set['winner']
    X = data_set.drop('winner', axis=1)
    X = X.drop('name', axis=1)
    X = X.drop('conference', axis=1)

    col_names = X.columns.values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=1)


    obj = SimpleImputer( strategy='mean')
    obj.fit(X_train)

    X_train = obj.transform(X_train)
    X_test = obj.transform(X_test)

    validationAcc = np.zeros(len(MAXDEPTHS))
    testAcc = np.zeros(len(MAXDEPTHS))
    confMat = [[] for i in range(len(MAXDEPTHS))]

    index = 0

    for depth in MAXDEPTHS:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        scores = cross_val_score(clf, X_train, Y_train, cv=NUMFOLDS)
        validationAcc[index] = np.mean(scores)

        clf = clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        tree_acc = accuracy_score(Y_test, Y_pred)
        testAcc[index] = tree_acc

        cm = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])

        file4.write("Accuracy: Ind=" + str(index) + "\n")
        file4.write(str(cm))
        file4.write('\n\n')

        dot_data = tree.export_graphviz(clf, out_file=("DecisionTrees/NoWeather/" + item + "decision_tree" + str(index) + ".dot"), feature_names=list(col_names), class_names=['Win', 'Lose'])
        (graph,) = pydot.graph_from_dot_file("DecisionTrees/NoWeather/" + item + "decision_tree" + str(index) + ".dot")
        graph.write_png('DecisionTrees/NoWeather/' + item + 'decision_tree' + str(index) + '.png')
        index+=1

    minimum = min(round(validationAcc.min(0), 1), round(testAcc.min(0), 1)) - .1
    maximum = max(round(validationAcc.max(0), 1), round(testAcc.max(0), 1)) + .1


    #plot how accurate the data is
    plt.plot(MAXDEPTHS, validationAcc, 'ro--', MAXDEPTHS, testAcc, 'kv-')
    plt.xlabel('Maximum depth')
    plt.ylabel('Accuracy')
    plt.title('Decision tree accuracy')
    plt.legend(['Validation', 'Testing'])
    plt.ylim([minimum, maximum])
    plt.savefig('DecisionTrees/NoWeather/' + item + 'decision_tree_accuracy.png')



    bestHyperparam = np.argmax(validationAcc)
    file3.write('All Values - ' + item + '\n')
    file3.write('Best hyperparameter, maxdepth = ' + str(MAXDEPTHS[bestHyperparam]) + '\n')
    file3.write('Test Accuracy = ' + str(testAcc[bestHyperparam]) + '\n')
    file3.write('Data Shape = ' + str(data.shape) + '\n')
    file3.write('\n\n')



file3.close()
file4.close()