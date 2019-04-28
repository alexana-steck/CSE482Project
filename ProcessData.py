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
import plotly.plotly as plotly
import plotly.graph_objs as go

file = open("ScoresData.txt", 'r')
scores = json.load(file)
file.close()

file1 = open("LocationData.txt", 'r')
locations_dates_times = json.load(file1)
file1.close()

file2 = open("WeatherData.txt", 'r')
weather_data = json.load(file2)
file2.close()

file3 = open("TeamLoc.txt", 'r')
venue_locations = json.load(file3)
file3.close()

file5 = open("Conferences.txt", 'r')
conferences = json.load(file5)
file5.close()

file4 = open("DecisionTree.txt", 'w')

file6 = open("ConfusionMatrix.txt", 'w')

gameinfo = []
unique_conf = set()

for item in scores.values():  #iterate through years
    for oth in item.values(): #iterate through weeks
        for thing in oth:     #iterate through the games that week
                for i in range(0,2): #data for each competitor
                    try:
                        gameinfolst = []
                        date = thing['competitions'][0]['date'][:10]
                        if 'venue' in thing['competitions'][0]:
                            city = thing['competitions'][0]['venue']['address']['city']
                            state = thing['competitions'][0]['venue']['address']['state']
                        elif 'competitors' in thing['competitions'][0] and len(
                                thing['competitions'][0]['competitors']) == 2:
                            for x in thing['competitions'][0]['competitors']:
                                if thing['homeAway'] == 'home':
                                    loc = x['team']['location']
                                    if loc in venue_locations:
                                        city = venue_locations[loc][0]
                                        state = venue_locations[loc][1]

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

                        if "Precipitation" in weather_data[state][city][date]:
                            gameinfolst.append(weather_data[state][city][date]['Precipitation'])
                        else:
                            gameinfolst.append(0)
                        if "Snowfall" in weather_data[state][city][date]:
                            gameinfolst.append(weather_data[state][city][date]['Snowfall'])
                        else:
                            gameinfolst.append(0)
                        if "Snow Depth" in weather_data[state][city][date]:
                            gameinfolst.append(weather_data[state][city][date]['Snow Depth'])
                        else:
                            gameinfolst.append(0)
                        if "Maximum Temperature" in weather_data[state][city][date]:
                            gameinfolst.append(weather_data[state][city][date]['Maximum Temperature'])
                        else:
                            gameinfolst.append(None)
                        if "Minimum Temperature" in weather_data[state][city][date]:
                            gameinfolst.append(weather_data[state][city][date]['Minimum Temperature'])
                        else:
                            gameinfolst.append(None)
                        if "Average wind speed" in weather_data[state][city][date]:
                            gameinfolst.append(weather_data[state][city][date]['Average wind speed'])
                        else:
                            gameinfolst.append(None)

                        if thing['competitions'][0]['competitors'][i]['winner'] == 0:
                            thing = 'Lose'
                        else:
                            thing = 'Win'
                        gameinfolst.append(thing)
                        gameinfo.append(gameinfolst)

                    except:
                        continue


MAXDEPTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
NUMFOLDS = 5
NUMFOLDSCONF = 3

#Decision Tree information

data = pd.DataFrame(np.array(gameinfo), columns=['name', 'conference', 'attendance', 'score', 'home', 'precipitation', 'snowfall', 'snow depth', 'max temp', 'min temp', 'avg wind speed', 'winner'])

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

    file6.write("Accuracy: Ind=" + str(index) + "\n")
    file6.write(str(cm))
    file6.write('\n\n')

    dot_data = tree.export_graphviz(clf, out_file=("DecisionTrees/decision_tree_with_score" + str(index) + ".dot"), feature_names=list(col_names), class_names=['Win', 'Lose'])
    (graph,) = pydot.graph_from_dot_file("DecisionTrees/decision_tree_with_score" + str(index) + ".dot")
    graph.write_png('DecisionTrees/decision_tree_with_score' + str(index) + '.png')
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
plt.savefig('DecisionTrees/decision_tree_accuracy_with_score.png')



bestHyperparam = np.argmax(validationAcc)
file4.write('All Values\n')
file4.write('Best hyperparameter, maxdepth = ' + str(MAXDEPTHS[bestHyperparam]) + '\n')
file4.write('Test Accuracy = ' + str(testAcc[bestHyperparam]) + '\n')
file4.write('Data Shape = ' + str(data.shape) + '\n')
file4.write('\n\n')



#by conference
for item in unique_conf:
    try:
        data = pd.DataFrame(np.array(gameinfo),
                            columns=['name', 'conference', 'attendance', 'score', 'home', 'precipitation', 'snowfall', 'snow depth',
                                     'max temp', 'min temp', 'avg wind speed', 'winner'])
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
            scores = cross_val_score(clf, X_train, Y_train, cv=NUMFOLDSCONF)
            validationAcc[index] = np.mean(scores)

            clf = clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            tree_acc = accuracy_score(Y_test, Y_pred)
            testAcc[index] = tree_acc
            cm = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])

            file6.write("Accuracy - " + item + ": Ind=" + str(index) + "\n")
            file6.write(str(cm))
            file6.write('\n\n')

            dot_data = tree.export_graphviz(clf, out_file=("DecisionTrees/" + item +"decision_tree_with_score" + str(index) + ".dot"), feature_names=list(col_names), class_names=['Win', 'Lose'])
            (graph,) = pydot.graph_from_dot_file("DecisionTrees/" + item +"decision_tree_with_score" + str(index) + ".dot")
            graph.write_png('DecisionTrees/' + item + 'decision_tree_with_score' + str(index) + '.png')
            index+=1

        minimum = min(round(validationAcc.min(0), 1), round(testAcc.min(0), 1)) - .1
        maximum = max(round(validationAcc.max(0), 1), round(testAcc.max(0), 1)) + .1

        #plot how accurate the data is
        plt.clf()
        plt.plot(MAXDEPTHS, validationAcc, 'ro--', MAXDEPTHS, testAcc, 'kv-')
        plt.xlabel('Maximum depth')
        plt.ylabel('Accuracy')
        plt.title('Decision tree accuracy for ' + item)
        plt.legend(['Validation', 'Testing'])
        plt.ylim([minimum, maximum])
        plt.savefig('DecisionTrees/' + item + 'decision_tree_accuracy_with_score.png')



        bestHyperparam = np.argmax(validationAcc)
        file4.write('All Values ' + item + '\n')
        file4.write('Best hyperparameter, maxdepth = ' + str(MAXDEPTHS[bestHyperparam]) + '\n')
        file4.write('Test Accuracy = ' + str(testAcc[bestHyperparam]) + '\n')
        file4.write('Data Shape = ' + str(data_set.shape) + '\n')
        file4.write('\n\n')
    except:
        file4.write('All Values ' + item + '\n')
        file4.write('No Data\n')
        file4.write('\n\n')
        continue

#Decision Tree information no score

data = pd.DataFrame(np.array(gameinfo), columns=['name', 'conference', 'attendance', 'score', 'home', 'precipitation', 'snowfall', 'snow depth', 'max temp', 'min temp', 'avg wind speed', 'winner'])

Y = data['winner']
X = data.drop('winner', axis=1)
X = X.drop('name', axis=1)
X = X.drop('score', axis=1)
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
    confMat[index] = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])

    cm = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])

    file6.write("Accuracy, no score" + ": Ind=" + str(index) + "\n")
    file6.write(str(cm))
    file6.write('\n\n')

    dot_data = tree.export_graphviz(clf, out_file=("DecisionTrees/decision_tree_no_score" + str(index) + ".dot"), feature_names=col_names, class_names=['Win', 'Lose'])
    (graph,) = pydot.graph_from_dot_file("DecisionTrees/decision_tree_no_score" + str(index) + ".dot")
    graph.write_png('DecisionTrees/decision_tree_no_score' + str(index) + '.png')
    index+=1

minimum = min(round(validationAcc.min(0), 1), round(testAcc.min(0), 1)) - .1
maximum = max(round(validationAcc.max(0), 1), round(testAcc.max(0), 1)) + .1

#plot how accurate the data is
plt.clf()
plt.plot(MAXDEPTHS, validationAcc, 'ro--', MAXDEPTHS, testAcc, 'kv-')
plt.xlabel('Maximum depth')
plt.ylabel('Accuracy')
plt.title('Decision tree accuracy, no score')
plt.legend(['Validation', 'Testing'])
plt.ylim([minimum, maximum])
plt.savefig('DecisionTrees/decision_tree_accuracy_no_score.png')

bestHyperparam = np.argmax(validationAcc)
file4.write('No Score Values\n')
file4.write('Best hyperparameter, maxdepth = ' + str(MAXDEPTHS[bestHyperparam]) + '\n')
file4.write('Test Accuracy = ' + str(testAcc[bestHyperparam]) + '\n')
file4.write('Data Shape = ' + str(data.shape) + '\n')
file4.write('\n\n')



#by conference
for item in unique_conf:
    try:
        data = pd.DataFrame(np.array(gameinfo),
                            columns=['name', 'conference', 'attendance', 'score', 'home', 'precipitation', 'snowfall', 'snow depth',
                                     'max temp', 'min temp', 'avg wind speed', 'winner'])
        data_set = data[data['conference']==item]
        Y = data_set['winner']
        X = data_set.drop('winner', axis=1)
        X = X.drop('name', axis=1)
        X = X.drop('score', axis=1)
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
            scores = cross_val_score(clf, X_train, Y_train, cv=NUMFOLDSCONF)
            validationAcc[index] = np.mean(scores)

            clf = clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            tree_acc = accuracy_score(Y_test, Y_pred)
            testAcc[index] = tree_acc
            cm = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])


            file6.write("Accuracy, no score - " + item + ": Ind=" + str(index) + "\n")
            file6.write(str(cm))
            file6.write('\n\n')


            dot_data = tree.export_graphviz(clf, out_file=("DecisionTrees/" + item +"decision_tree_no_score" + str(index) + ".dot"), feature_names=list(col_names), class_names=['Win', 'Lose'])
            (graph,) = pydot.graph_from_dot_file("DecisionTrees/" + item +"decision_tree_no_score" + str(index) + ".dot")
            graph.write_png('DecisionTrees/' + item + 'decision_tree_no_score' + str(index) + '.png')
            index+=1

        minimum = min(round(validationAcc.min(0), 1), round(testAcc.min(0), 1)) - .1
        maximum = max(round(validationAcc.max(0), 1), round(testAcc.max(0), 1)) + .1

        #plot how accurate the data is
        plt.clf()
        plt.plot(MAXDEPTHS, validationAcc, 'ro--', MAXDEPTHS, testAcc, 'kv-')
        plt.xlabel('Maximum depth')
        plt.ylabel('Accuracy')
        plt.title('Decision tree accuracy for ' + item + ', no score')
        plt.legend(['Validation', 'Testing'])
        plt.ylim([minimum, maximum])
        plt.savefig('DecisionTrees/' + item + 'decision_tree_accuracy_no_score.png')



        bestHyperparam = np.argmax(validationAcc)
        file4.write('No Scores ' + item + '\n')
        file4.write('Best hyperparameter, maxdepth = ' + str(MAXDEPTHS[bestHyperparam]) + '\n')
        file4.write('Test Accuracy = ' + str(testAcc[bestHyperparam]) + '\n')
        file4.write('Data Shape = ' + str(data_set.shape) + '\n')
        file4.write('\n\n')
    except:
        file4.write('No Scores ' + item + '\n')
        file4.write('No Data\n')
        file4.write('\n\n')
        continue


#Decision Tree information no score or attendance

data = pd.DataFrame(np.array(gameinfo), columns=['name', 'conference', 'attendance', 'score', 'home', 'precipitation', 'snowfall', 'snow depth', 'max temp', 'min temp', 'avg wind speed', 'winner'])

Y = data['winner']
X = data.drop('winner', axis=1)
X = X.drop('name', axis=1)
X = X.drop('score', axis=1)
X = X.drop('conference', axis=1)
X = X.drop('attendance', axis=1)

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
    confMat[index] = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])

    cm = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])

    file6.write("Accuracy, no score or attendance: Ind=" + str(index) + "\n")
    file6.write(str(cm))
    file6.write('\n\n')

    dot_data = tree.export_graphviz(clf, out_file=("DecisionTrees/decision_tree_no_score_or_attendance" + str(index) + ".dot"), feature_names=col_names, class_names=['Win', 'Lose'])
    (graph,) = pydot.graph_from_dot_file("DecisionTrees/decision_tree_no_score_or_attendance" + str(index) + ".dot")
    graph.write_png('DecisionTrees/decision_tree_no_score_or_attendance' + str(index) + '.png')
    index+=1

minimum = min(round(validationAcc.min(0), 1), round(testAcc.min(0), 1)) - .1
maximum = max(round(validationAcc.max(0), 1), round(testAcc.max(0), 1)) + .1

#plot how accurate the data is
plt.clf()
plt.plot(MAXDEPTHS, validationAcc, 'ro--', MAXDEPTHS, testAcc, 'kv-')
plt.xlabel('Maximum depth')
plt.ylabel('Accuracy')
plt.title('Decision tree accuracy, no score, no attendance')
plt.legend(['Validation', 'Testing'])
plt.ylim([minimum, maximum])
plt.savefig('DecisionTrees/decision_tree_accuracy_no_score_or_attendance.png')

bestHyperparam = np.argmax(validationAcc)
file4.write('No Score or Attendance Values\n')
file4.write('Best hyperparameter, maxdepth = ' + str(MAXDEPTHS[bestHyperparam]) + '\n')
file4.write('Test Accuracy = ' + str(testAcc[bestHyperparam]) + '\n')
file4.write('Data Shape = ' + str(data.shape) + '\n')
file4.write('\n\n')



#by conference
for item in unique_conf:
    try:
        data = pd.DataFrame(np.array(gameinfo),
                            columns=['name', 'conference', 'attendance', 'score', 'home', 'precipitation', 'snowfall', 'snow depth',
                                     'max temp', 'min temp', 'avg wind speed', 'winner'])
        data_set = data[data['conference']==item]
        Y = data_set['winner']
        X = data_set.drop('winner', axis=1)
        X = X.drop('name', axis=1)
        X = X.drop('score', axis=1)
        X = X.drop('conference', axis=1)
        X = X.drop('attendance', axis=1)

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
            scores = cross_val_score(clf, X_train, Y_train, cv=NUMFOLDSCONF)
            validationAcc[index] = np.mean(scores)

            clf = clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            tree_acc = accuracy_score(Y_test, Y_pred)
            testAcc[index] = tree_acc
            cm = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])


            file6.write("Accuracy, no score or attendance - " + item + ": Ind=" + str(index) + "\n")
            file6.write(str(cm))
            file6.write('\n\n')


            dot_data = tree.export_graphviz(clf, out_file=("DecisionTrees/" + item +"decision_tree_no_score_or_attendance" + str(index) + ".dot"), feature_names=list(col_names), class_names=['Win', 'Lose'])
            (graph,) = pydot.graph_from_dot_file("DecisionTrees/" + item +"decision_tree_no_score_or_attendance" + str(index) + ".dot")
            graph.write_png('DecisionTrees/' + item + 'decision_tree_no_score_or_attendance' + str(index) + '.png')
            index+=1

        minimum = min(round(validationAcc.min(0), 1), round(testAcc.min(0), 1)) - .1
        maximum = max(round(validationAcc.max(0), 1), round(testAcc.max(0), 1)) + .1

        #plot how accurate the data is
        plt.clf()
        plt.plot(MAXDEPTHS, validationAcc, 'ro--', MAXDEPTHS, testAcc, 'kv-')
        plt.xlabel('Maximum depth')
        plt.ylabel('Accuracy')
        plt.title('Decision tree accuracy for ' + str(item) + ', no score, no attendance')
        plt.legend(['Validation', 'Testing'])
        plt.ylim([minimum, maximum])
        plt.savefig('DecisionTrees/' + item + 'decision_tree_accuracy_no_score_or_attendance.png')



        bestHyperparam = np.argmax(validationAcc)
        file4.write('No Scores or Attendance ' + item + '\n')
        file4.write('Best hyperparameter, maxdepth = ' + str(MAXDEPTHS[bestHyperparam]) + '\n')
        file4.write('Test Accuracy = ' + str(testAcc[bestHyperparam]) + '\n')
        file4.write('Data Shape = ' + str(data_set.shape) + '\n')
        file4.write('\n\n')
    except:
        file4.write('No Scores ' + item + '\n')
        file4.write('No Data\n')
        file4.write('\n\n')
        continue


#Decision Tree information no score or attendance

data = pd.DataFrame(np.array(gameinfo), columns=['name', 'conference', 'attendance', 'score', 'home', 'precipitation', 'snowfall', 'snow depth', 'max temp', 'min temp', 'avg wind speed', 'winner'])

Y = data['winner']
X = data.drop('winner', axis=1)
X = X.drop('name', axis=1)
X = X.drop('score', axis=1)
X = X.drop('conference', axis=1)
X = X.drop('attendance', axis=1)

col_names = X.columns.values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, random_state=1)


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
    confMat[index] = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])

    cm = confusion_matrix(Y_test, Y_pred, labels=['Win', 'Lose'])

    file6.write("Accuracy, no score, attendance, bigger validation set: Ind=" + str(index) + "\n")
    file6.write(str(cm))
    file6.write('\n\n')

    dot_data = tree.export_graphviz(clf, out_file=("DecisionTrees/decision_tree_no_score_or_attendance_test_60" + str(index) + ".dot"), feature_names=col_names, class_names=['Win', 'Lose'])
    (graph,) = pydot.graph_from_dot_file("DecisionTrees/decision_tree_no_score_or_attendance_test_60" + str(index) + ".dot")
    graph.write_png('DecisionTrees/decision_tree_no_score_or_attendance_test_60' + str(index) + '.png')
    index+=1

minimum = min(round(validationAcc.min(0), 1), round(testAcc.min(0), 1)) - .1
maximum = max(round(validationAcc.max(0), 1), round(testAcc.max(0), 1)) + .1

#plot how accurate the data is
plt.clf()
plt.plot(MAXDEPTHS, validationAcc, 'ro--', MAXDEPTHS, testAcc, 'kv-')
plt.xlabel('Maximum depth')
plt.ylabel('Accuracy')
plt.title('Decision tree accuracy, no score, no attendance, smaller test set')
plt.legend(['Validation', 'Testing'])
plt.ylim([minimum, maximum])
plt.savefig('DecisionTrees/decision_tree_accuracy_no_score_or_attendance_test_60.png')

bestHyperparam = np.argmax(validationAcc)
file4.write('No Score or Attendance Values, bigger validation set\n')
file4.write('Best hyperparameter, maxdepth = ' + str(MAXDEPTHS[bestHyperparam]) + '\n')
file4.write('Test Accuracy = ' + str(testAcc[bestHyperparam]) + '\n')
file4.write('Data Shape = ' + str(data.shape) + '\n')
file4.write('\n\n')


file4.close()

file6.close()