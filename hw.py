import numpy as np

import pandas as pd

import re as re

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import Imputer

# read data

amazon = pd.read_csv("amazon.csv")

rotten_tomatoes = pd.read_csv("rotten_tomatoes.csv", encoding="ISO-8859-1")

train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

rotten_selected = rotten_tomatoes.iloc[:, :10]

## amazon data processing

# deal with time period

amazon = amazon.drop('cost', 1)

for i in range(amazon.shape[0]):

    starcol = str(amazon.loc[i]['star'])

    if bool(re.search(r'\d', starcol)):
        amazon.set_value(i, 'time', starcol)

        amazon.set_value(i, 'star', np.nan)  # cope with time column

    timelong = str(amazon.loc[i]['time'])

    if timelong.find('/') != -1:

        amazon.set_value(i, 'time', np.nan)

    else:

        h = int(timelong.find('hour'))

        mins = int(timelong.find('minute'))

        comma = int(timelong.find(','))

        if h != -1:

            hours = int(timelong[h - 2])

            minutes = int(timelong[comma + 2:mins - 1])

        elif mins != -1:

            hours = 0

            minutes = int(timelong[:mins - 1])

        else:

            amazon.set_value(i, 'time', np.nan)

        amazon.set_value(i, 'time', hours * 60 + minutes)

# deal with star names

for i in range(amazon.shape[0]):

    # deal with star

    starcol = str(amazon.loc[i]['star'])

    if starcol.find('/') == -1:

        starcol = set(starcol.split(', '))

    else:

        starcol = set(starcol.split('/'))

    amazon.set_value(i, 'star', starcol)

amazon_selected = amazon
## rotten tomatoes data cleansing

# deal with time

prob = []

for index, row in rotten_selected.iterrows():

    timelong = row['time']

    if str(timelong) != 'nan':

        m = re.search(r'(\d+) hr. (\d+) min.', timelong)

        m1 = re.search(r'(\d+) hr.', timelong)

        m2 = re.search(r'(\d+) min.', timelong)

        if m:

            time_tup = m.groups()

            minute = int(time_tup[0]) * 60 + int(time_tup[1])

        elif m1:

            time_tup = m1.groups()

            minute = int(time_tup[0]) * 60

        elif m2:

            time_tup = m2.groups()

            minute = int(time_tup[0])

        else:

            prob.append(index)

        rotten_selected.set_value(index, 'time', minute)

rotten_selected = rotten_selected.drop('year', 1)

# combine star1 to star6 into a column of sets star

rotten_selected['star'] = [set() for x in range(len(rotten_selected.index))]

for index, row in rotten_selected.iterrows():

    star = set()

    for i in range(3, 9):

        if str(row[i]) != 'nan':
            star.add(row[i])

    rotten_selected.set_value(index, 'star', star)

rotten_selected = rotten_selected.iloc[:, [0, 1, 2, 9]]

# transform training set

import distance


def transformation(X):
    """

    transform train into standard format

    """

    X['time_diff'] = 0

    X['director_diff'] = 0

    X['star_diff'] = 0

    for i in range(X.shape[0]):
        id_amazon = X.iloc[i, 0]

        id_rotten = X.iloc[i, 1]

        row_amazon = amazon_selected[amazon_selected['id'] == id_amazon]

        row_rotten = rotten_selected[rotten_selected['id'] == id_rotten]

        X.iloc[i, 3] = np.absolute(float(row_amazon['time']) - float(row_rotten['time']))

        X.iloc[i, 4] = distance.levenshtein(str(row_amazon['director']), str(row_rotten['director']), normalized=True)

        # the intersection of star names

        amazon_star = list(row_amazon['star'])[0]

        rotten_star = list(row_rotten['star'])[0]

        X.iloc[i, 5] = len(amazon_star.intersection(rotten_star))

    X_train = X.iloc[:, [0, 1, 3, 4, 5]]

    y_train = X.iloc[:, 2]

    return X_train, y_train

X_train,y_train=transformation(train)

test['gold'] = 0
X_test,y_test = transformation(test)

imp = Imputer(strategy="mean").fit(X_train)
X_train_imp = imp.transform(X_train)
X_test_imp = imp.transform(X_test)

rf = RandomForestClassifier( n_estimators=15, random_state=0)
rf.fit(X_train_imp[:,2:5],y_train)
rf.score(X_train_imp[:,2:5], y_train)

rf.predict(X_test_imp[:,2:5])

from sklearn.metrics import f1_score,precision_score,recall_score
train_pred=rf.predict(X_train_imp[:,2:5])
print(rf.score(X_train_imp[:,2:5], y_train))
print(f1_score(y_train,train_pred,average="macro"))
print(precision_score(y_train,train_pred,average="macro"))
print(recall_score(y_train,train_pred,average="macro"))
