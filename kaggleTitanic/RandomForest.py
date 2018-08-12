# -*- coding: utf-8 -*-
import csv
import os
import random
import math
from sklearn.ensemble import RandomForestClassifier

def readData(fileName):
    result = {}
    with open(fileName,'r') as f:
        rows = csv.reader(f)
        for row in rows:
            if 'attr_list' in result:
                for i in range(len(result['attr_list'])):
                    key = result['attr_list'][i]
                    if key not in result:
                        result[key] = []
                    result[key].append(row[i])
            else:
                result['attr_list'] = row
    return result

def writeData(fileName, data):
    csvFile = open(fileName, 'w')
    writer = csv.writer(csvFile)
    n = len(data)
    for i in range(n):
        writer.writerow(data[i])
    csvFile.close()

def convertData(dataList):
    hashTable = {}
    count = 0
    for i in range(len(dataList)):
        if dataList[i] not in hashTable:
            hashTable[dataList[i]] = count
            count += 1
        dataList[i] = str(hashTable[dataList[i]])

def convertValueData(dataList):
    sumValue = 0.0
    count = 0
    for i in range(len(dataList)):
        if dataList[i] == "":
            continue
        sumValue += float(dataList[i])
        count += 1
        dataList[i] = float(dataList[i])
    avg = sumValue / count
    for i in range(len(dataList)):
        if dataList[i] == "":
            dataList[i] = avg

def dataPredeal(data):
    useDataList = ['Sex','Pclass', 'SibSp','Parch','Embarked']
    convertValueData(data["Age"])
    for i in range(len(useDataList)):
        attrName = useDataList[i]
        convertData(data[attrName])
    
def train(train_data):
    dataPredeal(train_data)
    useList = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
    x = []
    y = []
    for i in range(len(train_data['Survived'])):
        item = []
        for j in range(len(useList)):
            item.append(train_data[useList[j]][i])
        x.append(item)
        y.append(train_data['Survived'][i])
    clf = RandomForestClassifier().fit(x,y)
    return clf

def predict(clf, test_data, pos):
    x = [[]]
    useList = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
    for i in range(len(useList)):
        x[0].append(test_data[useList[i]][pos])
    result = clf.predict(x)
    return [test_data['PassengerId'][pos],int(result[0])]

def run():
    dataRoot = '/Users/aaron/Downloads/kaggle/titanic/'
    train_data = readData(dataRoot + 'train.csv')
    test_data = readData(dataRoot + 'test.csv')
    clf = train(train_data) 
    dataPredeal(test_data)
    result_list = []
    result_list.append(['PassengerId', 'Survived'])
    for i in range(len(test_data['PassengerId'])):
        result_list.append(predict(clf, test_data, i))
        print ('cal:' + str(i))
    writeData(dataRoot + 'randomforest_predictions.csv', result_list)

run()

