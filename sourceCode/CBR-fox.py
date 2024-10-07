import pandas as pd
import numpy
import numpy as np
import math
from scipy import signal
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D
from numpy import hstack
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import os
# def loadImports ():
    

def loadData(url="/weatherdata.csv"):
    # import and return the dataset you want to analyze
    data = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+url, parse_dates=True, index_col=1)
    data.head()
    return data

def loadForecastingModel(data, step_days = 14):

    dataset = data.filter(['HUM_MIN', 'HUM_AVG', 'HUM_MAX', 'PRES_MIN', 'PRES_AVG', 'PRES_MAX', 'TEMP_MIN',
                              'TEMP_AVG','TEMP_MAX']).values
    dataset = np.array(dataset)
    global inputnn, target, input_train, input_test, target_test, target_train
    inputnn, target = split_sequences(dataset, step_days)
    input_train, input_test, target_train, target_test = train_test_split(inputnn, target, test_size=0.30,
                                                                          random_state=4, shuffle=True)
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(step_days, input_train.shape[2])))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(input_train, target_train, validation_data = (input_test, target_test),batch_size = 16, epochs = 100)
    return model

def configExplanationParameters(nonOutputColumns_ = [0, 2, 3, 5, 6, 8],
                                titleColumns_ = ["Humidity", "Vapor Pressure", "Temperature"],
                                finalWindowNumber_ = 30,
                                # 0 number of results, 1 average, 2 Max values, 3 min values, 4 median
                                explicationMethodResult_ = 1
                                ):
   global nonOutputColumns, windowsLen, componentsLen, windowLen, titleColumns, smoothnessFactor, punishedSumFactor, finalWindowNumber, explicationMethodResult,windows, targetWindow,titleIndexes
   nonOutputColumns =nonOutputColumns_
   titleColumns = titleColumns_
   finalWindowNumber = finalWindowNumber_
   explicationMethodResult = explicationMethodResult_
   windows = np.delete(inputnn, nonOutputColumns, 2)
   targetWindow, windows = windows[-1], windows[:-1]
   windowsLen = len(windows)
   componentsLen = windows.shape[2]
   windowLen = windows.shape[1]
   titleIndexes = ["Window Index {0}".format(index) for index in range(windowsLen)
   ]
   smoothnessFactor = .03
   punishedSumFactor = .5



def split_sequences(sequences, n_steps, outputColumns):
    # outputColumns contains the desired columns to be outputs with format of tuple like: (1,2,3,4)

    inputnn, target = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix + 1 > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix], sequences[end_ix, outputColumns]
        inputnn.append(seq_x)
        target.append(seq_y)
    return array(inputnn), array(target)

def explain(data, model):
    prediction_train = model.predict(input_train)
    RMSE = math.sqrt(np.square(np.subtract(prediction_train, target_train)).mean())
    print("Root Mean Square Error Train:\n", RMSE)
    prediction = model.predict(input_test)
    global actualPrediction
    actualPrediction = prediction[-1]
    RMSE = math.sqrt(np.square(np.subtract(prediction, target_test)).mean())
    print("Root Mean Square Error Train:\n", RMSE)

    pearsonCorrelation = np.array(
        ([np.corrcoef(windows[currentWindow, :, currentComponent], targetWindow[:, currentComponent])[0][1]
          for currentWindow in range(len(windows)) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                        componentsLen)
    euclideanDistance = np.array(
        ([np.linalg.norm(targetWindow[:, currentComponent] - windows[currentWindow, :, currentComponent])
          for currentWindow in range(windowsLen) for currentComponent in range(componentsLen)])).reshape(-1,
                                                                                                         componentsLen)
    normalizedEuclideanDistance = ((euclideanDistance-np.amin(euclideanDistance, axis=0)) /
                                   (np.amax(euclideanDistance, axis=0)-np.amin(euclideanDistance, axis=0)))
    normalizedCorrelation = (.5 + (pearsonCorrelation - 2 * normalizedEuclideanDistance + 1) / 4)
    correlationPerWindow = np.sum(((normalizedCorrelation + punishedSumFactor) ** 2), axis=1)
    correlationPerWindow /= max(correlationPerWindow)

    smoothedCorrelation = lowess(correlationPerWindow, np.arange(len(correlationPerWindow)), smoothnessFactor)[:, 1]
    df_loess_3 = pd.DataFrame(smoothedCorrelation)
    valleyIndex, peakIndex = signal.argrelextrema(smoothedCorrelation, np.less)[0], \
                             signal.argrelextrema(smoothedCorrelation, np.greater)[0]


    concaveSegments = np.split(np.transpose(np.array((np.arange(windowsLen), correlationPerWindow))), valleyIndex)
    convexSegments = np.split(np.transpose(np.array((np.arange(windowsLen), correlationPerWindow))), peakIndex)

    bestWindowsIndex, worstWindowsIndex = list(), list()

    for split in concaveSegments:
        bestWindowsIndex.append(int(split[np.where(split == max(split[:, 1]))[0][0], 0]))
    for split in convexSegments:
        worstWindowsIndex.append(int(split[np.where(split == min(split[:, 1]))[0][0], 0]))

    bestDic = {index: correlationPerWindow[index] for index in bestWindowsIndex}
    worstDic = {index: correlationPerWindow[index] for index in worstWindowsIndex}
    global bestSorted
    bestSorted = sorted(bestDic.items(), reverse=True, key=lambda x: x[1])
    worstSorted = sorted(worstDic.items(), key=lambda x: x[1])

    global maxComp, minComp, lims 
    maxComp, minComp, lims = [], [], []
    for i in range(componentsLen):
        maxComp.append(int(max(max(a) for a in windows[:, :, i])))
        minComp.append(int(min(min(a) for a in windows[:, :, i])))
        lims.append(range(minComp[i], maxComp[i], int((maxComp[i] - minComp[i]) / 8)))

    bestMAE, worstMAE = [], []
    for i in range(len(bestSorted)):
        rawBestMAE = rawWorstMAE = 0
        for f in range(componentsLen):
            rawBestMAE += (windows[bestSorted[i][0]][windowLen - 1][f] - minComp[f]) / maxComp[f]
            rawWorstMAE += (windows[worstSorted[i][0]][windowLen - 1][f] - minComp[f]) / maxComp[f]
        bestMAE.append(rawBestMAE / componentsLen)
        worstMAE.append(rawWorstMAE / componentsLen)

    d = {'index': dict(bestSorted).keys(), 'CCI': dict(bestSorted).values(), "MAE": bestMAE,
         'index.1': dict(worstSorted).keys(), 'CCI.1': dict(worstSorted).values(), "MAE.1": worstMAE}
    df = pd.DataFrame(data=d)
    print(df)
    global cont
    cont = np.arange(1, windowLen + 1)
    if explicationMethodResult == 0:
        notCombinedOption()
    else:
        combinedOption()
    pass

def notCombinedOption():
    for i in range(len(bestSorted)):
        plt.figure(figsize=(12,8))
        for f in range(componentsLen):
            plt.subplot(componentsLen,1,f+1)
            plt.title(titleColumns[f])
            plt.plot(cont,targetWindow[:,f], '.-k', label = "Target")
            plt.plot(cont,windows[bestSorted[i][0],:,f], '.-g' ,label= "Data")
            plt.plot(windowLen+1,actualPrediction[f], 'dk', label = "Prediction")
            plt.plot(windowLen+1,windows[bestSorted[i][0]][windowLen-1][f], 'dg', label = "Next day")
            plt.grid()
            plt.xticks(range(1,windowLen+2,1))
            plt.yticks(lims[f])
        plt.tight_layout()
        plt.show()

def subOptions(op):
    if op==1:
        newCase = np.sum(windows[list(dict(bestSorted).keys())], axis=0)/len(bestSorted)
    elif op==2:
        newCase = np.max(windows[list(dict(bestSorted).keys())], axis=0)
    elif op==3:
        newCase = np.min(windows[list(dict(bestSorted).keys())], axis=0)
    elif op==4:
        newCase = np.median(windows[list(dict(bestSorted).keys())], axis=0)
    return newCase

def combinedOption():
    plt.figure(figsize=(12,8))
    newCase = np.zeros((windowLen,componentsLen))
    try:
        newCase = subOptions(explicationMethodResult)
    except:
        print("Unavailable option")
    for f in range(componentsLen):
        plt.subplot(componentsLen,1,f+1)
        plt.title(titleColumns[f])
        plt.plot(cont,targetWindow[:,f], '.-k', label = "Target")
        plt.plot(cont,newCase[:,f], '.-g' ,label= "Data")
        plt.plot(windowLen+1,actualPrediction[f], 'dk', label = "Prediction")
        plt.plot(windowLen+1,newCase[windowLen-1][f], 'dg', label = "Next day")
        plt.grid()
        plt.xticks(range(1,windowLen+2,1))
        plt.yticks(lims[f])
    plt.tight_layout()
    plt.show()

# loadImports()
data = loadData()
model = loadForecastingModel(data)
configExplanationParameters()
explain(data,model)
