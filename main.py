#!/usr/bin/python

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Função para montar a matriz de dados que será trabalhada posteriormente
def detectVectos(base):
    matrix = []
    count  = 0

    rows, colums  = base.shape

    for i in range(0, colums):
        matrix.append([])

    for i in base:
        for j in base[i]:
            #print(j)
            matrix[count].append((j)*1.0)
        count += 1

    #print("matrix:", matrix)

    return matrix


def meanVectors(matrix):
    rowConter = 0
    columCounter = 0
    itemCounter = 0

    # print(len(matrix))
    # print(len(matrix[1]))

    s = (r, c) = len(matrix), len(matrix[1])

    meanMatrix = np.zeros(s)

    for i in matrix:
        mean = np.mean(i)
        #print()
        print("mean: ", mean)
        print("row: ", i)
        #print()

        for j in i:
            meanMatrix[columCounter][rowConter] = (float((j*1.000) - mean))

            rowConter += 1

        rowConter = 0
        columCounter += 1

    print(meanMatrix)
    return meanMatrix


def covarianceMatix(meanMatrix):
    #meanMatrix = np.transpose(meanMatrix)
    #print(meanMatrix)

    print(np.cov(meanMatrix))
    return np.cov(meanMatrix)


def eigen(covMatrix):
    return np.linalg.eig(covMatrix)


def PCA(matrix):
    print("Esta a função para executar o PCA")
    originalMatrix = matrix
    print("Original Matrix: ", originalMatrix)
    print(len(matrix))

    meanMatrix = meanVectors(matrix)
    print()

    covMatrix = covarianceMatix(meanMatrix)
    print()

    eingen = eigen(covMatrix)
    eigenVect = np.transpose(eingen[1])

    print("Eigenvectors: ", eigenVect)
    print()

    featureVector = []

    if len(matrix) == 2:
        featureVector = eigenVect[0]
    elif len(matrix) == 3:
        featureVector.append(eigenVect[1])
        featureVector.append(eigenVect[2])
    elif len(matrix) > 3:
        featureVector.append(eigenVect[3])
        featureVector.append(eigenVect[0])
        featureVector.append(eigenVect[1])



    print("Feature Vector: ", featureVector)
    print()

    #matrixT = np.transpose(originalMatrix)
    #print("Matrix Tranpose: \n", matrix)
    #rint()

    fData = np.matmul(featureVector, matrix)

    # print(featureVector)

    # for i in range(0, len(featureVector)):
    #     featureVector[i] = 1/featureVector[i]

    #featureVector2 = np.invert(eingen)


    #fData = fData.view(np.complex)

    print(fData)
    print(featureVector)

    return (fData, eingen)

# ================================= Metodos para o Menu ===================================
def basesMenu():
    print("==================================")
    print("1 - Water Alps Base")
    print("2 - Books, Attend and Grades Base")
    print("3 - US Census")
    print("4 - Height and Shoes")
    print("5 - Hald Database")
    print("6 - Exemple Data")
    print("==================================")


def waterOption():
    water = r'./bases/alpswater.xlsx'

    aw = pd.read_excel(water)

    x = aw["BPt"]
    y = aw["Pressure"]
    w = []

    matrix = detectVectos(aw)

    pca = PCA(matrix)

    mc1 = []
    mc2 = []
    print(pca)

    print(pca[0])

    print('Vector 1', pca[1][1][0])
    print('Vector 2', pca[1][1][1])

    fig = plt.figure(figsize=(10, 5), dpi=70, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 2, 1)

    ax.scatter(matrix[0], matrix[1])

    ax = fig.add_subplot(1, 2, 2)

    ax.scatter(pca[0], pca[0])

    plt.plot()
    plt.show()


def booksOption():
    books = r'./bases/Books_attend_grade.xls'
    bag = pd.read_excel(books)

    x1 = bag["BOOKS"]
    x2 = bag["ATTEND"]
    y = bag["GRADE"]
    w = []

    matrix = detectVectos(bag)

    print(matrix)

    pca = PCA(matrix)


    fig = plt.figure(figsize=(20, 10), dpi= 90, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # print('x1: ', x1)
    # print('x2: ', x2)
    # print('y: ', y)

    mx = []

    mx.append(x1)
    mx.append(x2)
    mx.append(y)

    np.transpose(mx)



    ax.scatter(x1, x2, y)

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(pca[0][0], pca[0][1])

    plt.plot()
    plt.show()


def censusOption():
    census = r'./bases/us_census.xlsx'
    uc = pd.read_excel(census)

    x = uc["year"]
    y = uc["number"]
    w = []

    matrix = detectVectos(uc)

    pca = PCA(matrix)

    mc1 = []
    mc2 = []
    print(pca)

    print(pca[0])

    print('Vector 1', pca[1][1][0])
    print('Vector 2', pca[1][1][1])

    fig = plt.figure(figsize=(10, 5), dpi=70, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 2, 1)

    ax.scatter(matrix[0], matrix[1])

    ax = fig.add_subplot(1, 2, 2)

    ax.scatter(pca[0], pca[0])

    plt.plot()
    plt.show()


def shoesOption():
    height = r'./bases/height-shoes.xlsx'
    hs = pd.read_excel(height)

    print(hs)
    matrix = detectVectos(hs)

    print( matrix)

    pca = PCA(matrix)

    mc1 = []
    mc2 = []
    print(pca)

    print(pca[0])

    print('Vector 1', pca[1][1][0])
    print('Vector 2', pca[1][1][1])

    fig = plt.figure(figsize=(10, 5), dpi=70, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 2, 1)

    ax.scatter(matrix[0], matrix[1])

    ax = fig.add_subplot(1, 2, 2)

    ax.scatter(pca[0], pca[0])

    plt.plot()
    plt.show()


def haldOption():
    hald = r'./bases/hald.xlsx'
    h = pd.read_excel(hald)

    #print(h)

    matrix = detectVectos(h)

    pca = PCA(matrix)

    fig = plt.figure(figsize=(10, 10), dpi=90, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # print('x1: ', x1)
    # print('x2: ', x2)
    # print('y: ', y)


    ax.scatter(pca[0][0], pca[0][1], pca[0][2])


    plt.show()


def exempleDataOption():
    exemple = r'./bases/exemple_data.xlsx'
    ex = pd.read_excel(exemple)

    matrix = detectVectos(ex)

    pca = PCA(matrix)

    mc1 = []
    mc2 = []
    print(pca)

    print(pca[0])

    print('Vector 1', pca[1][1][0])
    print('Vector 2', pca[1][1][1])


    # for i in range(0, len(matrix[0])):
    #     mc1.append(matrix[0][i] * pca[1][1][0][1]/pca[1][1][1][1])
    # #
    # for i in  range(0, len(matrix[0])):
    #     mc2.append(matrix[0][i] * pca[1][1][0][0]/pca[1][1][1][0])
    #
    # print(mc1)
    # print(mc2)
    #
    # print()

    fig = plt.figure(figsize=(10, 5), dpi=70, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 2, 1)

    ax.scatter(matrix[0], matrix[1])

    ax = fig.add_subplot(1, 2, 2)

    ax.scatter(pca[0], pca[0])

    plt.plot()
    plt.show()




def dataBaseChoice(inp):
    switcher = {
        "1": waterOption,
        "2": booksOption,
        "3": censusOption,
        "4": shoesOption,
        "5": haldOption,
        "6": exempleDataOption
    }

    func = switcher.get(inp, lambda: "This option do not exist !")

    return func()
# ==============================================================================================

if __name__ == '__main__':
    basesMenu()
    inp = input("Type your Option: ")
    dataBaseChoice(inp)

