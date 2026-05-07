import pandas as pd
from math import sqrt

df = pd.read_csv(r"Cancer_Data.csv")

diagnosis = df["diagnosis"].tolist()
X = df.drop(columns=["id", "diagnosis"])
x = X.values.tolist()

inputs = []

user_input = 'y'
while user_input != 'n':
    for i, feature in enumerate(X.columns):
        inputs.append(float(input(f'Enter {feature.replace("_"," ").title()}: ')))

    nearest_neighbors = []
    for i in range(len(x[0])):
        result = sqrt((x[0][i] - inputs[0])**2 +
                     (x[1][i] - inputs[1])**2 +
                     (x[2][i] - inputs[2])**2 +
                     (x[3][i] - inputs[3])**2 +
                     (x[4][i] - inputs[4])**2 +
                     (x[5][i] - inputs[5])**2 +
                     (x[6][i] - inputs[6])**2 +
                     (x[7][i] - inputs[7])**2 +
                     (x[8][i] - inputs[8])**2 +
                     (x[9][i] - inputs[9])**2 +
                     (x[10][i] - inputs[10])**2 +
                     (x[11][i] - inputs[11])**2 +
                     (x[12][i] - inputs[12])**2 +
                     (x[13][i] - inputs[13])**2 +
                     (x[14][i] - inputs[14])**2 +
                     (x[15][i] - inputs[15])**2 +
                     (x[16][i] - inputs[16])**2 +
                     (x[17][i] - inputs[17])**2 +
                     (x[18][i] - inputs[18])**2 +
                     (x[19][i] - inputs[19])**2 +
                     (x[20][i] - inputs[20])**2 +
                     (x[21][i] - inputs[21])**2 +
                     (x[22][i] - inputs[22])**2 +
                     (x[23][i] - inputs[23])**2 +
                     (x[24][i] - inputs[24])**2 +
                     (x[25][i] - inputs[25])**2 +
                     (x[26][i] - inputs[26])**2 +
                     (x[27][i] - inputs[27])**2 +
                     (x[28][i] - inputs[28])**2 +
                     (x[29][i] - inputs[29])**2)
        nearest_neighbors.append((result, diagnosis[i]))
 
    nearest_neighbors.sort()
    print("Nearest 5 neighbors:")
    cancer_type = 0
    for i in range(5):
        print(f"{i+1}: Distance = {nearest_neighbors[i][0]}, Diagnosis = {nearest_neighbors[i][1]}")
        cancer_type += nearest_neighbors[i][1]
    if cancer_type >= 3:
        print("Predicted Cancer Type: Malignant")
    else:
        print("Predicted Cancer Type: Benign")
    user_input = input("Do you want to make another prediction? Enter (y) for Yes OR (n) for No: ")
print("Model ran successfully!")