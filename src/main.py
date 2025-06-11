from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
import random

def Z (antigen):
    Z_descriptions = {"A": (0.07, -1.73, 0.09),  "V": (-2.69, -2.53, -1.29), "L": (-4.19, -1.03, -0.98), 
        "I": (-4.44, -1.68, -1.03), "P": (-1.22, 0.88, 2.23), "F": (-4.92, 1.30, 0.45),  
        "W": (-4.75, 3.65, 0.85),  "M": (-2.49, -0.27, -0.41), "K": (2.84, 1.41, -3.14),
        "R": (2.88, 2.52, -3.44),  "H": (2.41, 1.74, 1.11),  "G": (2.23, -5.36, 0.30),
        "S": (1.96, -1.63, 0.57),  "T": (0.92, -2.09, -1.40), "C": (0.71, -0.97, 4.13),
        "Y": (-1.39, 2.32, 0.01),  "N": (3.22, 1.45, 0.84), "Q": (2.18, 0.53, -1.14),
        "D": (3.64, 1.13, 2.36),   "E": (3.08, 0.39, -0.07)}
    if antigen not in Z_descriptions:
        return (0, 0, 0)  
    else:
        return Z_descriptions[antigen]

def sequence_encoder (sequences): 
    encoded_sequences = []
    max_aa = 10
    for s in sequences:
        s = s[:max_aa]
        z_desc = [Z(amino_acid) for amino_acid in s]
        while (len(z_desc)<max_aa):
            z_desc.append((0,0,0))
        encoded_sequences.append(np.ravel(z_desc)) #flatten to 1D
    encoded_sequences = np.array(encoded_sequences)  
    return encoded_sequences

def ACC(dataset):
    ACCN = np.zeros((len(dataset),45))
    for i, d in enumerate(dataset):
        desc = [Z(amino_acid) for amino_acid in d]
        n = len (desc)
        l_list = [1,2,3,4,5]
        id = 0
        for l in l_list:
            if n<=l:
                continue
            #AJJ
            for j in range (3):
                cal = sum(desc[k][j] * desc[k + l][j] for k in range(n - l)) / (n - l)
                ACCN[i, id] = cal
                id += 1
            #CJK
            for aa1 in range(3):
                for aa2 in range (3):
                    if aa1 == aa2:
                        continue
                    cal = sum(desc[k][aa1] * desc[k + l][aa2] for k in range(n - l)) / (n - l)
                    ACCN[i, id] = cal
                    id += 1
    return ACCN

def combineZ_ACC(dataset):
    Z_features = sequence_encoder (dataset)
    ACC_features = ACC(dataset)
    combined = np.hstack((Z_features,ACC_features))
    return combined

def import_Bcells(p_file, n_file):
    random.seed(42)
    num_to_choose = 180
    p_sequences = []
    n_sequences = []
    with open(p_file, 'r') as file:
        for line in file:
            p_sequences.append(line)
    with open(n_file, 'r') as file1:
        for line in file1:
            n_sequences.append(line)
    choose_p = random.sample(p_sequences, num_to_choose)
    choose_n = random.sample (n_sequences, num_to_choose)
    dataset = choose_p+choose_n
    labels = [1]*num_to_choose+[0]*num_to_choose
    return p_sequences, dataset, labels

def import_Tcells (p_file, n_file):
    random.seed(42)
    num_to_choose = 150
    P_sequence_CTL = []
    N_sequence_CTL = []
    P_sequence_HTL = []
    N_sequence_HTL = []
    with open (p_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith(">"):
                if len(line)<=11: #assuming CTL is for length of 11 and less, and HTL will be longer
                    P_sequence_CTL.append(line)
                else:
                    P_sequence_HTL.append(line)
    with open (n_file, 'r') as file1:
        for line in file1:
            if not line.startswith(">"):
                if len(line)<=11:
                    N_sequence_CTL.append(line)
                else:
                    N_sequence_HTL.append(line)
    choose_P_CTL = random.sample (P_sequence_CTL, num_to_choose)
    choose_N_CTL = random.sample (N_sequence_CTL, num_to_choose)
    choose_P_HTL = random.sample (P_sequence_HTL, num_to_choose)
    choose_N_HTL = random.sample (N_sequence_HTL, num_to_choose)
    datasetCTL = choose_P_CTL+choose_N_CTL
    datasetHTL = choose_P_HTL+choose_N_HTL
    labels_CTL = [1]*num_to_choose+[0]*num_to_choose
    labels_HTL = [1]*num_to_choose+[0]*num_to_choose
    return datasetCTL, datasetHTL, labels_CTL, labels_HTL, P_sequence_CTL, P_sequence_HTL

def SVM_and_RFE (dataset, labels):

    #rfe portion
    x = combineZ_ACC(dataset)
    y = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)  

    svm_for_rfe = svm.SVC(kernel="linear")
    rfe = RFECV(estimator = svm_for_rfe, step =1, cv=5) 
    rfe.fit(x_train,y_train)
    select = rfe.support_
    rank = rfe.ranking_
    x_rfe_train = x_train[:,select]
    x_rfe_test = x_test[:,select]
    print ("RFE feature selection:") 
    for i in range (len(select)):
        print(f'Feature number: {i+1}; Selected {select[i]}; Rank: {rank[i]}')
    print ("\n")

    #SVM portion
    params_grid = [
    {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']}, 
    {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 'scale'], 'kernel':['rbf', 'poly', 'sigmoid']}
    ]
    gs = GridSearchCV(svm.SVC(), params_grid, cv = 5)
    gs.fit(x_rfe_train,y_train)
    print ("Results: ")
    print (gs.best_params_)
    SVM_selected = gs.best_estimator_
    SVM_selected.fit(x_rfe_train, y_train)
    y_predicted = SVM_selected.predict(x_rfe_test)
    a = accuracy_score(y_test, y_predicted)
    print(f"{gs.best_params_['kernel']} best performance with accuracy {a:.3f}")
    return SVM_selected, select
    
def select_top_epitopes (SVM_model, select, unused_data):
    positive_feats = combineZ_ACC(unused_data)[:, select]
    confidence = SVM_model.decision_function(positive_feats)
    index = np.argsort(confidence)
    sorted_index = index [::-1]
    selected_epitopes = []
    for s_i in sorted_index[:10]:
        selected_epitopes.append(unused_data[s_i].strip())
    return selected_epitopes

#import cells from txt files
positive_b, dataset, labels = import_Bcells("PositiveB.txt", "NegativeB.txt")
datasetCTL, datasetHTL, labelsCTL, labelsHTL, P_sequence_CTL, P_sequence_HTL = import_Tcells("PositiveT.txt", "NegativeT.txt")

#Training models for each type
print ("B cells: ")
SVMB_selected, selectB = SVM_and_RFE(dataset, labels)
print ("\nCTL cells: ")
CTL_SVM, selectCTL = SVM_and_RFE(datasetCTL, labelsCTL)
print ("\nHTL cells: ")
HTL_SVM, selectHTL = SVM_and_RFE(datasetHTL, labelsHTL)

#tests a given prediction
# new_sequence = "LCFLEDLERN"
# features_sequence = combineZ_ACC([new_sequence])[:,select]
# predicted_result = SVMB_selected.predict(features_sequence)[0]
# if predicted_result == 1:
#     print (f"\nPrediction for {new_sequence}: Effective")
# else:
#     print ((f"\nPrediction for {new_sequence}: Ineffective"))

#Predict new epitope vaccine sequence
unique_positiveB = list(set(positive_b)-set(dataset[:100]))
unique_postitiveCTL = list(set(P_sequence_CTL)-set(datasetCTL[:100]))
unique_postitiveHTL = list(set(P_sequence_HTL)-set(datasetHTL[:100]))

top_B = select_top_epitopes(SVMB_selected, selectB, unique_positiveB)
top_CTL = select_top_epitopes(CTL_SVM, selectCTL, unique_postitiveCTL)
top_HTL = select_top_epitopes(HTL_SVM, selectHTL, unique_postitiveHTL)

B_linker = "KK"
CTL_linker = "AAY"
HTL_linker = "GPGPGP"
vaccine = B_linker.join(top_B)+"\n"+CTL_linker.join(top_CTL)+"\n"+HTL_linker.join(top_HTL)
print (f"\nSequence constructed: {vaccine}")
with open("constructed_vaccine_sequence.txt", "w") as f:
    f.write("Sequence constructed: \n")
    f.write(vaccine)
