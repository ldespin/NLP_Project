'''Functions file for task0 project. '''
import numpy as np
import pandas as pd
import os
import yaml
import codecs
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from evaluate import distance



########## Access data ##########

tags_file = yaml.load(open("task0-data/tags.yaml"), Loader=yaml.FullLoader)
categories = tags_file['ordering']
categories_df = pd.DataFrame(categories, columns=["categories"])


def charac_dict(lang, path="task0-data/alpha.all"):
    '''Takes task0's character file, return a character dict for the specified language whose keys are the language alphabet and values are integers.'''
    characters_file = open(path,'r')
    lines = list(characters_file)
    for i, line in enumerate(lines):
        if lang in line and f"'{lang}'" not in line:
            charac = sorted(lines[i+1].split())
            charac_index = dict([(charac, i) for i, charac in enumerate(charac)])
            charac_index[' ']=len(charac_index)
            characters_file.close()
            return charac_index


def read(fname):
    '''Read a train set from a task0 name. Returns a dictionnary whose keys are lemmas.'''
    D = {}
    with codecs.open(fname, 'rb', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lemma, word, tag = line.split("\t")
            if lemma not in D:
                D[lemma] = {}
            D[lemma][tag] = word
    return D


def dict_to_frame(Dict: dict, Type: str):
    '''Turning a train dict or a test dict (Type='train' or 'test'), whose keys are lemmas, to a dataframe.
    Lemmas having several bundles are duplicated in the dataframe.
    '''
    lemmas = list(Dict.keys())
    duplicated_lemmas = []
    for lemma in lemmas:
        duplicated_lemmas += [lemma for i in range(len(Dict[lemma]))]

    if Type == 'train':
        columns = ['bundle', 'form']
    elif Type == 'test':
        columns = ['bundle']

    df = pd.concat(
        [pd.DataFrame(Dict[lemma].items(), columns = columns) for lemma in lemmas],
        axis=0, 
        ignore_index=True
    )
    df['lemma'] = duplicated_lemmas
    
    return df

def extracting_train_input_models(df: pd.DataFrame):
    '''
    Extract from non encoded dataframe Values and Non-encoded values of bundles, 
    Indexes of those values and sets corresponding subsets.
    Return dictionnaries of the corresponding entities.
    '''
    # Encoding
    encoded_df = encode_tags_dataset(df)
    
    # Extracting    
    Values = {}
    Non_encoded_Values = {}
    Indexes = {}
    sets_per_classes = {}

    Y = encoded_df[categories].copy()
    i = 0
    while len(Y) != 0:
        value = Y.values[0]
        Values[i] = value

        P = pd.DataFrame(Y == value, columns=categories)
        Q = P.all(axis=1)
        index = Q.where(Q==True).dropna().index
        Indexes[i] = index

        sets_per_classes[i] = df.loc[index].copy()
        Non_encoded_Values[i] = sets_per_classes[i]['bundle'].unique()[0]

        Y.drop(index, axis=0, inplace=True)
        i += 1
    
    return Values, Non_encoded_Values, Indexes, sets_per_classes

def extracting_test_sets_per_classes(df, Values):
    # Encoding
    encoded_df = encode_tags_dataset(df)
    encoded_df['model_class'] = encoded_df.apply(
        lambda x: assign_class_test(x[categories].values, Values, []), axis=1
    )

    test_sets_per_classes = {}
    for i in range(len(Values)):
            test_set = encoded_df[
                encoded_df["model_class"] == i
            ][["bundle", "form", "lemma"]].reset_index(drop=True)
            
            if not test_set.empty:
                test_sets_per_classes[i] = test_set
    
    return test_sets_per_classes



########## Encoding forms and lemmas ##########


def get_longest(df):
    '''
    Return the length of the longest word observed in the dataset
    '''
    max = 0
    for word in df["form"]:
        if len(word)>max:
            max = len(word)
    for word in df['lemma']:
        if len(word)>max:
            max = len(word)
    return max


def encode_data(df, charac_dict, M):
    '''
    Encode data with one hot vectors, based on the dictionary of characters previously defined
    '''
    N = len(df["lemma"].values)
    T = len(charac_dict)
    
    X = np.zeros((N,M,T))
    Y = np.zeros((N,M,T))
    
    for i, row in df.iterrows():
        x = row["lemma"]
        y = row["form"]
        if len(x)<M and len(y)<M:
            for j,char in enumerate(x):
                k = charac_dict[char]
                X[i,j,k]=1.0
            for j,char in enumerate(y):
                k = charac_dict[char]
                Y[i,j,k]=1.0
    return X,Y



########## Encoding tags ##########


def encode_tag(bundle):
    '''Return the encoded form of bundle as a dictionnary, whose keys are the categories, values are integer '''
    encoding = {}
    
    #Isolating categories information
    bundle = bundle.split(';')
    nbr_cat = len(bundle)
    
    #Encoding dict
    starting_research = 0
    for i in range(nbr_cat):
        value = bundle[i]
        # Finding the corresponding categories:
        for cat in categories[starting_research:]:
            cat_values = tags_file['categories'][cat]
            if value in cat_values:
                starting_research = categories.index(cat) + 1
                # Mapping of categories value with N* ; 0 meaning no component in the categories
                encoding[cat] = cat_values.index(bundle[i]) + 1
            else:
                encoding[cat] = 0
    
    return encoding

def encode_tags_dataset(dataset: pd.DataFrame):
    '''Encode all the tags from a given task0 dataframe.'''
    encoded_df = pd.concat(
        [
            pd.DataFrame(
                encode_tag(dataset['bundle'][i]), 
                columns = categories, 
                index =[0]
            ) 
            for i in range(len(dataset))
        ],
        axis=0, 
        ignore_index=True
    )
    encoded_df = pd.concat([dataset, encoded_df], axis=1)
    
    return encoded_df



########## Test bundle classification ##########


def indicator_non_zero(A: list):
    '''Return A where non zero values are replaced by 1. A is a one dimensional array.'''
    B = A.copy()
    B[pd.Series(A).where(B != 0).dropna().index] = 1
    return B


def assign_class_test(test_bundle: np.array, Values: dict, categories_to_drop: list):
    ''' Take a bundle and the training dict classes values 'Values', 
    projects this bundle in the classification categories for
    affecting a class affected to the bundle (integer). Returns the affected class (int).
    '''
    #Projection
    test_bundle_projected = test_bundle.copy()
    test_bundle_projected[
        categories_df.where(categories_df["categories"].apply(lambda x: x in categories_to_drop)).dropna().index
    ] = 0
    
    #Affectation
    #if the projected test bundle is the Values
    in_Values = np.array(
        [(test_bundle_projected == list(Values.values())[i]).all() for i in range(len(Values))]
    )
    
    if in_Values.any():
        bundle_class = np.where(in_Values == True)[0][0]
    else:
        #If test_bundle not in the Values, 
        #find the closest bundle in terms of structure (what categories are present (non zero value))
        indicator_Values = {}
        for i in range(len(Values)):
                indicator_Values[i] = indicator_non_zero(Values[i])

        indicator_test_bundle_projected =  indicator_non_zero(test_bundle_projected)
        
        distances_indicator = [
            np.sum(np.abs(indicator_test_bundle_projected - list(indicator_Values.values())[i])) 
            for i in indicator_Values.keys()
        ]
        
        bundle_class_candidates = np.where(distances_indicator == min(distances_indicator))[0]
        
        if len(bundle_class_candidates) == 1:
            bundle_class = bundle_class_candidates[0]
        else:
            distances = [
                np.sum(np.abs(test_bundle_projected - list(Values.values())[i])) 
                for i in bundle_class_candidates
            ]
            refined_bundle_class_candidates = np.where(distances == min(distances))[0]
            if len(refined_bundle_class_candidates) == 1:
                bundle_class = refined_bundle_class_candidates[0]
            else:
                bundle_class = np.random.choice(bundle_class_candidates)

    return bundle_class


########## Training ##########


def train_linear_reg(
    training_sets_per_classes: dict, 
    char_dict, 
    M_max
):
    '''
    Trains one linear regression model per class of bundles, to predict an encoded form, from an encoded lemma
    returns a dictionary containing the trained models for each class,
    and a dictionary containing the encoded lemmas and forms, for each class
    '''
    
    X_train = {}
    Y_train = {}
    models = {}
    
    for i in training_sets_per_classes:
        model = LinearRegression()
        subset = training_sets_per_classes[i]
        new_index = np.arange(len(subset))
        subset = subset.set_index(new_index)
        
        X_train_temp, Y_train_temp = encode_data(subset, char_dict, M_max)
        (N,M,T) = X_train_temp.shape
        X_train_temp = X_train_temp.reshape((N,M*T))
        Y_train_temp = Y_train_temp.reshape((N,M*T))
        
        X_train[i] = X_train_temp
        Y_train[i] = Y_train_temp
                
        models[i] = model.fit(X_train[i], Y_train[i])
    
    return models, X_train, Y_train

def train_len_words(
    training_sets_per_classes: dict, 
):
    '''
    Trains a linear regression per class of bundle, to predict the length of a form from the length of a lemma
    returns a dictionary containing the trained models for each class,
    and a dictionary containing the lengths of lemmas and forms, for each class
    '''
    
    X_train = {}
    Y_train = {}
    models = {}
    
    for i in training_sets_per_classes:
        model = LinearRegression()
        subset = training_sets_per_classes[i]
        new_index = np.arange(len(subset))
        subset = subset.set_index(new_index)
        
        X_train[i] = np.array([len(lemma) for lemma in subset["lemma"].values])
        Y_train[i] = np.array([len(form) for form in subset["form"].values])
                
        models[i] = model.fit(X_train[i].reshape(-1, 1), Y_train[i])
    
    return models, X_train, Y_train
    
def encode_sets(sets_per_classes, char_dict, M):
    
    X = {}
    Y = {} 

    for i in sets_per_classes:
        subset = sets_per_classes[i]
        new_index = np.arange(len(subset))
        subset = subset.set_index(new_index)
        X_temp, Y_temp = encode_data(subset, char_dict, M)
        (N,_,T) = X_temp.shape
        
        X_temp = X_temp.reshape((N,M*T))
        Y_temp = Y_temp.reshape((N,M*T))

        X[i] = X_temp
        Y[i] = Y_temp
        
    return X, Y

def encode_len(sets_per_classes):
    
    X_len = {}
    Y_len = {}
    
    for i in sets_per_classes:
        subset = sets_per_classes[i]
        new_index = np.arange(len(subset))
        subset = subset.set_index(new_index)
        
        X_len[i] = np.array([len(lemma) for lemma in subset["lemma"].values])
        Y_len[i] = np.array([len(form) for form in subset["form"].values])
    
    return X_len, Y_len

def predict_len(X_len, sets_per_classes, models_len):
    Y_len_hat = {}
    for i in sets_per_classes:
        Y_len_hat[i] = models_len[i].predict(X_len[i].reshape(-1, 1)).round()
    return Y_len_hat

def predict_and_save(X, Y, sets_per_classes, models, models_len, M, T, char_dict, Non_encoded_train_Values, saving_file="results/results.txt"):
    
    (X_len, _) = encode_len(sets_per_classes)
    Y_len_hat = predict_len(X_len, sets_per_classes, models_len)
    
    Y_hat = {}
    
    #We predict the form, for each the different datasets associated with different classes
    for i in sets_per_classes:
        Y_hat[i] = models[i].predict(X[i])
        
    #We reshape our encoded vectors so they are of form (N,M,T) 
    for i in sets_per_classes:
        (N,_) = X[i].shape
        X[i] = X[i].reshape((N,M,T))
        Y[i] = Y[i].reshape((N,M,T))
        Y_hat[i] = Y_hat[i].reshape((N,M,T))
        
    f = open(saving_file,"w")
    avg_distance = []
    
    for i in sets_per_classes:
        M_dict = Y_len_hat[i]
        lemma = decode_vector(X[i], char_dict)
        form = decode_vector(Y[i], char_dict)
        Y_hat[i] = convert_pred_limit(Y_hat[i], M_dict)
        form_hat = decode_vector(Y_hat[i], char_dict)
        bundle = Non_encoded_train_Values[i]

        for j in range(len(lemma)):
            avg_distance.append(distance(form[j],form_hat[j]))
            f.write(f"{lemma[j]}\t{bundle}\t{form[j]}\t{form_hat[j]}\n")
    f.close()
    return np.mean(avg_distance)

def decode_vector(X, char_dict):
    '''
    Translate and encoded lemma or form into a string
    '''
    idx_dict = {char_dict[k]:k for k in char_dict}
    words = []
    (N,M,T) = X.shape
    for n in range(N):
        word = ""
        for m in range(M):
            for t in range(T):
                if X[n,m,t]==1.0:
                    word+=idx_dict[t]
        words.append(word)
    return words


def convert_pred(Y_pred):
    '''
    Convert the predicted encoded form into a proper encoded version composed of one-hot vectors
    '''
    Y_pred_converted = np.zeros(Y_pred.shape)
    (Nc,Mc,_) = Y_pred.shape
    for n in range(Nc):
        for m in range(Mc):
            Y_pred_converted[n,m,np.argmax(Y_pred[n,m])]=1.0
    return Y_pred_converted

def convert_pred_limit(Y_pred, M):
    '''
    Convert the predicted encoded form into a proper encoded version composed of one-hot vectors,
    limiting to the predicted length of the form
    '''
    Y_pred_converted = np.zeros(Y_pred.shape)
    (Nc,Mc,_) = Y_pred.shape
    for n in range(Nc):
        for m in range(Mc):
            if m<M[n]:
                Y_pred_converted[n,m,np.argmax(Y_pred[n,m])]=1.0
    return Y_pred_converted    