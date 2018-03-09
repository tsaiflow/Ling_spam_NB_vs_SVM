import numpy as np
import sklearn
import sklearn.datasets as skd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.externals import joblib 

### evaluation dataset

ls_train = skd.load_files(container_path='./lingspam_public/lemm_stop/train')
ls_eval  =sklearn.datasets.load_files('./eval', shuffle=False)

count_vect = CountVectorizer()

x_train = count_vect.fit_transform(ls_train.data)
x_eval = count_vect.transform(ls_eval.data)

# np.shape(x_eval)
import pickle
with open("top_1000.txt", "rb") as fp:   # Unpickling
    top_train_1000 = pickle.load(fp)

   
clf3 = joblib.load('multinomial_NB_1000.pkl') 



new_x_eval_1000 = x_eval[:,top_train_1000[0]]
for i in range(1,1000):
    new_x_eval_1000 = hstack((new_x_eval_1000, x_eval[:,top_train_1000[i]]))

# print(np.shape(new_x_eval_1000))

# Use the trained model of multinomial binary 1000
clf3_predict = clf3.predict(new_x_eval_1000)
# print(clf3_predict)
np.savetxt('./eval/results.txt',clf3_predict,fmt='%s')