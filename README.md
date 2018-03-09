# Ling_spam_NB_vs_SVM

TO REPRODUCE,
a. To reproduce the top-N features selected for N = {10, 100, 1000}, run all of the cells in the ipynb. This will take 1-2 minutes due to calculating information gain for features. Then uncomment the lines in the last cell to print out N = 10, N = 100 or N = 1000.

b. to reproduce the precision and recall rate, just run all of the cells in .ipynb and outputs will be printed among the notebook. The order is in Multinomial NB with TF, Multinomial NB with BINARY, and Bernouli NB, with features selected from 10, 100 , 1000.

c. the svm classifier is trained from top 1000 binary features, its regularization parameter is set to default, which is C = 1.0.; the kernel used is rbf. Its results can be found in the 2nd to last cell in the ipynb

TO  EVALUATE,
copy the datasets to a folder named "eval" with subfolder named "all". Then just run the following command on terminal,
                                                python eval.py
Results will output to eval/results.txt
