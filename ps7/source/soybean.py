"""
Author      : Arjun Natarajan
Class       : HMC CS 158
Date        : 2020 Apr 08
Description : Multiclass Classification on Soybean Dataset
              This code was adapted from course material by Tommi Jaakola (MIT)
"""

# python libraries
import os

# data science libraries
import numpy as np

# scikit-learn libraries
from sklearn.svm import SVC
from sklearn import metrics

# utilities
import util

######################################################################
# output code functions
######################################################################

def generate_output_codes(num_classes, code_type) :
    """
    Generate output codes for multiclass classification.
    
    For one-versus-all
        num_classifiers = num_classes
        Each binary task sets one class to +1 and the rest to -1.
        R is ordered so that the positive class is along the diagonal.
    
    For one-versus-one
        num_classifiers = num_classes choose 2
        Each binary task sets one class to +1, another class to -1, and the rest to 0.
        R is ordered so that
          the first class is positive and each following class is successively negative
          the second class is positive and each following class is successively negatie
          etc
    
    Parameters
    --------------------
        num_classes     -- int, number of classes
        code_type       -- string, type of output code
                           allowable: 'ova', 'ovo'
    
    Returns
    --------------------
        R               -- numpy array of shape (num_classes, num_classifiers),
                           output code
    """
    
    # part a: generate output codes
    # professor's solution: 13 lines
    # hint: initialize with np.ones(...) and np.zeros(...)
    import math
    if code_type == 'ova':
        num_classifiers = num_classes
        R = np.ones((num_classes, num_classifiers))
        R.fill(-1)
        np.fill_diagonal(R, 1)
    elif code_type == 'ovo':
        num_classifiers = math.factorial(num_classes)/(2 * math.factorial(num_classes - 2))
        R = np.zeros((num_classes, int(num_classifiers)))
        col = 0
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                R[i, col] = 1
                R[j, col] = -1
                col += 1
    return R


def load_code(filename) :
    """
    Load code from file.
    
    Parameters
    --------------------
        filename -- string, filename
    """
    
    # determine filename
    import util
    dir = os.path.dirname(util.__file__)
    f = os.path.join(dir, '..', 'data', filename)
    
    # load data
    with open(f, 'r') as fid :
        data = np.loadtxt(fid, delimiter=",")
    
    return data


def test_output_codes():
    R_act = generate_output_codes(3, 'ova')
    R_exp = np.array([[  1, -1, -1],
                      [ -1,  1, -1],
                      [ -1, -1,  1]])      
    assert (R_exp == R_act).all(), "'ova' incorrect"
    
    R_act = generate_output_codes(3, 'ovo')
    R_exp = np.array([[  1,  1,  0],
                      [ -1,  0,  1],
                      [  0, -1, -1]])
    assert (R_exp == R_act).all(), "'ovo' incorrect"


######################################################################
# classes
######################################################################

class MulticlassSVM :
    
    def __init__(self, R, C=1.0, kernel='linear', **kwargs) :
        """
        Multiclass SVM.
        
        Attributes
        --------------------
            R       -- numpy array of shape (num_classes, num_classifiers)
                       output code
            svms    -- list of length num_classifiers
                       binary classifiers, one for each column of R
            classes -- numpy array of shape (num_classes,) classes
        
        Parameters
        --------------------
            R       -- numpy array of shape (num_classes, num_classifiers)
                       output code
            C       -- numpy array of shape (num_classifiers,1) or float
                       penalty parameter C of the error term
            kernel  -- string, kernel type
                       see SVC documentation
            kwargs  -- additional named arguments to SVC
        """
        
        num_classes, num_classifiers = R.shape
        
        # store output code
        self.R = R
        
        # use first value of C if dimension mismatch
        try :
            if len(C) != num_classifiers :
                raise Warning("dimension mismatch between R and C " +
                                "==> using first value in C")
                C = np.ones((num_classifiers,)) * C[0]
        except :
            C = np.ones((num_classifiers,)) * C
        
        # set up and store classifier corresponding to jth column of R
        self.svms = [None for _ in range(num_classifiers)]
        for j in range(num_classifiers) :
            svm = SVC(kernel=kernel, C=C[j], **kwargs)
            self.svms[j] = svm
    
    
    def fit(self, X, y) :
        """
        Learn the multiclass classifier (based on SVMs).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), features
            y    -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            self -- an instance of self
        """
        classes = np.unique(y)
        num_classes, num_classifiers = self.R.shape
        if len(classes) != num_classes :
            raise Exception('num_classes mismatched between R and data')
        self.classes = classes    # keep track for prediction
        
        # part b: train binary classifiers
        # professor's solution: 13 lines
        
        # HERE IS ONE WAY (THERE MAY BE OTHER APPROACHES)
        #
        # keep two lists, pos_ndx and neg_ndx, that store indices
        #   of examples to classify as pos / neg for current binary task
        #
        # for each class C
        # a) find indices for which examples have class equal to C
        #    [use np.nonzero(CONDITION)[0]]
        # b) update pos_ndx and neg_ndx based on output code R[i,j]
        #    where i = class index, j = classifier index
        #
        # set X_train using X with pos_ndx and neg_ndx
        # set y_train using y with pos_ndx and neg_ndx
        #     y_train should contain only {+1,-1}
        #
        # train the binary classifier

        for i in range(num_classifiers):
            neg_class = []
            pos_class = []
            for j in range(num_classes):
                if self.R[j,i] == 1:
                    pos_class.append(self.classes[j])
                elif self.R[j,i] == -1:
                    neg_class.append(self.classes[j])

            a = np.isin(y, pos_class)
            pos_ndx = np.nonzero(a)[0]
            a = np.isin(y, neg_class)
            neg_ndx = np.nonzero(a)[0]
            pos_classes = np.ones(len(pos_ndx), dtype = int)
            neg_classes = np.full(len(neg_ndx), -1)

            X_train = np.append(X[neg_ndx], X[pos_ndx], axis = 0)
            y_train = np.append(neg_classes, pos_classes)
            self.svms[i].fit(X_train, y_train)
            # print(self.svms[i].support_)

    def predict(self, X) :
        """
        Predict the optimal class.
        
        Parameters
        --------------------
            X         -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y         -- numpy array of shape (n,), predictions
        """
        
        n,d = X.shape
        num_classes, num_classifiers = self.R.shape
        
        # setup predictions
        y = np.zeros(n)
        
        # discrim_func is a matrix that stores the discriminant function values
        #   row index represents the index of the data point
        #   column index represents the index of binary classifiers
        discrim_func = np.zeros((n,num_classifiers))
        for j in range(num_classifiers) :
            discrim_func[:,j] = self.svms[j].decision_function(X)
        
        # scan through the examples
        for i in range(n) :
            # compute votes for each class
            votes = np.dot(self.R, np.sign(discrim_func[i,:]))
            
            # predict the label as the one with the maximum votes
            ndx = np.argmax(votes)
            y[i] = self.classes[ndx]
        
        return y


######################################################################
# main
######################################################################

def main() :
    # load data
    converters = {35: ord} # label (column 35) is a character
    train_data = util.load_data("soybean_train.csv", converters)
    test_data = util.load_data("soybean_test.csv", converters)
    num_classes = len(set(train_data.y))
    
    # part a : generate output codes
    test_output_codes()
    
    # parts b-c : train component classifiers, make predictions,
    #             compare output codes
    #
    # use generate_output_codes(...) to generate OVA and OVO codes
    # use load_code(...) to load random codes
    #
    # for each output code
    #   train a multiclass SVM on training data and evaluate on test data
    #   setup the binary classifiers using the specified parameters from the handout
    #
    # if you implemented MulticlassSVM.fit(...) correctly,
    #   using OVA
    #   your first trained binary classifier should have
    #   the following indices for support vectors
    #     array([ 12,  22,  29,  37,  41,  44,  49,  55,  76, 134, 
    #            157, 161, 167, 168,   0,   3,   7])
    #   you should find 54 errors on the test data
    R_ova = generate_output_codes(num_classes, 'ova')
    R_ovo = generate_output_codes(num_classes, 'ovo')
    R1 = load_code("R1.csv")
    R2 = load_code("R2.csv")

    num_examples = len(test_data.y)
    outputs = [('ova', R_ova), ('ovo', R_ovo), ('R1',R1), ('R2',R2)]
    for outputName, output in outputs:
        clf = MulticlassSVM(output, C = 10, kernel = 'poly', degree = 4, gamma = 'scale')
        clf.fit(train_data.X,train_data.y)
        pred = clf.predict(test_data.X)
        print("{}........".format(outputName))
        correct = metrics.accuracy_score(test_data.y, pred, normalize= False)
        print('num incorrect: {}'.format(num_examples - correct))

if __name__ == "__main__" :
    main()