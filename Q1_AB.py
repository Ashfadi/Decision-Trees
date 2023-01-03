#!/usr/bin/env python
# coding: utf-8

# In[2]:


def main():
    print('START Q1_AB\n')
    
    # Importing Libraries and Dataset
    import numpy as np

    def clean_data(line):
        return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')
    def fetch_data(filename):
        with open(filename, 'r') as f:
            input_data = f.readlines()    
            clean_input = list(map(clean_data, input_data))    
            f.close()
        return clean_input    
    def readFile(dataset_path):
        input_data = fetch_data(dataset_path)
        input_np = np.array(input_data)
        return input_np

    training = './datasets/Q1_train.txt'
    test = './datasets/Q1_test.txt'
    Training_Data = readFile(training)
    Test_Data = readFile(test)
    
    # Converting Label of Training Data to '0' and '1'
    for i in Training_Data:
        if i[3]=='W':
            i[3]=i[3].replace('W','1')
            i[3]=int(i[3])
        else:
            i[3]=i[3].replace('M','0')
            i[3]=int(i[3])
    Training_Data=Training_Data.astype(float)
    
    # Converting Label of Test Data to '0' and '1'
    for i in Test_Data:
        if i[3]=='M':
            i[3]=i[3].replace('M','0')
            i[3]=int(i[3])
        else:
            i[3]=i[3].replace('W','1')
            i[3]=int(i[3])
    Test_Data=Test_Data.astype(float)
    
    # Assigning feature columns to 'X' and label column to 'Y' for Training and Test Data
    X_Train = Training_Data[:,[0,1,2]]
    Y_Train = Training_Data[:,[3]]
    X_Test = Test_Data[:,[0,1,2]]
    Y_Test = Test_Data[:,[3]]
    
    class Node():
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
            # for decision node
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.info_gain = info_gain
            # for leaf node
            self.value = value
    
    class DecisionTreeClassifier():
        def __init__(self, min_samples_split, max_depth):
            # initialize the root of the tree 
            self.root = None
            # stopping conditions
            self.min_samples_split = min_samples_split
            self.max_depth = max_depth

        # function to compute entropy
        def entropy(self, y):
            class_labels = np.unique(y)
            entropy = 0
            for cls in class_labels:
                p_cls = len(y[y == cls]) / len(y)
                entropy += -p_cls * np.log2(p_cls)
            return entropy

        # function to compute information gain
        def information_gain(self, parent, l_child, r_child):
            weight_l = len(l_child) / len(parent)
            weight_r = len(r_child) / len(parent)
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
            return gain

        # function to split the data
        def split(self, dataset, feature_index, threshold):

            dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
            dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
            return dataset_left, dataset_right

        # function to find the best split
        def get_best_split(self, dataset, num_samples, num_features):
            # dictionary to store the best split
            best_split = {}
            max_info_gain = -float("inf")
            # loop over all the features
            for feature_index in range(num_features):
                feature_values = dataset[:, feature_index]
                possible_thresholds = np.unique(feature_values)
                # loop over all the feature values present in the data
                for threshold in possible_thresholds:
                    # get current split
                    dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                    # check if childs are not null
                    if len(dataset_left)>0 and len(dataset_right)>0:
                        y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                        # compute information gain
                        curr_info_gain = self.information_gain(y, left_y, right_y)
                        # update the best split if needed
                        if curr_info_gain>max_info_gain:
                            best_split["feature_index"] = feature_index
                            best_split["threshold"] = threshold
                            best_split["dataset_left"] = dataset_left
                            best_split["dataset_right"] = dataset_right
                            best_split["info_gain"] = curr_info_gain
                            max_info_gain = curr_info_gain           
            # return best split
            return best_split

        # function to compute leaf node
        def calculate_leaf_value(self, Y):
            Y = list(Y)
            return max(Y, key=Y.count)

        # recursive function to build the tree
        def build_tree(self, dataset, curr_depth=0):
            X, Y = dataset[:,:-1], dataset[:,-1]
            num_samples, num_features = np.shape(X)
            # split until stopping conditions are met
            if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
                # find the best split
                best_split = self.get_best_split(dataset, num_samples, num_features)
                # check if information gain is positive
                if best_split["info_gain"]>0:
                    # recur left
                    left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                    # recur right
                    right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                    # return decision node
                    return Node(best_split["feature_index"], best_split["threshold"], 
                                left_subtree, right_subtree, best_split["info_gain"])
            # compute leaf node
            leaf_value = self.calculate_leaf_value(Y)
            # return leaf node
            return Node(value=leaf_value)

        # function to print the tree
        def print_tree(self, tree=None, indent=" "):
            if not tree:
                tree = self.root
            if tree.value is not None:
                print(tree.value)
            else:
                print("X_"+str(tree.feature_index), "<=", tree.threshold, ", Information_Gain =", tree.info_gain)
                print("%sleft:" % (indent), end="")
                self.print_tree(tree.left, indent + indent)
                print("%sright:" % (indent), end="")
                self.print_tree(tree.right, indent + indent)

        # function to train the tree
        def fit(self, X, Y):
            dataset = np.concatenate((X, Y), axis=1)
            self.root = self.build_tree(dataset)

        # function to predict a single data point
        def make_prediction(self, x, tree):
            if tree.value!=None: return tree.value
            feature_val = x[tree.feature_index]
            if feature_val<=tree.threshold:
                return self.make_prediction(x, tree.left)
            else:
                return self.make_prediction(x, tree.right)

        # function to predict dataset
        def predict(self, X):
            preditions = [self.make_prediction(x, self.root) for x in X]
            return preditions
        
    # Applying decision tree classifier to our training data
    classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=5)
    classifier.fit(X_Train,Y_Train)
    classifier.print_tree()
    
    # Predicting labels for test and training data, calculating accuracy between predicted and orginal data
    def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    for i in range(1,6):
        classifier = DecisionTreeClassifier(min_samples_split=2, max_depth=i)
        classifier.fit(X_Train,Y_Train)
        Y_pred1 = classifier.predict(X_Train)
        accuracy1 = accuracy_metric(Y_Train, Y_pred1)
        Y_pred2 = classifier.predict(X_Test)
        accuracy2 = accuracy_metric(Y_Test, Y_pred2)
        print('DEPTH =',i,'\nAccuracy | Train =',accuracy1,'| Test =',accuracy2)
    
    # For depths "4" and "5", it indicates overfitting because accuracy is increased to 100% for training data while it decreased for test data.  
    
    print('\nEND Q1_AB\n')


if __name__ == "__main__":
    main()


# In[ ]:




