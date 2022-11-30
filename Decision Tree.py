"""
BBM409 Introduction to Machine Learning Lab.  Fall 2022.

Assignment 2: PART 1 :  Employee Attrition Prediction


Contributors:
Ali Argun Sayilgan   : 21827775
Mehmet Giray Nacakci :  21989009


It takes less than 10 minutes on average to get all the results.
"""

import pandas as pd
import numpy as np
import copy
import time as time
from scipy import stats

""" Preprocessing """

df = pd.read_csv("../WA_Fn-UseC_-HR-Employee-Attrition.csv")
pd.options.display.max_columns = len(df.columns)
pd.set_option('display.precision', 6)
header = df.drop(['Attrition'], axis=1).columns
ground_truth_label = "Attrition"
ground_truth_classes = df.Attrition.unique()

## Number of occurences of class types
print(df.Attrition.value_counts())
print("\n\n")

X = df.drop(['Attrition'], axis=1)
y = df.Attrition
mode_of_each_column = X.mode().iloc[0].to_list()

X = X.to_numpy()
Y = y.to_numpy()


""" ML model performance evaluation metrics """

def accuracy(preds, labels):
    """Calculates accuracy given two numpy arrays"""
    return np.mean(preds == labels)


def confusion_matrix(preds, labels, num_of_labels):
    """Creates a confusion matrix from given two numpy arrays"""
    unique_label_classes = np.sort(np.unique(labels))
    int_encoded_labels = np.searchsorted(unique_label_classes, labels)
    int_encoded_pred = np.searchsorted(unique_label_classes, preds)

    matrix = np.zeros((num_of_labels, num_of_labels))
    for i in range(len(preds)):
        matrix[int_encoded_pred[i], int_encoded_labels[i]] += 1
    return matrix


def precision_and_recall(preds, labels):
    """Returns individual Precision and Recall values of each class"""
    num_of_labels = len(np.unique(labels))
    matrix = confusion_matrix(preds, labels, num_of_labels)
    r = []
    p = []

    number_of_not_NA_precision, number_of_not_NA_recall, precision_sum, recall_sum = 0, 0, 0, 0

    for i in range(num_of_labels):
        TP = float(matrix[i, i])
        TP_FP = np.sum(matrix[i, :])
        TP_FN = np.sum(matrix[:, i])

        if TP_FN != 0:
            recall_sum += TP / TP_FN
            number_of_not_NA_recall += 1

        if TP_FP != 0:
            precision_sum += TP / TP_FP
            number_of_not_NA_precision += 1

    recall = recall_sum / number_of_not_NA_recall if number_of_not_NA_recall != 0 else "NA"
    precision = precision_sum / number_of_not_NA_precision if number_of_not_NA_precision != 0 else "NA"

    return (precision, recall)


def fscore(preds, labels):
    """Calculates macro f score given two numpy arrays"""
    pred_labels, pred_numberized = np.unique(preds, return_inverse=True)
    labels, labels_numberized = np.unique(labels, return_inverse=True)
    p, r = precision_and_recall(pred_numberized, labels_numberized)
    if p == "NA" or r == "NA" or p + r == 0:
        return "NA"

    return 2 * p * r / (p + r)




""" Helper functions that are used at ID3 Decision Tree Algorithm"""


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


def class_counts(numpy_arr):
    """Returns the unique elements that are inside the
    numpy array, and their individual counts"""

    if len(numpy_arr.shape) == 2:
        valueCounts = pd.Series(numpy_arr[:, 0]).value_counts()
        return np.array(valueCounts.keys().to_list()), np.array(valueCounts.to_list())
    else:
        valueCounts = pd.Series(numpy_arr).value_counts()
        return np.array(valueCounts.keys().to_list()), np.array(valueCounts.to_list())


class Question:
    """Questions are used to partition a dataset."""

    def __init__(self, column, partition_values, partition_type):
        self.column = column
        self.partition_values = partition_values
        self.partition_type = partition_type

    def match(self, row):
        """Answers question for a particular row (sample).
        Return index of which Child Branch to go."""

        if self.partition_type == "discrete":
            for t, val in enumerate(self.partition_values):
                if row[self.column] == val:
                    return t

            # if test data has a value not seen while training
            # unseen label is getting considered as mode label of that column
            for t, val in enumerate(self.partition_values):
                if mode_of_each_column[self.column] == val:
                    return t
            return 0


        # for continuous value, assign index of appropriate interval node child
        elif self.partition_type == "interval":
            if len(self.partition_values) == 1:
                return 0

            for k in range(1, len(self.partition_values) - 1):
                if self.partition_values[k - 1] <= row[self.column] < self.partition_values[k]:
                    return k
            if row[self.column] >= self.partition_values[-2]:
                return len(self.partition_values) - 1
            if row[self.column] < self.partition_values[0]:
                return 0

    def __repr__(self):
        """Helper method to print question in readable form"""

        if self.partition_type == "discrete":
            return "which {} category ?".format(header[self.column])

        elif self.partition_type == "interval":
            return "which {} interval ?".format(header[self.column])

        equality_condition = "=="
        if is_numeric(self.value):
            if self.partition_type == "discrete":
                return "Is {} {} {} ?".format(header[self.column], equality_condition, str(self.value))
            elif self.partition_type == "interval":
                return "Is {} at range {} ?".format(header[self.column], str(self.value))

            return "Is {} {} {} ?".format(header[self.column], equality_condition, str(self.value))



def entropy(Y):
    """Calculate entropy of data from labels"""
    labels, counts = class_counts(Y)
    probs = counts / float(Y.shape[0])
    entropy = - np.sum(probs * np.log2(probs))
    return entropy


def information_gain(y, intervals_indexes, current_entropy):
    """Information Gain:  The Entropy of the starting node, minus the weighted entropy of child nodes. """
    sum_ = 0
    len_sum = 0

    # weighted sum of entropies of each split (or interval)
    for interval_indices in intervals_indexes:
        sum_ += float(len(interval_indices) * entropy(y[interval_indices]))
        len_sum += len(interval_indices)

    return current_entropy - sum_ / len_sum


def gain_ratio(y, intervals_indexes, current_entropy):
    """ Information_Gain divided by entropies of sub-dataset proportions. """

    # "split information" measure
    len_sum = 0
    for interval_indices in intervals_indexes:
        len_sum += len(interval_indices)

    sum_ = 0
    for interval_indices in intervals_indexes:
        sum_ += len(interval_indices) * np.log2(len_sum / len(interval_indices))
    split_information = sum_ / len_sum

    # prevent division by zero
    if split_information < 0.0001:
        split_information = 0.0001

    return information_gain(y, intervals_indexes, current_entropy) / split_information




""" ID3 Decision Tree Algorithm """

class Node:
    """ Holds a reference to the question, and the child nodes that is partitioned by the question. """

    def __init__(self, child_branches, question, gain, y):
        self.child_branches = child_branches
        self.question = question
        self.gain = gain

        counts = np.column_stack(class_counts(y))
        self.label_counts = {row[0]: row[1] for row in counts}



class Leaf:
    """
    Leaf node contains a dictionary of counts of classes.
    Output is usually the majority class.
    """

    def __init__(self, y):

        # y passed as predictions dictionary while pruning the tree
        if (type(y) is dict):
            self.predictions = y
        else:
            counts = np.column_stack(class_counts(y))
            self.predictions = {row[0]: row[1] for row in counts}



class Decision_Tree:

    def __init__(self, maximum_depth):
        self.depth = maximum_depth

    def train(self, X_train, y_train):
        mode_of_each_column = stats.mode(X_train)[0][0] # is used when there is unseen labels inside test data
        self.root = self._build_tree(X_train, y_train, self.depth, [])

    def find_best_split(self, X, y, used_attributes):
        """
        Find the best question to ask by iterating over every feature.
        Choose the best possible attribute and best possible split in terms of information gain.
        """
        copy_used_attributes = used_attributes[0:]
        best_gain = 0
        best_question = None
        best_partition = []
        used_attribute = 0
        current_entropy = entropy(y)
        n_features = X.shape[1]

        # To not use any attribute (to split the data-subset) twice, in a single path from root to leaf.
        for col in range(n_features):
            if col in copy_used_attributes:
                continue

            # if this column is a continuous attribute: it needs Discretization
            if is_numeric(X[0, col]):

                this_column = X[:, col]
                min_ = np.min(this_column)
                max_ = np.max(this_column)

                # rarely, but in the case that all values are same:
                if min_ == max_:  # no split (so-called split into only one branch, not two.)
                    partition_to_indexesOfSamples = [X[:, col] <= max_]
                    gain = information_gain(y, partition_to_indexesOfSamples, current_entropy)
                    if gain >= best_gain:
                        best_gain = gain
                        best_partition = partition_to_indexesOfSamples
                        best_question = Question(col, [min_], "interval")
                        used_attribute = col

                else:
                    # find best number of splits in terms of information gain
                    for n_splits in range(2, 10):
                        interval_size = (max_ - min_) / n_splits

                        range_values = []
                        partitions_to_indexesOfSamples = []

                        # create partitions (intervals)
                        # n_split = n many intervals
                        for m in range(n_splits):

                            range_values.append(min_ + (m + 1) * interval_size)
                            indexes_in_this_interval = []

                            if m == 0:
                                indexes_in_this_interval = X[:, col] < min_ + interval_size
                            elif m == n_splits - 1:
                                indexes_in_this_interval = X[:, col] >= min_ + m * interval_size
                            else:
                                indexes_in_this_interval = filter_interval(X, col, min_, m, interval_size)

                            partitions_to_indexesOfSamples.append(indexes_in_this_interval)

                        """ Do not choose such partitioning that some empty tree branches will be created. """

                        empty_interval_alert = False

                        for indexes_in_this_interval in partitions_to_indexesOfSamples:
                            if len(indexes_in_this_interval) < 1:
                                empty_interval_alert = True
                                break

                        # skip the current n_split value. n_split many intervals is not what we are looking for.
                        if empty_interval_alert:
                            continue

                        # for n_splits, calculate information gain.
                        # gain = information_gain(y, partitions_to_indexesOfSamples, current_entropy)
                        #  gain /= n_splits
                        gain = gain_ratio(y, partitions_to_indexesOfSamples, current_entropy)

                        if gain >= best_gain:
                            best_gain = gain
                            best_partition = partitions_to_indexesOfSamples
                            best_question = Question(col, range_values, "interval")
                            used_attribute = col


            # this column is a discrete attribute
            else:
                # multi-way split: as many branches as len(unique_values)
                unique_values = np.unique(X[:, col])
                partitions_to_indexesOfSamples = []

                for value in unique_values:
                    indexes_in_this_partition = X[:, col] == value
                    partitions_to_indexesOfSamples.append(indexes_in_this_partition)

                # gain = information_gain(y, partitions_to_indexesOfSamples, current_entropy)
                # gain /= len(unique_values)
                gain = gain_ratio(y, partitions_to_indexesOfSamples, current_entropy)

                if gain >= best_gain:
                    best_gain = gain
                    best_partition = partitions_to_indexesOfSamples
                    best_question = Question(col, unique_values, "discrete")
                    used_attribute = col

        copy_used_attributes.append(used_attribute)
        return best_gain, best_partition, best_question, copy_used_attributes

    def _build_tree(self, X_train, y_train, depth, used_attributes):
        """ recursive decision tree building function """

        if X_train.shape[0] < 2:
            return Leaf(y_train)

        if depth < 1:
            return Leaf(y_train)

        all_samples_in_subset_are_same_label = np.all(y_train == y_train[0])
        if all_samples_in_subset_are_same_label:
            return Leaf(y_train)

        gain, partitions, question, used_attributes = self.find_best_split(X_train, y_train, used_attributes)

        if gain == 0 or question is None:
            return Leaf(y_train)

        child_branches = []
        for indices_in_this_partition in partitions:
            child_branches.append(
                self._build_tree(X_train[indices_in_this_partition], y_train[indices_in_this_partition], depth - 1,
                                 used_attributes))

        return Node(child_branches, question, gain, y_train)

    def predict(self, row):
        prediction = self._classify(row, self.root)
        return prediction

    def _classify(self, row, node):
        """Recursively traverses the decision tree towards a prediction(Leaf node)"""

        # Base case: At a leaf node
        if isinstance(node, Leaf):
            max_count = 0
            max_label = None

            for k, v in node.predictions.items():
                if int(v) >= max_count:
                    max_count = int(v)
                    max_label = k

            return max_label

        branch_to_go = node.child_branches[node.question.match(row)]
        return self._classify(row, branch_to_go)

    def test(self, X_test, y_test):
        """Returns accuracy and fscore of test dataset"""
        # Not vectorized.
        preds = []
        for i, row in enumerate(X_test):
            preds.append(self.predict(row))

        preds = np.array(preds)

        a = accuracy(preds, y_test)
        f = fscore(preds, y_test)
        p, r = precision_and_recall(preds, y_test)

        return a, f, p, r


def filter_interval(X, col, min_, m, interval_size):
    indices = []
    for j in range(np.shape(X)[0]):
        if (min_ + m * interval_size) <= X[j, col] < (min_ + (m + 1) * interval_size):
            indices.append(j)
    return indices




""" Visual Representation of Decision Tree """

def print_tree(node, markerStr="+- ", levelMarkers=[]):
    level = len(levelMarkers)
    emptyStr = " " * 15
    connectionStr = "|" + emptyStr[:-4]
    mapper = lambda draw: connectionStr if draw else emptyStr
    markers = "".join(map(mapper, levelMarkers[:-1]))
    markers += markerStr if level > 0 else ""

    if isinstance(node, Leaf):
        print(f"{markers}Prediction ({ground_truth_label}): {node.predictions}")
        return

    print(f"{markers}{node.question}")
    if (node.question.partition_type == "discrete"):
        for i, child in enumerate(node.child_branches):
            label_answer = "(" + str(node.question.partition_values[i]) + ") -- "
            isLast = i == len(node.child_branches) - 1
            print_tree(child,
                       label_answer, [*levelMarkers, not isLast])

    else:
        for i, child in enumerate(node.child_branches):
            if (i == 0):
                range_answer = "(" + " ," + format(node.question.partition_values[i], '.2f') + ") -- "
            elif (i == len(node.child_branches) - 1):
                range_answer = "[" + format(node.question.partition_values[i - 1], '.2f') + ", ) -- "
            else:
                range_answer = "[" + format(node.question.partition_values[i - 1], '.2f') + "," + format(
                    node.question.partition_values[i], '.2f') + ") -- "

            isLast = i == len(node.child_branches) - 1
            print_tree(child,
                       range_answer, [*levelMarkers, not isLast])





""" Error Analysis for Classification """

from sklearn.model_selection import KFold

kf = KFold(n_splits=5, random_state=24, shuffle=True)
print(""" \n---------------------- PART 1 ---------------------------------\n\n\n""")

scores_array = []
for max_depth in [2, 3, 4, 5, 6, 15]:
    row = []
    print("max_depth : " + str(max_depth))
    i = 1

    test_F_scores, test_accuracies, test_ps, test_rs = [], [], [], []
    train_F_scores, train_accuracies, train_ps, train_rs = [], [], [], []

    start = time.time()
    for train_index, test_index in kf.split(X):  # Each FOLD

        print("   fold" + str(i) + " :    ", end=" ")
        i += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model = Decision_Tree(max_depth)
        model.train(X_train, y_train)

        train_acc, train_f, train_p, train_r = model.test(X_train, y_train)
        print("Train : F1 Score: {:.3f}, Accuracy: {:.3f}     ".format(train_f, train_acc), end="")

        test_acc, test_f, test_p, test_r = model.test(X_test, y_test)

        test_accuracies.append(test_acc)
        test_F_scores.append(test_f)
        test_ps.append(test_p)
        test_rs.append(test_r)

        train_accuracies.append(train_acc)
        train_F_scores.append(train_f)
        train_ps.append(train_p)
        train_rs.append(train_r)
        print(" TEST : F1 Score: {:.3f} ,  Accuracy: {:.3f}  , Precision: {:.3f} , Recall: {:.3f}".format(test_f,
                                                                                                          test_acc,
                                                                                                          test_p,
                                                                                                          test_r))

    print(
        "   AVERAGE :                                                        F1 Score: {:.3f} ,  Accuracy: {:.3f}  , Precision: {:.3f} , Recall: {:.3f}".format(
            sum(test_F_scores) / 5, sum(test_accuracies) / 5, sum(test_ps) / 5, sum(test_rs) / 5))

    row.extend([max_depth, sum(train_F_scores) / 5, sum(test_F_scores) / 5, sum(train_accuracies) / 5,
                sum(test_accuracies) / 5,
                sum(train_ps) / 5, sum(test_ps) / 5, sum(train_rs) / 5, sum(test_rs) / 5
                ])
    scores_array.append(row)
    finish = time.time()
    seconds = finish - start
    minutes = seconds // 60
    seconds -= 60 * minutes
    print('Elapsed time: %d:%d   minutes:seconds \n' % (minutes, seconds))

scores_df = pd.DataFrame(scores_array,
                         columns=['max_depth', 'train_f1', 'test_f1', 'train_accuracy', 'test_accuracy',
                                  'train_precision', 'test_precision', 'train_recall', 'test_recall'
                                  ])
print("\nAverage of 5 folds:\n")
print(scores_df)



""" Print Overfitted Tree"""
print("\n\n Overfitted Tree : max_depth=15 : \n")

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=24, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    model = Decision_Tree(15)
    model.train(X_train, y_train)

    train_acc, train_f, train_p, train_r = model.test(X_train, y_train)
    print("   Train : F1 Score: {:.3f}, Accuracy: {:.3f}     ".format(train_f, train_acc), end="")

    test_acc, test_f, test_p, test_r = model.test(X_test, y_test)
    print(" TEST : F1 Score: {:.3f} ,  Accuracy: {:.3f}   , Precision: {:.3f} , Recall: {:.3f}".format(test_f,
                                                                                                       test_acc,
                                                                                                       test_p,
                                                                                                       test_r))

    break  # we only need first fold

print("   This tree contains very long text lines and might not print well, due to line wrapping settings. Output exceeds size limit.\n")
print("\n Overfitted Tree : max_depth=15 : \n")
print_tree(model.root)





""" Print Best Performed Tree """

kf = KFold(n_splits=5, random_state=24, shuffle=True)
this_fold = 0
for train_index, test_index in kf.split(X):

    # we only need fold_4
    this_fold += 1
    if this_fold != 4:
        continue

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    model = Decision_Tree(2)
    model.train(X_train, y_train)

    train_acc, train_f, train_p, train_r = model.test(X_train, y_train)

    test_acc, test_f, test_p, test_r = model.test(X_test, y_test)
    break

print("\n\n\n BEST PERFORMING TREE  max_depth=2 : \n")
print_tree(model.root)



""" Misclassification Examples"""

df_test_data = df.filter(items=test_index, axis=0)

# In this branch, "Yes" Test samples will be misclassified as "No", since majority in Training is "No" .
df_misclassified = df_test_data[
    (df_test_data['MonthlyIncome'] < 10504.00) & (df_test_data['YearsAtCompany'] < 13.33) & (
                df_test_data['Attrition'] == 'Yes')]
df_misclassified = df_misclassified[['MonthlyIncome', 'YearsAtCompany', 'Attrition']]
df_correct_classified = df_test_data[
    (df_test_data['MonthlyIncome'] < 10504.00) & (df_test_data['YearsAtCompany'] < 13.33) & (
                df_test_data['Attrition'] == 'No')]
df_correct_classified = df_correct_classified[['MonthlyIncome', 'YearsAtCompany', 'Attrition']]

# reordering columns
df_misclassified.insert(0, 'Attrition', df_misclassified.pop("Attrition"))
df_misclassified.insert(0, 'MonthlyIncome', df_misclassified.pop("MonthlyIncome"))
df_misclassified.insert(0, 'YearsAtCompany', df_misclassified.pop("YearsAtCompany"))

df_correct_classified.insert(0, 'Attrition', df_correct_classified.pop("Attrition"))
df_correct_classified.insert(0, 'MonthlyIncome', df_correct_classified.pop("MonthlyIncome"))
df_correct_classified.insert(0, 'YearsAtCompany', df_correct_classified.pop("YearsAtCompany"))

print("\nHere are a few misclassified samples at indexes, in above BEST model: ")
print(df_misclassified.head(10).T)
print("... as expected.\n\n\n")





""" ---------------------- Assignment 2: PART 2 :  Pruning Decision Tree  -------------------------------"""


print("""\n ---------------------- Assignment 2: PART 2 :  Pruning Decision Tree  -------------------------------\n\n\n""")

# Split dataset into: (60 % train) - (20 % validation) - (20 % test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=24, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=24, shuffle=True)



""" Pruning methods: """

def _find_twigs(Node, twigs):
    if isinstance(Node, Leaf):
        return
    current_is_twig = True
    for child in Node.child_branches:
        if not isinstance(child, Leaf):
            _find_twigs(child, twigs)
            current_is_twig = False

    if current_is_twig:
        twigs.append(Node)


def find_twig_with_least_info_gain(root_):
    twigs = []
    _find_twigs(root_, twigs)

    if len(twigs) == 0:
        return None
    min_gain = twigs[0].gain
    min_index = 0

    for i, twig in enumerate(twigs):
        if twig.gain < min_gain:
            min_gain = twig.gain
            min_index = i
    return twigs[i], len(twigs)


def remove_twig_from_tree(node, twig):
    if isinstance(node, Leaf) or node is None:
        None
    else:

        for i, child in enumerate(node.child_branches):
            if child is twig:
                node.child_branches[i] = Leaf(child.label_counts)
                print("pruned twig predictions: " + str(node.child_branches[i].predictions))
                break
            else:
                remove_twig_from_tree(child, twig)


def remove_twig_from_model(model, twig):
    remove_twig_from_tree(model.root, twig)


def prune_by_least_info_gain(model, X_val, y_val):
    # recursive function, but "model" does not get assigned deeper nodes in the tree, stays at root level

    prev_acc, prev_f, prev_p, prev_r = model.test(X_val, y_val)

    modified_model = copy.deepcopy(model)
    twig, twig_count = find_twig_with_least_info_gain(modified_model.root)

    if twig == modified_model.root:
        return model
    remove_twig_from_model(modified_model, twig)

    acc, f, p, r = modified_model.test(X_val, y_val)

    print("Amount of twigs at tree {} \t\tOld acc {}, new acc {}\n".format(twig_count, prev_acc, acc))

    if acc >= prev_acc:
        return prune_by_least_info_gain(modified_model, X_val, y_val)
    return model




"""Pruning Decision Tree with 4 maximum_depth"""

# First, create a Decision Tree based on Training set.Then prune it.
model = Decision_Tree(4)
model.train(X_train, y_train)
print("PRUNING MAX_DEPTH = 4 TREE: \n")
pruned_model = prune_by_least_info_gain(model, X_val, y_val)


#### Comparison of Before and After Pruning:

print("\nBEFORE PRUNING:  (MAX_DEPTH = 4 TREE)\n")
print_tree(model.root)
print("\n\n")
print("AFTER PRUNING:   (MAX_DEPTH = 4 TREE)\n")
print_tree(pruned_model.root)
print("\n")


#### Comparing Before and After Pruning Scores:

print("BEFORE PRUNING  (MAX_DEPTH = 4 TREE) : ")
train_acc, train_f, train_p, train_r = model.test(X_train, y_train)
print("  Train :      F1 Score: {:.3f},   Accuracy: {:.3f}".format(train_f, train_acc))
val_acc, val_f, val_p, val_r = model.test(X_val, y_val)
print("  VALIDATION : F1 Score: {:.3f} ,  Accuracy: {:.3f}".format(val_f, val_acc))
tes_acc, tes_f, tes_p, tes_r = model.test(X_test, y_test)
print("  TEST :       F1 Score: {:.3f} ,  Accuracy: {:.3f} \n".format(tes_f, tes_acc))

print("AFTER PRUNING  (MAX_DEPTH = 4 TREE) :")
train_acc, train_f, train_p, train_r = pruned_model.test(X_train, y_train)
print("  Train :      F1 Score: {:.3f},   Accuracy: {:.3f}".format(train_f, train_acc))
val_acc, val_f, val_p, val_r = pruned_model.test(X_val, y_val)
print("  VALIDATION : F1 Score: {:.3f} ,  Accuracy: {:.3f}".format(val_f, val_acc))
tes_acc, tes_f, tes_p, tes_r = pruned_model.test(X_test, y_test)
print("  TEST :       F1 Score: {:.3f} ,  Accuracy: {:.3f} \n".format(tes_f, tes_acc))



""" Pruning Decision Tree with 15 maximum_depth"""

# First, create a Decision Tree based on Training set.Then prune it.
model = Decision_Tree(15)
model.train(X_train, y_train)
print("\n\n\n\nPRUNING MAX_DEPTH = 15 TREE: \n")
pruned_model = prune_by_least_info_gain(model, X_val, y_val)


#### Comparison of Before and After Pruning:

print("\nBEFORE PRUNING  (MAX_DEPTH = 15 TREE) :\n")
print("This tree contains very long text lines and might not print well, due to line wrapping settings. Output exceeds size limit.\n")
print_tree(model.root)
print("\n\nAFTER PRUNING  (MAX_DEPTH = 15 TREE) :\n")
print("This tree contains very long text lines and might not print well, due to line wrapping settings. Output exceeds size limit.\n")
print_tree(pruned_model.root)
print("\n")


#### Comparing Before and After Pruning Scores:

print("\nBEFORE PRUNING  (MAX_DEPTH = 15 TREE) :")
train_acc, train_f, train_p, train_r = model.test(X_train, y_train)
print("  Train :      F1 Score: {:.3f},   Accuracy: {:.3f}".format(train_f, train_acc))
val_acc, val_f, val_p, val_r = model.test(X_val, y_val)
print("  VALIDATION : F1 Score: {:.3f} ,  Accuracy: {:.3f}".format(val_f, val_acc))
tes_acc, tes_f, tes_p, tes_r = model.test(X_test, y_test)
print("  TEST :       F1 Score: {:.3f} ,  Accuracy: {:.3f} \n\n".format(tes_f, tes_acc))

print("\nAFTER PRUNING  (MAX_DEPTH = 15 TREE) :")
train_acc, train_f, train_p, train_r = pruned_model.test(X_train, y_train)
print("  Train :      F1 Score: {:.3f},   Accuracy: {:.3f}".format(train_f, train_acc))
val_acc, val_f, val_p, val_r = pruned_model.test(X_val, y_val)
print("  VALIDATION : F1 Score: {:.3f} ,  Accuracy: {:.3f}".format(val_f, val_acc))
tes_acc, tes_f, tes_p, tes_r = pruned_model.test(X_test, y_test)
print("  TEST :       F1 Score: {:.3f} ,  Accuracy: {:.3f} \n".format(tes_f, tes_acc))

print("\n\n ... finished. ")
