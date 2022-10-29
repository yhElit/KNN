import numpy as np
import sklearn.metrics
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler


def max_elements(seq):
    # Return list of positions of the largest element
    max_indices = []
    if seq:
        max_val = seq[0]
        for i, val in ((i, val) for i, val in enumerate(seq) if val >= max_val):
            if val == max_val:
                max_indices.append(i)
            else:
                max_val = val
                max_indices = [i]

    return max_indices


class MyKNeighborsClassifier:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predicted_labels = []

        # Find the distance/labels for each training data examples
        for x in x_test:
            # Euclidean distance between one point and the train set
            distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.x_train]

            # k nearest neighbors for one point
            neighbors = np.argsort(distances)[:self.n_neighbors]

            # Labels of the k nearest neighbors
            neighbors_labels = [self.y_train[x] for x in neighbors]

            # Majority vote
            most_common = max(set(neighbors_labels), key=neighbors_labels.count)

            predicted_labels.append(most_common)

        return np.array(predicted_labels)


def model_test(x_train, y_train, x_test, y_test):
    accuracies = []
    # define k neighbors
    ks = range(1, 25, 1)
    for k in ks:
        # print(k)

        # Classifier implementing the k-nearest neighbors vote
        knn_self = MyKNeighborsClassifier(n_neighbors=k)

        # Fit classifier with training set
        knn_self.fit(x_train, y_train)

        # Predict labels for the svm01 set
        predictions = knn_self.predict(x_test)

        # Calculate the accuracy
        accuracy = sklearn.metrics.accuracy_score(y_test, predictions)

        # Accuracy self
        # accuracy = np.sum(prediction == y_test) / len(y_test)*100

        accuracies.append(accuracy)

    # Visualize accuracy and k
    fig, ax = plt.subplots()
    ax.plot(ks, accuracies)
    ax.set(xlabel="k", ylabel="Accuracy")

    return accuracies, max_elements(accuracies)


def main():
    # Load the dataset
    dataset = datasets.load_wine()
    #dataset = datasets.load_digits()
    #dataset = datasets.load_breast_cancer()

    # Create features and targets
    x = dataset.data
    y = dataset.target

    # Split dataset into random training set and svm01 set with fold cross validation
    kf = KFold(n_splits=10, shuffle=True)
    n_accuracies = []
    fs_accuracies = []
    n_bestK = []
    fs_bestK = []
    counter = 1

    for train, test in kf.split(x):
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        print("Split", counter, "started")
        counter += 1
        # print("x_train:", x_train)
        # print("x_test:", x_test)

        # Test  model across varying ks
        test1 = model_test(x_train, y_train, x_test, y_test)

        # Feature scaling (normalisation) with standardization
        ss = StandardScaler().fit(x_train)
        x_train, x_test = ss.transform(x_train), ss.transform(x_test)

        # Test  model across varying ks with feature scaling
        test2 = model_test(x_train, y_train, x_test, y_test)

        # track best value for k and accuracy of the model
        n_accuracies.append((sum(test1[0]) / float(len(test1[0]))))
        fs_accuracies.append((sum(test2[0]) / float(len(test2[0]))))
        n_bestK.append(test1[1])
        fs_bestK.append(test2[1])

    n_bestK = [j for sub in n_bestK for j in sub]
    fs_bestK = [j for sub in fs_bestK for j in sub]

    print("KNN has a mean accuracy of", (sum(n_accuracies) / float(len(n_accuracies))) * 100, "%")
    print("best mean for k:", max(set(n_bestK), key=n_bestK.count)+1)

    print("KNN with feature scaling has a mean accuracy of", (sum(fs_accuracies) / float(len(fs_accuracies))) * 100,
          "%")
    print("best mean for k:", max(set(fs_bestK), key=fs_bestK.count)+1)
    #plt.show()


if __name__ == '__main__':
    main()
