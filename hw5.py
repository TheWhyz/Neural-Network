# Jaydan Escober, CS 422 1002, HW5
import torch
from torch import nn
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics

# Read the data into a pandas dataframe
data = pd.read_csv("MNIST.csv")

# X = features
X = data.iloc[:, 1:].to_numpy()
# y = labels
y = data.iloc[:, 0].to_numpy()

# Create tensors of both X and y
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y).type(torch.LongTensor)
# Get the amount of features
shape = X.shape
shape = list(shape)
features = shape[1]


# Neural Network model
class NNClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.layer1 = nn.Linear(n_features, 250)
        self.layer2 = nn.Linear(250, 50)
        self.output = nn.Linear(50, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# Function used to calculate accuracy
def accuracy_func(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# Create 5 folds and list to store accuracies of the folds
cv = KFold(n_splits=5, shuffle=True)
accuracies = []


fold = 1
print("Learning Rates for Fold 1:")
# For every split
for train_index, test_index in cv.split(X, y):
    # Split the data into train and test datasets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print(y_train.shape)

    # Create the model, loss function, and optimizer
    NNmodel = NNClassifier(n_features=features, n_classes=10)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(NNmodel.parameters(), lr=0.001)

    n_epochs = 41
    # Train the model
    for epoch in range(0, n_epochs):
        NNmodel.train()

        # Pass X_train data into model
        y_train_data = NNmodel(X_train)
        # Get classification predictions
        y_train_pred = torch.softmax(y_train_data, dim=1).argmax(dim=1)

        # Calculate loss and accuracy
        loss = loss_func(y_train_data, y_train)
        acc = accuracy_func(y_true=y_train, y_pred=y_train_pred)

        # Print the learning rate of the first fold
        if fold == 1:
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}%")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get predictions from the trained model
    NNmodel.eval()
    with torch.inference_mode():
        y_test_data = NNmodel(X_test)

    y_data_nums = torch.softmax(y_test_data, dim=1)
    y_test_pred = y_data_nums.argmax(dim=1)

    # Get the accuracy of the predictions and insert into the list
    accuracy = metrics.accuracy_score(y_test, y_test_pred)
    print(f"Fold {fold} accuracy: {accuracy * 100}%")
    fold += 1
    accuracies.append(accuracy)

# Print the average accuracy
print(f"Average Accuracy: {sum(accuracies)/5 * 100}%")
