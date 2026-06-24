import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch import nn

df = pd.read_csv('./data/SMSSpamCollection', sep='\t', names=['title', 'message'])
df["spam"] = df["title"] == "spam"
df.drop("title",axis=1, inplace=True)


# convert the text data into numerical features using CountVectorizer, which creates a bag-of-words representation of the text data by counting the frequency of max_words in the text data. The max_features parameter is set to 1000, which means that only the top 1000 most frequent words will be used as features.
vectorizer = CountVectorizer(max_features=1000)
messages = vectorizer.fit_transform(df["message"])

print(messages.shape)  # Output: (5572, 1000)

# print(messages[0,:]) # shape is (1, 1000) --> frequency count of top words though it only prints the frequency of top_features in the first sentence, in this case there are 12 top features are present.


X = torch.tensor(messages.todense(), dtype=torch.float32)
Y = torch.tensor(df["spam"].values, dtype=torch.float32).reshape(-1,1) # without reshaping the shape of Y will be (5572,) but we need it to be (5572,1) for making it compitible with the input feature shape.

model = nn.Linear(1000,1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5000):
    optimizer.zero_grad()
    Y_pred = model(X)
    loss = loss_fn(Y_pred, Y)
    loss.backward()
    optimizer.step()

    if(epoch % 100 == 0):
        print(f"Epoch {epoch}, Loss: {loss.item()}")

model.eval()

with torch.no_grad():
    Y_pred = model(X)
    Y_pred = torch.sigmoid(Y_pred)  # Apply sigmoid to get probabilities, otherwise the output will be in the range of (-inf, inf) and we need it to be in the range of (0,1) for binary classification.
    
    accuracy = ((Y_pred > 0.5) == Y).float().mean()  # Calculate accuracy by comparing the predicted labels with the true labels.
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")
    sensitivity = ((Y_pred > 0.5) & (Y == 1)).float().sum() / (Y == 1).float().sum()  # Calculate sensitivity (true positive rate)
    specificity = ((Y_pred <= 0.5) & (Y == 0)).float().sum() / (Y == 0).float().sum()  # Calculate specificity (true negative rate)
    print(f"Sensitivity: {sensitivity.item() * 100:.2f}%")  
    print(f"Specificity: {specificity.item() * 100:.2f}%")  
    precision = ((Y_pred > 0.5) & (Y == 1)).float().sum() / (Y_pred > 0.5).float().sum()  # Calculate precision (positive predictive value)
    print(f"Precision: {precision.item() * 100:.2f}%")









