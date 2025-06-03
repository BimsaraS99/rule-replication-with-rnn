# 🧠 Modeling and Replicating Rule-Based Systems with Recurrent Neural Networks (RNN)

This project is a beginner-friendly way to understand **how Recurrent Neural Networks (RNNs)** work, by teaching a neural network to replicate a simple rule-based system. No need to start with complex datasets—here, we generate our own using simple logic!

---

## 📌 What is an RNN?

Recurrent Neural Networks (RNNs) are a type of neural network designed to handle **sequential data**. They are especially good at understanding **temporal patterns**, such as:

* Predicting the next word in a sentence
* Detecting anomalies in time-series data
* Speech recognition
* Music generation
* Stock price prediction

Unlike traditional feed-forward networks, RNNs have a “memory” of previous steps in a sequence, which helps them make better predictions when context matters.

---

## 🎯 What Is This Project About?

We built a simple RNN that **learns to replicate a rule-based system**. The rule is simple:

> If a sequence of digits contains a `3` followed by a `7` (in that order, not necessarily next to each other), then label it as `1`. Otherwise, label it as `0`.

The goal is to train the RNN to understand this pattern **without being explicitly told the rule**—just by looking at many examples.

---

## 🔧 How We Generated the Dataset

Instead of using a real-world dataset, we created a synthetic one using this rule. Here's the function that generates labeled sequences:

```python
def generate_order_dependent_sequence(min_len=5, max_len=10):
    seq_len = random.randint(min_len, max_len)
    sequence = [random.randint(0, 9) for _ in range(seq_len)]
    
    found_3 = False
    label = 0
    for num in sequence:
        if num == 3:
            found_3 = True
        elif num == 7 and found_3:
            label = 1
            break
    
    return sequence, label
```

We generated **10,000 samples** and saved them to a CSV file, padding shorter sequences with zeros to ensure fixed length (10 steps).

---

## 🧠 Model Architecture

Here’s a simplified version of the RNN model:

```python
class RNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # take the output at the final time step
        out = self.relu(self.fc1(out))
        return self.fc2(out)  # raw output, sigmoid applied later
```

We use a 2-layer RNN, followed by fully connected layers to produce a single output. The final prediction is done using `sigmoid()`.

---

## 🏋️ Training the Model

To train the model, we:

* Load the CSV dataset
* Normalize the inputs (0s are replaced with -1 to distinguish padding)
* Use `BCEWithLogitsLoss` (Binary Cross-Entropy with Sigmoid)
* Train for 200 epochs with the Adam optimizer

Training loop sample:

```python
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, num_epochs):
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

✅ After training, the model achieves very low loss and starts **predicting correctly even on unseen sequences**.

---

## 🧪 Testing the Model

Let’s try it manually with this test case:

```python
manual_sequence = [7, 2, 7, 8, 7, 3]
```

After preparing and padding the sequence, we feed it to the model and apply sigmoid to get the probability:

```python
with torch.no_grad():
    logits = model(input_tensor)
    prob = torch.sigmoid(logits)
    print("Predicted label:", 1 if prob.item() > 0.5 else 0)
```

We also calculate the label using the **rule-based method**:

```python
def get_label_from_sequence(sequence):
    found_3 = False
    for num in sequence:
        if num == 3:
            found_3 = True
        elif num == 7 and found_3:
            return 1
    return 0
```

🎉 The **RNN prediction matches the rule-based output exactly**, confirming that the model has successfully **learned the pattern**!

---

## 💡 What You Learned

This small project demonstrates:

* How to build an RNN using PyTorch
* How to simulate rule-based labeled data
* How an RNN can replicate logical rules *without being explicitly told*
* How to structure a training workflow from data to evaluation
