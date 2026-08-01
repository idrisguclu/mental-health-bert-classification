tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
- A tokenizer converts raw text into token IDs.
- BERT cannot process raw text directly.
- bert-base-uncased ignores uppercase/lowercase.
- AutoTokenizer automatically loads the correct tokenizer.

Padding makes all sequences in a batch the same length. The attention mask tells BERT which tokens are real and which ones are padding, so the model does not attend to the padded positions.


Model hiçbir zaman *accuracy*'yi optimize etmez. Model sadece **LOSS** optimize eder.

First, the ***tokenizer*** converts the raw text into tokens and assigns an **input ID** to each token. It also creates an **attention mask**, which tells the model which tokens are real and which ones are padding. The tokenized input is then passed to the BERT model. BERT produces **logits**, which are raw output scores for each class. We apply the **Softmax** function to convert these logits into **probabilities**. Finally, we use **argmax** to select the class with the highest probability as the model's prediction.

Batch size = number of sentences processed together.
**Interview Question :** What does the map() function do in the Hugging Face Dataset library?

**Bunu şöyle cevaplayabilirsin:** The map() function applies a given function to every example in the dataset. It is commonly used for preprocessing tasks such as tokenization.

**One epoch** means that the model has seen every training example exactly once.

What does loss.backward() do?
**loss.backward()** computes the gradients of the loss with respect to the model's parameters. These gradients are used to update the model during training.
Why do we call optimizer.step()?
**optimizer.step()** updates the model's parameters using the gradients computed during backpropagation.
Why do we call optimizer.zero_grad()?
PyTorch accumulates gradients by default. We call **optimizer.zero_grad()** to clear the old gradients before processing the next batch.

Can you explain the training loop of a BERT model?
First, the model makes predictions during the forward pass. Then it calculates the loss by comparing the predictions with the true labels. Next, **loss.backward()** computes the gradients of the loss with respect to the model parameters. After that, **optimizer.step()** updates the model parameters using these gradients. Finally, **optimizer.zero_grad()** clears the gradients before processing the next batch. This process is repeated for every batch in every epoch.

How can you prevent overfitting when fine-tuning BERT?

1-**Early stopping** (stop training when validation performance stops improving)
2-**Dropout** (randomly deactivate some neurons during training)
3-**Weight decay** (L2 regularization) (penalize overly large weights)
4-**Use more training data** (if available)
5-**Data augmentation** (create additional training examples)

generalizes to unseen data

**A gradient** tells the model how much and in which direction each parameter should be updated to reduce the loss and improve the predictions.

**outputs = model(**batch)**	The model makes predictions.
**loss = outputs.loss**	The model measures how wrong the predictions are.
**loss.backward()**	The model computes the gradients for every parameter.
**optimizer.step()**	The model updates the parameters using those gradients.
**optimizer.zero_grad()**	The model clears the gradients before the next batch.


Can you explain the entire BERT fine-tuning process from the input text to the parameter update?
First, **the tokenizer** converts the raw text into tokens and token IDs that BERT can process. **The DataLoader** groups these tokenized inputs into **batches**. During each batch, the model performs a forward pass to make predictions and computes **the loss**. Then **loss.backward()** calculates the **gradients** of the loss with respect to the model parameters. After that, **optimizer.step()** uses these gradients to update the model parameters. This process is repeated for every batch in each epoch until the model learns meaningful patterns. During training, we also monitor the **validation loss**. If the validation loss starts to increase while the training loss continues to decrease, it indicates **overfitting.** Our goal is to build a model that generalizes well to unseen data.

How does BERT transform token IDs into meaningful embeddings before making predictions?
A neural network cannot process raw text directly because it only works with numbers. In BERT, the tokenizer converts the raw text into tokens and then into token IDs. These token IDs are passed to the model, where they are converted into embeddings. The embeddings capture the semantic meaning of the tokens, allowing BERT to understand the context and make predictions.

Why do we call the embedding matrix "trainable"?
**The embedding matrix** is trainable because its dense vectors are updated during training. **loss.backward()** computes gradients for the embedding vectors, and **optimizer.step()** updates them to reduce the loss. As a result, the embeddings gradually capture more useful semantic and contextual information.

**Masked Language Modeling (MLM)** is a pre-training task in which some words in a sentence are replaced with a [MASK] token. BERT learns to predict the missing words by using the context from both the left and right sides of the sentence. This helps BERT learn the meaning and relationships between words.

What is Next Sentence Prediction (NSP)?
**Next Sentence Prediction** is a pre-training task in which BERT learns whether one sentence logically follows another. It helps BERT understand relationships and context between sentences

We **fine-tune** a pre-trained BERT model because it has already been trained on a large corpus of text and has learned grammar, context, and semantic relationships. Fine-tuning requires much less data and computational resources than training a model from scratch, making it more efficient for specific tasks such as mental health text classification.

First, we load the **tokenizer** from the Transformers library. The tokenizer converts raw text into tokens and assigns a unique ID to each token. These token IDs are converted into **embedding vectors** and passed through the BERT model. The model performs a forward pass and produces **logits** for each output class. During training, the loss function compares the logits with the true labels and calculates the prediction error. Then **loss.backward()** computes the **gradients** of the loss with respect to the model parameters. Next, **optimizer.step()** updates the model parameters using these gradients. This process is repeated for multiple **epochs** until the model learns to make accurate predictions.

Training loss decreases while validation loss increases.
A stronger answer: The model is **overfitting** because it performs well on the training data but poorly on unseen validation data. I would use techniques such as **early stopping**, **dropout**, **data augmentation**, **regularisation**, or **collecting more training data** to reduce overfitting.

We use **model.eval()** to switch the model into evaluation mode. This disables dropout and ensures that the model makes consistent predictions during validation or testing. It allows us to evaluate the model's performance accurately.

In training mode, the model learns from the training data by updating its parameters through backpropagation. Layers such as dropout are active to help prevent overfitting. In evaluation mode, the model's parameters are no longer updated. Dropout is disabled, and the model makes consistent predictions so that we can accurately evaluate its performance.