# Handwritten Digits Prediction

Predict handwritten digits in the browser using a basic tensorflow CNN trained on the MNIST handwritten digits dataset.

---

- The script loads MNIST images and labels using readIDX.
- It normalizes the images (pixel values 0–1).
- It applies some augmentation to the training data e.g small random shifts and noise which helps the network generalize better.
- Builds a lightweight CNN:
  2 Conv blocks with ReLU activations + max pooling
  Dense layer (64 units) with dropout
  Output layer (10 units, softmax)
- Compiles the model using Adam optimizer and categorical crossentropy loss.
- Trains for up to 15 epochs with early stopping on validation loss.
