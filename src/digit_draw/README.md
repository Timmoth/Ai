# Digit Draw: A TensorFlow.js Handwritten Digit Recognition Project

## [Live demo](https://ai.timmoth.com/demos/digit_draw/)

**Description:**

Digit Draw is a browser based interactive demo that allows you to draw digits (0-9) on a canvas, and it will predict what digit you're drawing using a TensorFlow convolutional neural network trained on the MNIST dataset.

**Technologies Used:**

-   **JavaScript:** 
-   **TensorFlow.js:** 
-   **HTML5 Canvas:**
-   **Node.js:**
-   **MNIST Dataset:**

**How it Works:**

1.  **Data Loading:** The Node.js backend reads & parses the MNIST training data files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`)
2.  **Data Preprocessing:** The image data is normalized by dividing pixel values by 255.0 to scale them between 0 and 1.
3.  **One-Hot Encoding:** The labels (digits 0-9) are converted into one-hot vectors, which represent each digit as a binary vector with a 1 at the corresponding index.
4.  **Data Augmentation:** The training dataset is augmented by applying random shifts and noise to the images. This helps improve the model's ability to generalize to different handwriting styles.
5.  **Model:** A convolutional neural network is constructed using TensorFlow.js layers:
    *   Reshape layer to prepare the input for convolutional layers.
    *   Two sets of Conv2D and MaxPooling2D layers to extract features from the images.
    *   Flatten layer to convert the 2D feature maps into a 1D vector.
    *   Dense layers with ReLU activation for classification.
    *   Output layer with softmax activation to produce probabilities for each digit (0-9).
6.  **Training:** The model is trained using the augmented training data and the Adam optimizer with a learning rate of 0.001. Categorical cross-entropy is used as the loss function. Early stopping prevents overfitting.
7.  **Evaluation:** The trained model is evaluated on the test dataset.
8.  **Inference:** The frontend fetches the model and predicts the users drawing. The canvas drawing is converted into a tensor, normalized, and passed to the model for prediction.

**Installation & Usage:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Timmoth/ai.git
    ```

2.  **Install Dependencies:**
    ```bash
    npm install
    ```

3.  **Download MNIST Data:** Download the mnist dataset [(kaggle)](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and place them in the same directory as `index.js`:
    *   `train-images.idx3-ubyte`
    *   `train-labels.idx1-ubyte`
    *   `t10k-images.idx3-ubyte`
    *   `t10k-labels.idx1-ubyte`

4.  **Train the Model:**
    ```bash
    node index.js
    ```