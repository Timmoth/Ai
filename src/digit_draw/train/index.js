const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

function readIDX(filename) {
    const buffer = fs.readFileSync(filename);
    const magic = buffer.readUInt32BE(0);

    if (magic === 2049) { // labels
        const numItems = buffer.readUInt32BE(4);
        const labels = new Uint8Array(numItems);
        for (let i = 0; i < numItems; i++) {
            labels[i] = buffer[8 + i];
        }
        return labels;
    } else if (magic === 2051) { // images
        const numImages = buffer.readUInt32BE(4);
        const numRows = buffer.readUInt32BE(8);
        const numCols = buffer.readUInt32BE(12);
        // Flatten the images into a single Float32Array
        const images = new Float32Array(numImages * numRows * numCols);
        let idx = 16;
        for (let i = 0; i < images.length; i++) {
            // Normalization happens ONCE, right here.
            images[i] = buffer[idx++] / 255.0;
        }
        return { images, numImages };
    } else {
        throw new Error("Unknown IDX magic number " + magic);
    }
}

/**
 * Convert labels (0-9) to one-hot vectors.
 */
function oneHot(labels, numClasses) {
    return tf.tidy(() => tf.oneHot(tf.tensor1d(labels, "int32"), numClasses));
}

function augmentImages(images, numImages) {
    return tf.tidy(() => {
        let imgTensor = tf.tensor4d(images, [numImages, 28, 28, 1]);

        const augmented = [];

        for (let i = 0; i < numImages; i++) {
            let img = imgTensor.slice([i, 0, 0, 0], [1, 28, 28, 1]);

            // Random shift ±2 pixels
            const dx = Math.floor(Math.random() * 5 - 2);
            const dy = Math.floor(Math.random() * 5 - 2);

            // pad and slice to simulate shift
            const paddings = [
                [Math.max(dy, 0), Math.max(-dy, 0)],
                [Math.max(dx, 0), Math.max(-dx, 0)]
            ];
            let padded = tf.pad(img.squeeze(), paddings);
            img = padded.slice(
                [Math.max(-dy, 0), Math.max(-dx, 0)],
                [28, 28]
            ).reshape([1, 28, 28, 1]);

            // Add small random noise
            const noise = tf.randomUniform([1, 28, 28, 1], -0.05, 0.05);
            img = img.add(noise).clipByValue(0, 1);

            augmented.push(img.squeeze());
        }

        return tf.stack(augmented).reshape([numImages, 28 * 28]);
    });
}


async function run() {
    console.log("Loading data...");
    const trainImgs = readIDX("./train-images.idx3-ubyte");
    const trainLbls = readIDX("./train-labels.idx1-ubyte");
    const testImgs = readIDX("./t10k-images.idx3-ubyte");
    const testLbls = readIDX("./t10k-labels.idx1-ubyte");

    // --- convert to tensors
    let xTrainOrig = tf.tensor2d(trainImgs.images, [trainImgs.numImages, 28 * 28]);
    const yTrainOrig = oneHot(Array.from(trainLbls), 10);
    const xTest = tf.tensor2d(testImgs.images, [testImgs.numImages, 28 * 28]);
    const yTest = oneHot(Array.from(testLbls), 10);

    // --- apply simple augmentation
    console.log("Applying data augmentation...");
    xTrainOrig = augmentImages(trainImgs.images, trainImgs.numImages);

    console.log("Building model...");
    const model = tf.sequential();
    model.add(tf.layers.reshape({ targetShape: [28, 28, 1], inputShape: [784] }));

    // Conv Block 1
    model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: 'same', activation: 'relu' }));
    model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: 'same', activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    // Conv Block 2
    model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu' }));
    model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: 'same', activation: 'relu' }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.flatten());

    // Dense
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));

    // Output
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));


    model.summary();

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    console.log("Training...");
    await model.fit(xTrainOrig, yTrainOrig, {
        epochs: 15,
        batchSize: 128,
        validationSplit: 0.1,
        callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 })
    });

    xTrainOrig.dispose();
    yTrainOrig.dispose();

    console.log("Evaluating...");
    const evalResult = model.evaluate(xTest, yTest);
    const testAcc = (await evalResult[1].data())[0];
    console.log(`\nTest accuracy: ${(testAcc * 100).toFixed(2)} %`);

    await model.save("file://../demo");
    console.log("Improved model saved!");
}

run().catch(err => console.error(err));