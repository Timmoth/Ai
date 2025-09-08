const tf = require("@tensorflow/tfjs-node");
const fs = require("fs").promises; // Use the promises-based version of fs
const path = require("path");

// --- Configuration ---
//const CLASSES = ["fan", "moon", "microphone", "calculator", "van", "spider", "parrot", "piano", "scorpion", "broccoli", "sea turtle"]; 
//const CLASSES = ["fan", "moon", "microphone", "calculator", "van", "spider", "parrot", "piano", "scorpion", "broccoli", "sea turtle", "envelope", "mouth", "birthday cake", "beard", "rake", "motorbike", "teddy-bear", "cell phone", "airplane", "hedgehog", "snake", "mermaid", "nail", "shark", "key", "horse", "cloud", "ambulance", "saw", "cactus", "dumbbell", "triangle", "mountain", "bus", "donut", "light bulb", "baseball bat", "book", "clock", "submarine", "radio", "camouflage", "picture frame", "stethoscope", "bulldozer", "lollipop", "string bean", "streetlight", "lantern", "teapot", "postcard", "fence", "animal migration", "campfire", "bucket", "pickup truck", "penguin", "hurricane", "ant", "circle", "butterfly", "popsicle", "ocean", "speedboat", "lion", "lobster", "diving board", "flower", "truck", "fire hydrant", "wine glass", "sink", "rainbow", "eyeglasses", "pineapple", "television", "roller coaster", "screwdriver", "sandwich", "bread", "paint can", "sun", "flashlight", "hot dog", "passport", "cookie", "eye", "apple", "candle", "raccoon", "zigzag", "ladder", "barn", "sleeping bag", "microwave", "dog", "river", "stop sign", "trumpet", "telephone", "wheel", "helicopter", "sword", "skyscraper", "anvil", "pillow", "power outlet", "hockey puck", "mailbox", "foot", "carrot", "bear", "shovel", "spoon", "cup", "fork", "church", "bee", "pig", "finger", "bird", "megaphone", "snowflake", "owl", "eraser", "alarm clock", "clarinet", "shoe", "lipstick", "ice cream", "hand", "knee", "skateboard", "pool", "tractor", "whale", "nose", "hamburger", "baseball", "pear", "waterslide", "laptop", "mouse", "tiger", "car", "bottlecap", "stove", "snorkel", "square", "strawberry", "rollerskates", "boomerang", "rain", "harp", "bat", "The Great Wall of China", "The Mona Lisa", "door", "see saw", "paintbrush", "flip flops", "kangaroo", "school bus", "asparagus", "computer", "The Eiffel Tower", "pencil", "hat", "drums", "bed", "cooler", "face", "t-shirt", "hexagon", "tennis racquet", "arm", "house", "washing machine", "binoculars", "smiley face", "tree", "squiggle", "mosquito", "pond", "trombone", "keyboard", "bush", "marker", "bandage", "umbrella", "lightning", "wine bottle", "bathtub", "hot air balloon", "floor lamp", "toe", "necklace", "saxophone", "blueberry", "oven", "tornado", "frog", "toaster", "guitar", "line", "dragon", "feather", "swan", "flying saucer", "police car", "headphones", "elephant", "steak", "sheep", "peanut", "snail", "castle", "spreadsheet", "garden hose", "zebra", "hot tub", "cruise ship", "shorts", "grass", "mushroom", "chandelier", "swing set", "golf club", "bracelet", "crown", "octagon", "traffic light", "calendar", "pliers", "bridge", "octopus", "cow", "toothpaste", "garden", "tooth", "beach", "fish", "cannon", "yoga", "panda", "pants", "snowman", "duck", "bicycle", "hourglass", "sock", "hospital", "skull", "compass", "flamingo", "rabbit", "wristwatch", "ear", "diamond", "sweater", "giraffe", "scissors", "crab", "brain", "crayon", "monkey", "chair", "frying pan", "belt", "broom", "elbow", "soccer ball", "jail", "mug", "helmet", "canoe", "toilet", "moustache", "basket", "rhinoceros", "dishwasher", "dolphin", "basketball", "camel", "cello", "grapes", "tent", "paper clip", "hammer", "toothbrush", "pizza", "coffee cup", "drill", "vase", "violin", "squirrel", "sailboat", "underwear", "bench", "leaf", "bowtie", "stitches", "hockey stick", "parachute", "crocodile", "suitcase", "jacket", "windmill", "cat", "lighthouse", "angel", "goatee", "cake", "stairs", "couch", "purse", "firetruck", "banana", "star", "palm tree", "fireplace", "matches", "dresser", "train", "camera", "stereo", "watermelon", "remote control", "peas", "blackberry", "onion", "table", "ceiling fan", "map", "potato", "leg", "house plant", "axe", "backpack"];
const CLASSES = ["fan", "moon", "microphone", "calculator", "van", "spider", "parrot", "piano", "scorpion", "broccoli", "sea turtle", "envelope", "mouth", "birthday cake", "beard", "rake", "motorbike", "teddy-bear", "cell phone", "airplane", "hedgehog", "snake", "mermaid", "nail", "shark", "key", "horse", "cloud", "ambulance", "saw", "cactus", "dumbbell", "triangle", "mountain", "bus", "donut", "light bulb", "baseball bat", "book", "clock"];
const IMAGE_SIZE = 128;
const BATCH_SIZE = 500;
const EPOCHS = 50;
const DATA_DIR = "./doodle"; // The root directory for your class folders

/**
 * Gets a shuffled list of all file paths and a class name to index map.
 */
async function getFilePaths() {
    const allFiles = [];
    const classToIndex = {};
    for (let i = 0; i < CLASSES.length; i++) {
        const className = CLASSES[i];
        classToIndex[className] = i;
        const classDir = path.join(DATA_DIR, className);
        const filesInDir = await fs.readdir(classDir);
        for (const file of filesInDir) {
            allFiles.push({ filePath: path.join(classDir, file), className });
        }
    }
    tf.util.shuffle(allFiles); // Shuffle the entire dataset before splitting
    return { allFiles, classToIndex };
}

/**
 * An asynchronous generator function that loads and processes images one by one.
 */
async function* dataGenerator(files, classToIndex) {
    for (const fileInfo of files) {
        const imgBuffer = await fs.readFile(fileInfo.filePath);
        const labelIndex = classToIndex[fileInfo.className];

        // Use tf.tidy to auto-dispose intermediate tensors
        const { xs, ys } = tf.tidy(() => {
            const xs = tf.node.decodeImage(imgBuffer, 1) // 1 channel for grayscale
                .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
                .toFloat()
                .div(255.0); // Normalize to [0, 1]
            const ys = tf.oneHot(labelIndex, CLASSES.length);
            return { xs, ys };
        });
        yield { xs, ys };
    }
}

/**
 * Creates a tf.data.Dataset object from the file list.
 */
function createDataset(files, classToIndex) {
    return tf.data.generator(() => dataGenerator(files, classToIndex))
        .shuffle(1024) // Buffer size for shuffling
        .batch(BATCH_SIZE)
        .prefetch(1); // Prefetch batches
}

async function run() {

    console.log("Loading file paths...");
    const { allFiles, classToIndex } = await getFilePaths();

    // 1. Create a Validation Set (critical for checking accuracy)
    const splitIndex = Math.floor(allFiles.length * 0.8);
    const trainFiles = allFiles.slice(0, splitIndex);
    const valFiles = allFiles.slice(splitIndex);

    console.log(`Training on ${trainFiles.length} images, validating on ${valFiles.length}.`);

    console.log("Creating datasets...");
    const trainDataset = createDataset(trainFiles, classToIndex);
    const valDataset = createDataset(valFiles, classToIndex);

    console.log("Building model...");
    const model = tf.sequential();

    // 2. Add Data Augmentation Layers (critical for preventing overfitting)
    model.add(tf.layers.inputLayer({ inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1] }));

    // --- CNN Architecture ---
    model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: "same", activation: "relu" }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: "same", activation: "relu" }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, padding: "same", activation: "relu" }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: CLASSES.length, activation: "softmax" }));

    model.summary();

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    console.log("Training...");
    await model.fitDataset(trainDataset, {
        epochs: EPOCHS,
        validationData: valDataset, // Provide the validation dataset here
        callbacks: tf.callbacks.earlyStopping({ monitor: 'val_acc', patience: 3 })
    });

    console.log("Saving model...");
    await model.save("file://./doodle-model");
    console.log("Model saved to ./doodle-model");
}

run().catch(err => console.error(err));