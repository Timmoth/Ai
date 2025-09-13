import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflowjs as tfjs
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
import os
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Config ---
CLASSES = ["fan", "moon", "microphone", "calculator", "van", "spider", "parrot", "piano", "scorpion", "broccoli", "sea turtle", "envelope", "mouth", "birthday cake", "beard", "rake", "motorbike", "teddy-bear", "cell phone", "airplane", "hedgehog", "snake", "mermaid", "nail", "shark", "key", "horse", "cloud", "ambulance", "saw", "cactus", "dumbbell", "triangle", "mountain", "bus", "donut", "light bulb", "baseball bat", "book", "clock", "submarine", "radio", "camouflage", "picture frame", "stethoscope", "bulldozer", "lollipop", "string bean", "streetlight", "lantern", "teapot", "postcard", "fence", "animal migration", "campfire", "bucket", "pickup truck", "penguin", "hurricane", "ant", "circle", "butterfly", "popsicle", "ocean", "speedboat", "lion", "lobster", "diving board", "flower", "truck", "fire hydrant", "wine glass", "sink", "rainbow", "eyeglasses", "pineapple", "television", "roller coaster", "screwdriver", "sandwich", "bread", "paint can", "sun", "flashlight", "hot dog", "passport", "cookie", "eye", "apple", "candle", "raccoon", "zigzag", "ladder", "barn", "sleeping bag", "microwave", "dog", "river", "stop sign", "trumpet", "telephone", "wheel", "helicopter", "sword", "skyscraper", "anvil", "pillow", "power outlet", "hockey puck", "mailbox", "foot", "carrot", "bear", "shovel", "spoon", "cup", "fork", "church", "bee", "pig", "finger", "bird", "megaphone", "snowflake", "owl", "eraser", "alarm clock", "clarinet", "shoe", "lipstick", "ice cream", "hand", "knee", "skateboard", "pool", "tractor", "whale", "nose", "hamburger", "baseball", "pear", "waterslide", "laptop", "mouse", "tiger", "car", "bottlecap", "stove", "snorkel", "square", "strawberry", "rollerskates", "boomerang", "rain", "harp", "bat", "The Great Wall of China", "The Mona Lisa", "door", "see saw", "paintbrush", "flip flops", "kangaroo", "school bus", "asparagus", "computer", "The Eiffel Tower", "pencil", "hat", "drums", "bed", "cooler", "face", "t-shirt", "hexagon", "tennis racquet", "arm", "house", "washing machine", "binoculars", "smiley face", "tree", "squiggle", "mosquito", "pond", "trombone", "keyboard", "bush", "marker", "bandage", "umbrella", "lightning", "wine bottle", "bathtub", "hot air balloon", "floor lamp", "toe", "necklace", "saxophone", "blueberry", "oven", "tornado", "frog", "toaster", "guitar", "line", "dragon", "feather", "swan", "flying saucer", "police car", "headphones", "elephant", "steak", "sheep", "peanut", "snail", "castle", "spreadsheet", "garden hose", "zebra", "hot tub", "cruise ship", "shorts", "grass", "mushroom", "chandelier", "swing set", "golf club", "bracelet", "crown", "octagon", "traffic light", "calendar", "pliers", "bridge", "octopus", "cow", "toothpaste", "garden", "tooth", "beach", "fish", "cannon", "yoga", "panda", "pants", "snowman", "duck", "bicycle", "hourglass", "sock", "hospital", "skull", "compass", "flamingo", "rabbit", "wristwatch", "ear", "diamond", "sweater", "giraffe", "scissors", "crab", "brain", "crayon", "monkey", "chair", "frying pan", "belt", "broom", "elbow", "soccer ball", "jail", "mug", "helmet", "canoe", "toilet", "moustache", "basket", "rhinoceros", "dishwasher", "dolphin", "basketball", "camel", "cello", "grapes", "tent", "paper clip", "hammer", "toothbrush", "pizza", "coffee cup", "drill", "vase", "violin", "squirrel", "sailboat", "underwear", "bench", "leaf", "bowtie", "stitches", "hockey stick", "parachute", "crocodile", "suitcase", "jacket", "windmill", "cat", "lighthouse", "angel", "goatee", "cake", "stairs", "couch", "purse", "firetruck", "banana", "star", "palm tree", "fireplace", "matches", "dresser", "train", "camera", "stereo", "watermelon", "remote control", "peas", "blackberry", "onion", "table", "ceiling fan", "map", "potato", "leg", "house plant", "axe", "backpack"]

IMAGE_SIZE = 64*64
X_PATH = "X.dat"
Y_PATH = "y.dat"
X_size_bytes = os.path.getsize(X_PATH)
total_samples = X_size_bytes // (IMAGE_SIZE)
X = np.memmap(X_PATH, dtype=np.uint8, mode="r", shape=(total_samples, IMAGE_SIZE))
y = np.memmap(Y_PATH, dtype=np.int32, mode="r", shape=(total_samples,))

DATA_DIR = Path("./archive-2")
IMAGE_SIZE = 64
BATCH_SIZE = 256
NUM_CLASSES = len(CLASSES)
EPOCHS=50

total_samples = X.shape[0]
NUM_VALID = int(total_samples * 0.01)
NUM_TRAIN = total_samples - NUM_VALID

# --- Memory-mapped arrays ---
X_mem = np.memmap(X_PATH, dtype=np.uint8, mode="r", shape=(total_samples, IMAGE_SIZE * IMAGE_SIZE))
y_mem = np.memmap(Y_PATH, dtype=np.int32, mode="r", shape=(total_samples,))

# --- Generator yielding batches ---
def batch_generator(start_idx, end_idx):
    for i in range(start_idx, end_idx, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, end_idx)
        X_batch = X_mem[i:batch_end].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
        y_batch = tf.keras.utils.to_categorical(y_mem[i:batch_end], num_classes=NUM_CLASSES)
        yield X_batch, y_batch

# --- Wrap generator in tf.data.Dataset ---
def make_dataset(start_idx, end_idx, augment=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: batch_generator(start_idx, end_idx),
        output_signature=(
            tf.TensorSpec(shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    )

    if augment:
        # GPU-friendly augmentations
        data_augment = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomZoom(height_factor=(-0.2, 0.0), width_factor=(-0.2, 0.0))
        ])
        dataset = dataset.map(lambda x, y: (data_augment(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = make_dataset(0, NUM_TRAIN, augment=True)
val_ds = make_dataset(NUM_TRAIN, NUM_TRAIN + NUM_VALID, augment=False)

# --- Load all data ---
import os

class TFJSCheckpoint(Callback):
    def __init__(self, save_dir="tfjs_checkpoints"):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # Save a TFJS model at the end of every epoch
        tfjs_path = os.path.join(self.save_dir, f"epoch_{epoch+1:02d}")
        tfjs.converters.save_keras_model(self.model, tfjs_path)
        print(f"TFJS model saved to {tfjs_path}")

# --- Build model ---
def build_model(input_shape=(64,64,1), num_classes=340):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # Block 1
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Block 2
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Block 3
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        # Global pooling
        tf.keras.layers.GlobalAveragePooling2D(),

        # Fully connected
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
    )
    return model


# --- Main ---
def main():
   
    model = build_model()
    model.summary()

    checkpoint_cb = ModelCheckpoint(
        "quickdraw_model_epoch{epoch:02d}_val{val_loss:.2f}.h5",
        monitor="val_loss",       # metric to monitor
        save_best_only=False,     # save every epoch
        save_weights_only=False,  # save full model
        verbose=1
    )

    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=5, # Number of epochs with no improvement to wait
        restore_best_weights=True # Restore model weights from the best epoch
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2, # Reduce LR by a factor of 5
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    tfjs_cb = TFJSCheckpoint(save_dir="tfjs_checkpoints")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, tfjs_cb]
    )

    model.save("quickdraw_model_final.h5", save_format="h5")
    tfjs.converters.save_keras_model(model, "tfjs_model")

    print("Model saved!")


if __name__ == "__main__":
    main()


