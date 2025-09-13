import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
from collections import Counter

CLASSES = ["fan", "moon", "microphone", "calculator", "van", "spider", "parrot", "piano", "scorpion", "broccoli", "sea turtle", "envelope", "mouth", "birthday cake", "beard", "rake", "motorbike", "teddy-bear", "cell phone", "airplane", "hedgehog", "snake", "mermaid", "nail", "shark", "key", "horse", "cloud", "ambulance", "saw", "cactus", "dumbbell", "triangle", "mountain", "bus", "donut", "light bulb", "baseball bat", "book", "clock", "submarine", "radio", "camouflage", "picture frame", "stethoscope", "bulldozer", "lollipop", "string bean", "streetlight", "lantern", "teapot", "postcard", "fence", "animal migration", "campfire", "bucket", "pickup truck", "penguin", "hurricane", "ant", "circle", "butterfly", "popsicle", "ocean", "speedboat", "lion", "lobster", "diving board", "flower", "truck", "fire hydrant", "wine glass", "sink", "rainbow", "eyeglasses", "pineapple", "television", "roller coaster", "screwdriver", "sandwich", "bread", "paint can", "sun", "flashlight", "hot dog", "passport", "cookie", "eye", "apple", "candle", "raccoon", "zigzag", "ladder", "barn", "sleeping bag", "microwave", "dog", "river", "stop sign", "trumpet", "telephone", "wheel", "helicopter", "sword", "skyscraper", "anvil", "pillow", "power outlet", "hockey puck", "mailbox", "foot", "carrot", "bear", "shovel", "spoon", "cup", "fork", "church", "bee", "pig", "finger", "bird", "megaphone", "snowflake", "owl", "eraser", "alarm clock", "clarinet", "shoe", "lipstick", "ice cream", "hand", "knee", "skateboard", "pool", "tractor", "whale", "nose", "hamburger", "baseball", "pear", "waterslide", "laptop", "mouse", "tiger", "car", "bottlecap", "stove", "snorkel", "square", "strawberry", "rollerskates", "boomerang", "rain", "harp", "bat", "The Great Wall of China", "The Mona Lisa", "door", "see saw", "paintbrush", "flip flops", "kangaroo", "school bus", "asparagus", "computer", "The Eiffel Tower", "pencil", "hat", "drums", "bed", "cooler", "face", "t-shirt", "hexagon", "tennis racquet", "arm", "house", "washing machine", "binoculars", "smiley face", "tree", "squiggle", "mosquito", "pond", "trombone", "keyboard", "bush", "marker", "bandage", "umbrella", "lightning", "wine bottle", "bathtub", "hot air balloon", "floor lamp", "toe", "necklace", "saxophone", "blueberry", "oven", "tornado", "frog", "toaster", "guitar", "line", "dragon", "feather", "swan", "flying saucer", "police car", "headphones", "elephant", "steak", "sheep", "peanut", "snail", "castle", "spreadsheet", "garden hose", "zebra", "hot tub", "cruise ship", "shorts", "grass", "mushroom", "chandelier", "swing set", "golf club", "bracelet", "crown", "octagon", "traffic light", "calendar", "pliers", "bridge", "octopus", "cow", "toothpaste", "garden", "tooth", "beach", "fish", "cannon", "yoga", "panda", "pants", "snowman", "duck", "bicycle", "hourglass", "sock", "hospital", "skull", "compass", "flamingo", "rabbit", "wristwatch", "ear", "diamond", "sweater", "giraffe", "scissors", "crab", "brain", "crayon", "monkey", "chair", "frying pan", "belt", "broom", "elbow", "soccer ball", "jail", "mug", "helmet", "canoe", "toilet", "moustache", "basket", "rhinoceros", "dishwasher", "dolphin", "basketball", "camel", "cello", "grapes", "tent", "paper clip", "hammer", "toothbrush", "pizza", "coffee cup", "drill", "vase", "violin", "squirrel", "sailboat", "underwear", "bench", "leaf", "bowtie", "stitches", "hockey stick", "parachute", "crocodile", "suitcase", "jacket", "windmill", "cat", "lighthouse", "angel", "goatee", "cake", "stairs", "couch", "purse", "firetruck", "banana", "star", "palm tree", "fireplace", "matches", "dresser", "train", "camera", "stereo", "watermelon", "remote control", "peas", "blackberry", "onion", "table", "ceiling fan", "map", "potato", "leg", "house plant", "axe", "backpack"]

IMAGE_SIZE = 64*64
BATCH_SIZE = 1_000_000
REORDER_BATCH_SIZE = 100_000
NUM_PASSES = 4
NUM_CLASSES = len(CLASSES)

X_PATH = "X.dat"
Y_PATH = "y.dat"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

start_time = time.time()
X_size_bytes = os.path.getsize(X_PATH)
total_samples = X_size_bytes // (IMAGE_SIZE * 4)
log(f"Total samples determined from file size: {total_samples} (took {time.time()-start_time:.2f}s)")

X = np.memmap(X_PATH, dtype=np.uint8, mode="r", shape=(total_samples, IMAGE_SIZE))
y = np.memmap(Y_PATH, dtype=np.int32, mode="r", shape=(total_samples,))
log(f"Loaded memmaps (total_samples={total_samples})")

seen = set()
for i, label in enumerate(y):
    seen.add(label)
    if len(seen) == NUM_CLASSES:
        print(f"All {NUM_CLASSES} classes seen after {i+1} samples")
        break

OUTPUT_DIR = "sample_images"
NUM_SAMPLES = 10 
IMAGE_SIZE = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)

indices = random.sample(range(len(y)), NUM_SAMPLES)

for i, idx in enumerate(indices):
    img = X[idx].reshape(IMAGE_SIZE, IMAGE_SIZE) 
    label_idx = y[idx]
    label_name = CLASSES[label_idx]
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    out_path = os.path.join(OUTPUT_DIR, f"{i:03d}_{label_name}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

print(f"Saved {NUM_SAMPLES} sample images to {OUTPUT_DIR}/")

limit = 10_000_000
counter = Counter(y[:limit])
least_common = counter.most_common()[:-6:-1]

print("5 least frequent labels in first 1,000,000 rows:")
for label, count in least_common:
    print(f"Label {label} occurred {count} times")