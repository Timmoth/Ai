import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import time
import os

CLASSES = ["fan", "moon", "microphone", "calculator", "van", "spider", "parrot", "piano", "scorpion", "broccoli", "sea turtle", "envelope", "mouth", "birthday cake", "beard", "rake", "motorbike", "teddy-bear", "cell phone", "airplane", "hedgehog", "snake", "mermaid", "nail", "shark", "key", "horse", "cloud", "ambulance", "saw", "cactus", "dumbbell", "triangle", "mountain", "bus", "donut", "light bulb", "baseball bat", "book", "clock", "submarine", "radio", "camouflage", "picture frame", "stethoscope", "bulldozer", "lollipop", "string bean", "streetlight", "lantern", "teapot", "postcard", "fence", "animal migration", "campfire", "bucket", "pickup truck", "penguin", "hurricane", "ant", "circle", "butterfly", "popsicle", "ocean", "speedboat", "lion", "lobster", "diving board", "flower", "truck", "fire hydrant", "wine glass", "sink", "rainbow", "eyeglasses", "pineapple", "television", "roller coaster", "screwdriver", "sandwich", "bread", "paint can", "sun", "flashlight", "hot dog", "passport", "cookie", "eye", "apple", "candle", "raccoon", "zigzag", "ladder", "barn", "sleeping bag", "microwave", "dog", "river", "stop sign", "trumpet", "telephone", "wheel", "helicopter", "sword", "skyscraper", "anvil", "pillow", "power outlet", "hockey puck", "mailbox", "foot", "carrot", "bear", "shovel", "spoon", "cup", "fork", "church", "bee", "pig", "finger", "bird", "megaphone", "snowflake", "owl", "eraser", "alarm clock", "clarinet", "shoe", "lipstick", "ice cream", "hand", "knee", "skateboard", "pool", "tractor", "whale", "nose", "hamburger", "baseball", "pear", "waterslide", "laptop", "mouse", "tiger", "car", "bottlecap", "stove", "snorkel", "square", "strawberry", "rollerskates", "boomerang", "rain", "harp", "bat", "The Great Wall of China", "The Mona Lisa", "door", "see saw", "paintbrush", "flip flops", "kangaroo", "school bus", "asparagus", "computer", "The Eiffel Tower", "pencil", "hat", "drums", "bed", "cooler", "face", "t-shirt", "hexagon", "tennis racquet", "arm", "house", "washing machine", "binoculars", "smiley face", "tree", "squiggle", "mosquito", "pond", "trombone", "keyboard", "bush", "marker", "bandage", "umbrella", "lightning", "wine bottle", "bathtub", "hot air balloon", "floor lamp", "toe", "necklace", "saxophone", "blueberry", "oven", "tornado", "frog", "toaster", "guitar", "line", "dragon", "feather", "swan", "flying saucer", "police car", "headphones", "elephant", "steak", "sheep", "peanut", "snail", "castle", "spreadsheet", "garden hose", "zebra", "hot tub", "cruise ship", "shorts", "grass", "mushroom", "chandelier", "swing set", "golf club", "bracelet", "crown", "octagon", "traffic light", "calendar", "pliers", "bridge", "octopus", "cow", "toothpaste", "garden", "tooth", "beach", "fish", "cannon", "yoga", "panda", "pants", "snowman", "duck", "bicycle", "hourglass", "sock", "hospital", "skull", "compass", "flamingo", "rabbit", "wristwatch", "ear", "diamond", "sweater", "giraffe", "scissors", "crab", "brain", "crayon", "monkey", "chair", "frying pan", "belt", "broom", "elbow", "soccer ball", "jail", "mug", "helmet", "canoe", "toilet", "moustache", "basket", "rhinoceros", "dishwasher", "dolphin", "basketball", "camel", "cello", "grapes", "tent", "paper clip", "hammer", "toothbrush", "pizza", "coffee cup", "drill", "vase", "violin", "squirrel", "sailboat", "underwear", "bench", "leaf", "bowtie", "stitches", "hockey stick", "parachute", "crocodile", "suitcase", "jacket", "windmill", "cat", "lighthouse", "angel", "goatee", "cake", "stairs", "couch", "purse", "firetruck", "banana", "star", "palm tree", "fireplace", "matches", "dresser", "train", "camera", "stereo", "watermelon", "remote control", "peas", "blackberry", "onion", "table", "ceiling fan", "map", "potato", "leg", "house plant", "axe", "backpack"]
DATA_DIR = Path("./quickdraw_simplified")
MAX_PER_CLASS = 1_000_000

IMG_SIZE = 64
IMAGE_SIZE_FLAT = IMG_SIZE * IMG_SIZE

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def strokes_to_image_ndjson(drawing, size=64, line_width=2):
    """Convert stroke list from ndjson to a PIL grayscale image"""
    img = Image.new("L", (size, size), color=0)  # black background
    draw = ImageDraw.Draw(img)
    
    # Collect all points to normalize globally
    all_x = []
    all_y = []
    for stroke in drawing['drawing']:
        all_x.extend(stroke[0])
        all_y.extend(stroke[1])
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    # Prevent division by zero
    x_range = x_max - x_min if x_max > x_min else 1
    y_range = y_max - y_min if y_max > y_min else 1
    
    # Draw each stroke
    for stroke in drawing['drawing']:
        x = np.array(stroke[0])
        y = np.array(stroke[1])
        x = ((x - x_min) / x_range * (size-1)).astype(np.int32)
        y = ((y - y_min) / y_range * (size-1)).astype(np.int32)
        
        for i in range(len(x)-1):
            draw.line([x[i], y[i], x[i+1], y[i+1]], fill=255, width=line_width)
    
    return img


# ---------------- COUNT TOTAL SAMPLES ----------------
start_time = time.time()
total_samples = 0
for cls in CLASSES:
    count = 0
    with open(DATA_DIR / f"{cls}.ndjson", 'r') as f:
        for _ in f:
            count += 1
            if count >= MAX_PER_CLASS:
                break
    total_samples += count

log(f"Total samples: {total_samples} (counted in {time.time()-start_time:.2f}s)")

# ---------------- CREATE MEMMAPS ----------------
X = np.memmap("X.dat", dtype=np.uint8, mode="w+", shape=(total_samples, IMAGE_SIZE_FLAT))
y = np.memmap("y.dat", dtype=np.int32, mode="w+", shape=(total_samples,))

# ---------------- RANDOM SHUFFLE INDICES ----------------
perm = np.random.permutation(total_samples)
current_index = 0

# ---------------- STREAM AND WRITE DATA ----------------
for idx, cls in enumerate(CLASSES):
    n = 0
    with open(DATA_DIR / f"{cls}.ndjson", 'r') as f:
        for i, line in enumerate(f):
            if i >= MAX_PER_CLASS:
                break
            drawing = json.loads(line)
            img = strokes_to_image_ndjson(drawing, size=IMG_SIZE)
            
            # Optional: save PNGs for inspection
            # img.save(OUTPUT_DIR / f"{cls}_{i}.png")
            
            # Flatten image and write to memmap at shuffled position
            X[perm[current_index]] = np.array(img, dtype=np.uint8).ravel()
            y[perm[current_index]] = idx
            current_index += 1
            n += 1
    log(f"Added {n} samples from class '{cls}' (label={idx})")

# ---------------- FLUSH MEMMAPS ----------------
X.flush()
y.flush()
log("Finished writing memmaps X.dat and y.dat")