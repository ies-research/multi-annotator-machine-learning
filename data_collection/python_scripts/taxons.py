import numpy as np

TAXON_DICT = {
    # List of big cats.
    "Jaguar": 41970,
    "Leopard": 41963,
    "Cheetah": 41955,
    # List of insects.
    "European Paper Wasp": 84640,
    "European Hornet": 54327,
    "German Yellowjacket": 126155,
    "Yellow-legged Hornet": 119019,
    # List of rabbits and hares.
    "European Rabbit": 43151,
    "Brown Hare": 43128,
    "Marsh Rabbit": 43102,
    "Black-tailed Jackrabbit": 43130,
    "Desert Cottontail": 43115,
    # List of squirrels.
    "Douglas' Squirrel": 46259,
    "American Red Squirrel": 46260,
    "Eurasian Red Squirrel": 46001,
}

GROUPS_DICT = {
    "Big Cats": ["Jaguar", "Leopard", "Cheetah"],
    "Hares and Rabbits": [
        "European Rabbit",
        "Brown Hare",
        "Marsh Rabbit",
        "Black-tailed Jackrabbit",
        "Desert Cottontail",
    ],
    "Insects": [
        "European Paper Wasp",
        "European Hornet",
        "German Yellowjacket",
        "Yellow-legged Hornet",
    ],
    "Squirrels": [
        "Douglas' Squirrel",
        "American Red Squirrel",
        "Eurasian Red Squirrel",
    ],
}

CLASSES = np.array(
    [
        "German Yellowjacket",
        "European Paper Wasp",
        "Yellow-legged Hornet",
        "European Hornet",
        "Brown Hare",
        "Black-tailed Jackrabbit",
        "Marsh Rabbit",
        "Desert Cottontail",
        "European Rabbit",
        "Eurasian Red Squirrel",
        "American Red Squirrel",
        "Douglas' Squirrel",
        "Cheetah",
        "Jaguar",
        "Leopard",
    ],
    dtype=object,
)

ANNOTATORS = np.array(
    [
        "digital-dragon",
        "pixel-pioneer",
        "ocean-oracle",
        "starry-scribe",
        "sunlit-sorcerer",
        "emerald-empath",
        "sapphire-sphinx",
        "echo-eclipse",
        "lunar-lynx",
        "neon-ninja",
        "quantum-quokka",
        "velvet-voyager",
        "radiant-raven",
        "dreamy-drifter",
        "azure-artist",
        "twilight-traveler",
        "galactic-gardener",
        "cosmic-wanderer",
        "frosty-phoenix",
        "mystic-merlin",
    ],
    dtype=object,
)
