SPLIT_SCENES = {
    "gibson": {
        "train": [
            "Allensville",
            "Beechwood",
            "Marstons",
            "Benevolence",
            "Merom",
            "Coffeen",
            "Mifflinburg",
            "Newfields",
            "Onaga",
            "Cosmos",
            "Pinesdale",
            "Pomaria",
            "Forkland",
            "Ranchester",
            "Hanson",
            "Shelbyville",
            "Hiteman",
            "Stockman",
            "Klickitat",
            "Tolstoy",
            "Lakeville",
            "Wainscott",
            "Leonardo",
            "Lindenwood",
            "Woodbine",
        ],
        "val": [
            "Darden",
            "Collierville",
            "Markleeville",
            "Wiconisco",
            "Corozal ",
        ],
    },
    "mp3d": {
        "train": [
            "17DRP5sb8fy",
            "1LXtFkjw3qL",
            "1pXnuDYAj8r",
            "29hnd4uzFmX",
            "5LpN3gDmAk7",
            "5q7pvUzZiYa",
            "759xd9YjKW5",
            "7y3sRwLe3Va",
            "82sE5b5pLXE",
            "8WUmhLawc2A",
            "B6ByNegPMKs",
            "D7G3Y4RVNrH",
            "D7N2EKCX4Sj",
            "E9uDoFAP3SH",
            "EDJbREhghzL",
            "GdvgFV5R1Z5",
            "HxpKQynjfin",
            "JF19kD82Mey",
            "JeFG25nYj2p",
            "PX4nDJXEHrG",
            "Pm6F8kyY3z2",
            "PuKPg4mmafe",
            "S9hNv5qa7GM",
            "ULsKaCPVFJR",
            "Uxmj2M2itWa",
            "V2XKFyX4ASd",
            "VFuaQ6m2Qom",
            "VLzqgDo317F",
            "VVfe2KiqLaN",
            "Vvot9Ly1tCj",
            "XcA2TqTSSAj",
            "YmJkqBEsHnH",
            "ZMojNkEp431",
            "aayBHfsNo7d",
            "ac26ZMwG7aT",
            "b8cTxDM8gDG",
            "cV4RVeZvu5T",
            "dhjEzFoUFzH",
            "e9zR4mvMWw7",
            "gZ6f7yhEvPG",
            "i5noydFURQK",
            "jh4fc5c5qoQ",
            "kEZ7cmS4wCh",
            "mJXqzFtmKg4",
            "p5wJjkQkbXX",
            "pRbA3pwrgk9",
            "qoiz87JEwZ2",
            "r1Q1Z4BcV1o",
            "r47D5H71a5s",
            "rPc6DW4iMge",
            "s8pcmisQ38h",
            "sKLMLpTHeUy",
            "sT4fr6TAbpF",
            "uNb9QFRL6hY",
            "ur6pFq6Qu1A",
            "vyrNrziPKCB",
        ],
        "val": [
            "2azQ1b91cZZ",
            "8194nk5LbLH",
            "EU6Fwq7SyZv",
            "QUCTc6BB5sX",
            "TbHJrupSAjP",
            "X7HyMhZNoso",
            "Z6MFQCViBuw",
            "oLBMNvg9in8",
            "pLe4wQe7qrG",
            "x8F5xyUWy9e",
            "zsNo4HB9uLZ",
        ],
    },
}
OBJECT_CATEGORIES = {
    "gibson": [
        "floor",
        "wall",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "toilet",
        "tv",
        "dining-table",
        "oven",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "cup",
        "bottle",
    ],
    "mp3d": [
        "floor",
        "wall",
        "chair",
        "table",
        "picture",
        "cabinet",
        "cushion",
        "sofa",
        "bed",
        "chest_of_drawers",
        "plant",
        "sink",
        "toilet",
        "stool",
        "towel",
        "tv_monitor",
        "shower",
        "bathtub",
        "counter",
        "fireplace",
        "gym_equipment",
        "seating",
        "clothes ",
    ],
}
OBJECT_CATEGORY_MAP = {}
INV_OBJECT_CATEGORY_MAP = {}
NUM_OBJECT_CATEGORIES = {}
for dset, categories in OBJECT_CATEGORIES.items():
    OBJECT_CATEGORY_MAP[dset] = {obj: idx for idx, obj in enumerate(categories)}
    INV_OBJECT_CATEGORY_MAP[dset] = {v: k for k, v in OBJECT_CATEGORY_MAP[dset].items()}
    NUM_OBJECT_CATEGORIES[dset] = len(categories)

# -- visualization func from semantic map create
GIBSON_CATEGORIES = ["out-of-bounds"] + OBJECT_CATEGORIES["gibson"]
GIBSON_OBJECT_COLORS = [
    (0.9400000000000001, 0.7818, 0.66),
    (0.9400000000000001, 0.8868, 0.66),
    (0.8882000000000001, 0.9400000000000001, 0.66),
    (0.7832000000000001, 0.9400000000000001, 0.66),
    (0.6782000000000001, 0.9400000000000001, 0.66),
    (0.66, 0.9400000000000001, 0.7468000000000001),
    (0.66, 0.9400000000000001, 0.8518000000000001),
    (0.66, 0.9232, 0.9400000000000001),
    (0.66, 0.8182, 0.9400000000000001),
    (0.66, 0.7132, 0.9400000000000001),
    (0.7117999999999999, 0.66, 0.9400000000000001),
    (0.8168, 0.66, 0.9400000000000001),
    (0.9218, 0.66, 0.9400000000000001),
    (0.9400000000000001, 0.66, 0.8531999999999998),
    (0.9400000000000001, 0.66, 0.748199999999999),
]
GIBSON_COLOR_PALETTE = [
    1.0,
    1.0,
    1.0,  # Out-of-bounds
    0.9,
    0.9,
    0.9,  # Floor
    0.3,
    0.3,
    0.3,  # Wall
    *[oci for oc in GIBSON_OBJECT_COLORS for oci in oc],
]
GIBSON_LEGEND_PALETTE = [
    (1.0, 1.0, 1.0),  # Out-of-bounds
    (0.9, 0.9, 0.9),  # Floor
    (0.3, 0.3, 0.3),  # Wall
    *GIBSON_OBJECT_COLORS,
]

CAT_OFFSET = 1
FLOOR_ID = 1 # sem id, indicating navigatable are
MIN_OBJECTS_THRESH = 4

SCENE_DIR = "data/scene_datasets/gibson_semantic"
SCENE_CONFIG = "data/scene_datasets/gibson_semantic/gibson_semantic.scene_dataset_config.json"
SEM_MAP_SAVE_ROOT = "data/semantic_maps/gibson/semantic_maps" 
SCENE_BOUNDS_DIR = "data/semantic_maps/gibson/scene_boundaries"