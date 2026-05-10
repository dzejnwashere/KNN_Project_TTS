import json
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

fixed_path = "/home/alex/Documents/KNN/all-datasets/"
manifest_path = "/home/alex/Documents/KNN/KNN_Project_TTS/scripts/manifest-all.json"

content = ""

with open("files.txt", "r") as file_list:
    for line in file_list:
        line = line.strip()
        full_path = fixed_path + line

        tmp_json = {}
        tmp_json["audio_filepath"] = full_path
        tmp_json["text"] = ""

        if content != "":
            content = content + "\n" + json.dumps(tmp_json)
        else:
            content = content + json.dumps(tmp_json)

with open(manifest_path, "w") as manifest_file:
    manifest_file.write(content)