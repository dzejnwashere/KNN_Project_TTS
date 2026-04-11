import json

fixed_path = "/home/alex/Documents/KNN/datasets/bea_Angry/"
manifest_path = "/home/alex/Documents/KNN/KNN_Project_TTS/scripts/manifest-angry-test.json"

content = ""

with open("test-files.txt", "r") as file_list:
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