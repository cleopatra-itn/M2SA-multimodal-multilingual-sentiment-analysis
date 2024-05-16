import os
import json
import shutil

def append_string_to_file(text, file_path):
    file = open(file_path, 'a')
    file.write(text+'\n')
    file.close()


def save_list_to_file(input_list: list, file_path):
    file = open(file_path, 'w', encoding='utf-8')
    file.write("\n".join(str(item) for item in input_list))
    file.close()

def save_list_to_json_file(input_list: list, file_path):
    with open(file_path, 'w', encoding='utf8') as json_file:
        for i in input_list:
            json.dump(i, json_file, ensure_ascii=False)
            json_file.write("\n")

def file_exists(file_path):
    return os.path.exists(file_path)

def save_string_to_file(text, file_path):
    file = open(file_path, 'w')
    file.write(text)
    file.close()

def read_file_to_set(file_path):
    content = set()
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            for l in  file.readlines():
                content.add(l.strip())
    return content

def read_file_to_list(file_path):
    content = list()
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            for l in  file.readlines():
                content.append(l.strip().replace("\n", ""))
    return content

def read_file_to_string(file_path):
    content = ''
    if os.path.isfile(file_path):
        with open(file_path, "r") as file:
            for l in  file.readlines():
                content += l
    return content

def read_json_file(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        return data

def path_exists(dir_path):
    return os.path.exists(dir_path)

def create_folder(dir_path):
    if not path_exists(dir_path):
        os.mkdir(dir_path)
    pass

def delete_files_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_files_in_directory(dir_path):
    files = os.listdir(dir_path)

    valid_files = []
    for f in files:
        if f =='.DS_Store':
            continue
        else:
            valid_files.append(f)

    return valid_files