import os
import runpy
def list_of_examples():
    file_list = []
    base_dir = os.path.join(
        os.getcwd(), "..", "examples", "scripts"
    )
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                file_list.append(os.path.join(root, file))
    return file_list

print(list_of_examples())
# print(os.path.dirname(__file__))