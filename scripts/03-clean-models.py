import os

FOLDER = "output"


def clean_model(folder):
    for x in os.listdir(folder):
        if x.endswith(".pth"):
            print(x)
            parts = x.split("_")
            if "final" not in parts[1]:
                print("REMOVING", folder, x)
                os.remove(f"{folder}/{x}")
        if os.path.isdir(f"{folder}/{x}"):
            print(f"cleaning {x}")
            clean_model(f"{folder}/{x}")


if os.path.exists("core"):
    os.remove("core")

clean_model(FOLDER)
