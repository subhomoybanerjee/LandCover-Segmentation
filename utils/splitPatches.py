import os
import shutil
import time
import splitfolders

def splitPatches(root,outputFolder):
    inputFolder = root

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    splitfolders.ratio(inputFolder, output=outputFolder, seed=42, ratio=(.80, .10, .10), group_prefix=None)

def main():
    root='../datasets/allPatches'
    outputFolder='../datasets/split'
    splitPatches(root,outputFolder)

if __name__ == '__main__':
    main()

