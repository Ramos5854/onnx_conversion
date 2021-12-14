# Description: This file contain basic utility functions that are intended to be used within the following files:
#     ONNX_Master.ipynb
#     convertToONNX.py

import os
import shutil

from os import path
# ----------------------------------------------------
def CheckDirectories(onnxPath, outPath):
# Description: Create ONNX and end-pipeline folders if not already existing in cwd
    
    # Check for ONNX output folder
    if not path.exists(onnxPath):
        os.mkdir(onnxPath)
        
    # Check for end output folder
    if not path.exists(outPath):
        os.mkdir(outPath)
# ----------------------------------------------------
def CopyToDirectory(path, file):
# Description: Copy file to current working directory
    
    try:
        shutil.copy(path, os.getcwd())
        print("'", file, "' copied to current working directory...")
    except shutil.SameFileError:
        pass
    
# ----------------------------------------------------
def ChangeExtension(fileIn, extOut):
# Description: Change file extension type
    ind = fileIn.find('.') + 1
    base = fileIn[0:ind]
    fileOut = fileIn[0:ind] + extOut
    
    return fileOut, base
# ----------------------------------------------------
def GetExtension(fileIn):
# Description: Get extension type from file name
    ind = fileIn.find('.') + 1
    ext = fileIn[ind:len(fileIn)]
    
    return ext
# ----------------------------------------------------
def ModelSavedDialogue(modelSaved):
# Description: Print saved status upon creating ONNX and end output files
    import os
    
    print("Dir: ", os.getcwd())
    print("Model saved: ", modelSaved)