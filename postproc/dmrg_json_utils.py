"""
@defgroup   JSONUtilities JSON Utilities
@brief      Python module for reading JSON data files produced by a DMRG.x application.
"""

##  @addtogroup JSONUtilities
#   @{

import json
from shutil import copyfile

def LoadJSONFile(file,appendStr,funcName,count=0):
    """
    Loads data from a JSON file and corrects unfinished runs by appending a string to the file.
    """
    filemod = file[:-5]+"-mod"+".json"
    try:
        if count == 0:
            data = json.load(open(file))
        elif count == 1:
            data = json.load(open(filemod))
        else:
            raise ValueError('LoadJSONFile was not able to correct the file "{}" with an appended "{}". '
                'Check the file manually.'.format(file,appendStr))
        return data
    except json.JSONDecodeError:
        copyfile(file,filemod)
        fh = open(filemod, "a")
        fh.write(appendStr)
        fh.close()
        return LoadJSONFile(file,appendStr,funcName,count+1)

def LoadJSONDict(file):
    """
    Loads data from a JSON file with keys "headers" and "table", and corrects unfinished runs by
    appending "}".
    """
    return LoadJSONFile(file,"}","LoadJSONDict")

def LoadJSONTable(file):
    """
    Loads data from a JSON file with keys "headers" and "table", and corrects unfinished runs by
    appending "]}".
    """
    return LoadJSONFile(file,"]}","LoadJSONTable")

def LoadJSONArray(file):
    """
    Loads data from a JSON file represented as an array of dictionary entries and corrects
    unfinished runs by appending "]".
    """
    return LoadJSONFile(file,"]","LoadJSONArray")

##
#   @}
