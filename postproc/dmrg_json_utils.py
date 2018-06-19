"""
@package dmrg_json_utils
@brief Python module for reading JSON data files produced by a DMRG.x application.

@defgroup   JSONUtilities JSON Utilities
@brief      Python module for reading JSON data files produced by a DMRG.x application.
"""

##
#   @addtogroup JSONUtilities
#   @{

import json
from shutil import copyfile

def LoadJSONFile(path,appendStr,count=0):
    """ Loads data from a JSON file and corrects unfinished runs by appending a
    string to the file.

    Args:
        path:           path to json file
        appendStr:      string to append in case of a JSONDecodeError
        count:          number of times this function was called recursively

    Returns:
        dict object representing the loaded data

    Raises:
        ValueError: Unable to correctly load the file with the appended string.
    """
    filemod = path[:-5]+"-mod"+".json"
    try:
        if count == 0:
            data = json.load(open(path))
        elif count == 1:
            data = json.load(open(filemod))
        else:
            raise ValueError('LoadJSONFile was not able to correct the file "{}" with an appended "{}". '
                'Check the file manually.'.format(path,appendStr))
        return data
    except json.JSONDecodeError:
        copyfile(path,filemod)
        fh = open(filemod, "a")
        fh.write(appendStr)
        fh.close()
        return LoadJSONFile(path,appendStr,count+1)

def LoadJSONDict(path):
    """ Loads data from a JSON file containing dict entries and corrects unfinished
    runs by appending "}".

    Args:
        path:           path to json file

    """
    return LoadJSONFile(path,"}")

def LoadJSONTable(path):
    """Loads data from a JSON file with keys "headers" and "table", and corrects
    unfinished runs by appending "]}".

    Args:
        path:           path to json file
    """
    return LoadJSONFile(path,"]}")

def LoadJSONArray(path):
    """Loads data from a JSON file represented as an array of dictionary entries
    and corrects unfinished runs by appending "]".

    Args:
        path:           path to json file
    """
    return LoadJSONFile(path,"]")

##
#   @}
