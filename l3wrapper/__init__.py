"""l3wrapper - A simple Python 3 wrapper around L3 binaries."""

import logging
from os import mkdir, listdir, remove, chmod
from os.path import expanduser, exists, join, basename
from sys import platform
import requests
from tqdm import tqdm
import zipfile
import stat


__version__ = '0.4.1'
__author__ = 'Giuseppe Attanasio <giuseppe.attanasio@polito.it>'
__all__ = []


# Check if L3C binaries are installed and download them if necessary
user_home = expanduser("~")
l3wrapper_data_path = join(user_home, "l3wrapper_data")
bin_path = join(l3wrapper_data_path, "bin")

if not exists(l3wrapper_data_path):
    print(f"Creating {l3wrapper_data_path} to store binaries")
    try:
        mkdir(l3wrapper_data_path)
    except:
        raise RuntimeError(f"Could not create {l3wrapper_data_path}")


def platform_download():
    URL_OSX = "https://dbdmg.polito.it/wordpress/wp-content/uploads/2020/02/L3C_osx1015.zip" 
    URL_LINUX = "https://dbdmg.polito.it/wordpress/wp-content/uploads/2020/03/L3C_ubuntu1804.zip"
    URL_WIN32 = None

    try:    
        if platform == "darwin":
            filename = basename(URL_OSX)
            file_path = join(l3wrapper_data_path, filename)
            r = requests.get(URL_OSX, stream=True)
            with open(file_path, "wb") as fp:
                for data in tqdm(r.iter_content(chunk_size=None)):
                    fp.write(data)
        elif platform == "linux":
            filename = basename(URL_LINUX)
            file_path = join(l3wrapper_data_path, filename)
            r = requests.get(URL_LINUX, stream=True)
            with open(file_path, "wb") as fp:
                for data in tqdm(r.iter_content(chunk_size=None)):
                    fp.write(data)
        elif platform == "win32":
            raise NotImplementedError("Binaries for this OS are not available.")
        else:
            raise ValueError(f"The OS {platform} is not supported.")
    except:
        raise RuntimeWarning("Something went wrong while downloading. Check your internet connection.")

    with zipfile.ZipFile(file_path, "r") as zip_file:
        zip_file.extractall(l3wrapper_data_path)

    # Give to the owner the executable permissions
    [chmod(join(l3wrapper_data_path, "bin", rf), stat.S_IRWXU) for rf in required_files]

    print("Download completed")


required_files = ["convertitoreRegCompatteNonCompatte",
                  "DBcoverage",
                  "fpMacroRulesClassiFiltriItem",
                  "L3CFiltriItemClassifica",
                  "L3CFiltriItemTrain",
                  "leggiBin"]

logger = logging.getLogger(__file__)
if not exists(bin_path):
    print(f"{','.join(required_files)} are missing.\n Downloading...")
    platform_download()
else:
    binaries = listdir(bin_path)
    missings = [rf for rf in required_files if rf not in binaries]
    if missings:
        print(f"{','.join(missings)} are missing.\n Downloading...")
        platform_download()
    else:
        logger.info(f"L3C binaries are present. Using them for each classifier instance by default.")
