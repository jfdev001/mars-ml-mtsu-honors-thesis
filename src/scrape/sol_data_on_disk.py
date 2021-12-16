"""Class that encapsulates data stored in cwd and raw_data/.

Subdir format is 'SOL00001' etc. 'SOL' is always followed by 5 total
digits so number of leading zeroes is 5 - len(num).

The .TAB files are formated such that only 4 digits are shown
e.g. 0001, 0099, 0112, 1230, etc. 
"""

import os

class SolDataOnDisk:
    """Encapsulate all data on disk pertaining to PDS Scrape"""

    def __init__(self, read_path):
        """Construct class with empty lists for for sol data.

        :param path: path for write directory

        :self.__all_data: 1D list of all .TAB files in raw_data/ 
            directory
        :self.__adr_data: 1D list of all ADR .TAB files in raw_data/ 
            directory
        :self.__formatted_sub_dirs: 1D list of subdirectories based on 
            converted from list of regular files ... i.e.
            RME_......RMD000D____.TAB
        """

        self.__all_data = os.listdir(read_path)  # Modularizable?
        self.__adr_data = []
        self.__formatted_sub_dirs = []
        self.__all_dirs_on_disk = []


    def __build_adr_data(self):
        """Iterate through self.__all_data and append any ADR files."""

        for ele in self.__all_data:
            if (ele.find("ADR") != -1):
                self.__adr_data.append(ele)

    def __read_dirs_on_disk(self):
        r"""Reads and makes __all_dirs structure from file on disk.

            Format of dirs.txt file is 
            'parent\tsubdir1,subdir2,subdirN\n'
            """
        
        with open("./parent_subdir_structure/dirs.txt", "r") as fobj:
            for line in fobj:
                temp_row = []
                line = line.strip()
                parent = line.split("\t")[0]
                subdirs_list = line.split("\t")[1].split(",")
                temp_row.append(parent)
                temp_row.append(subdirs_list)
                self.__all_dirs_on_disk.append(temp_row)


    def get_dirs_on_disk(self):
        """Returns [[parent, list(subdirs)], ...] data structure."""

        self.__read_dirs_on_disk()
        return self.__all_dirs_on_disk


    def get_all_data(self):
        """Returns a list of all files in raw_data/ directory"""

        return self.__all_data

    
    def get_adr_data(self):
        """Returns a list containing all files with ADR in the name.
        
        Variable file type?
        """

        self.__build_adr_data()
        return self.__adr_data

    
    def get_formatted_sub_dirs(self, my_list, data_product):
        """Returns list of 'SOL#####/' sub directory format.

        :param my_list: A 1D list to convert to the subdir format
        :param data_product: A string indicating the data product
        
        :return: list
        """
        # Iterate through my_list and convert to subdir format
        for ele in my_list:
            start_index = ele.find(data_product) + len(data_product)
            stop_index = start_index + 4  # Constant number of digits
            sol_day = ele[start_index:stop_index]
            subdir_format = "SOL" + ((5 - len(sol_day)) * "0") + sol_day + "/"
            self.__formatted_sub_dirs.append(subdir_format)

        return self.__formatted_sub_dirs




