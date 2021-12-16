"""Scrape data from Planetary Data System and write to file.
Take URL and create several objects relevant to webscraping project.

Top level window is https://atmos.nmsu.edu/PDS/data/mslrem_1001/DATA/

r = request.get(base)
parentdir_r = bs4.BeautifulSoup(r.content, "html.parser")
|------------------
|DIR   .....
|SOL_00001_00089/
|....
|....
|....
|------------------
    |
    | r = request.get(base/parent)
    | subdir_temp_r = bs4.BeautifulSoup(r.content, "html.parser")
    V
    |------------------
    |DIR   .....
    |SOL_00001/
    |....
    |....
    |....
    |------------------
        |
        | request.get(base/parent/sub)
        |
        V
        |------------------
        |File .....
        |RME_...._ADR
        |....
        |....
        |....
        |------------------
            |
            |
            V

Inspection of these pages reveals that 
<table>
    <tbody>
        <tr>
        <tr>
        ....
    </tbody>
</table>
contains the data necessary for webscraping.
At the lowest level, the downloadable url is contained in a href="..."
https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url-in-requests
"""

import bs4
import requests
import time
import random
import os


class ScrapePDS:
    """Interface for requests & bs4 of NASA's Planetary Data System."""

    def __init__(self, url, write_path):
        """Get the top level html content and init instance vars.

        :param url: Where to start scraping.

        :self.__url: Landing page for data.
        :self.__parent_dirs: 1D list of directory ranges from HTML 
            table in self.__url.
        :self.__all_dirs: 2D list column 0 is directory range. 
            (parent directory) and column 1 is 1D list of subdirectories 
            in that directory range.
            [[parent1, list(subdirs1), ...]]
            For example -- 
            0 [ ['SOL_00001_00089/', ['SOL00001/', 'SOL00009/', ...]]
            1   ['SOL_00090_00179/', ['SOL00090/', 'SOL00091/', ...]]
            n ]
        :self.__some_dirs: Partial structure of var self.__all_dirs.
        :self.__all_data: 1D list of files already in data 
            directory.
        """

        self.__url = url
        self.__write_path = write_path
        time.sleep(random.uniform(1, 3))
        self.__top_level_response = requests.get(self.__url).content
        self.__top_level_soup = bs4.BeautifulSoup(self.__top_level_response,
                                                  "html.parser")

        # Encapsulate this into its own class?
        self.__parent_dirs = []
        self.__all_dirs = []
        self.__some_dirs = []
        self.__all_data = []

    def requests_parent_directories(self):
        """Makes and returns a list of top_level parent directories."""

        # Make an HTML table containing the parent directories
        parent_html_table = [str(i) for i in
                             self.__top_level_soup.find_all("tr")]

        # Create list by finding SOL_STARTFOLDER_ENDFOLDER
        for tr in parent_html_table:
            if (tr.find("SOL") != -1):
                start_index = tr.find("SOL")
                stop_index = tr.find("/", start_index)
                dir_name = tr[start_index:stop_index + 1]
                self.__parent_dirs.append(dir_name)

        return self.__parent_dirs

    def requests_all_directories(self, path=None):
        """Return table [[parent1, [subdir1, subdir2, ]], [...]]"""

        # Set default value for path
        if (path is None):
            path = self.__write_path

        # Enter parent directory, then build sub-directory
        for parent_dir in self.__parent_dirs:
            # Prevent blacklisting
            time.sleep(random.uniform(10, 30))

            # Parent directory response ~/SOL_START_END/
            next_url = self.__url + parent_dir
            parent_dir_response = requests.get(next_url).content
            parent_dir_soup = bs4.BeautifulSoup(parent_dir_response,
                                                "html.parser")

            # Construct list of subdirs in parentdirs ['SOL00001/', ...]
            sub_dir_html_table = [str(i) for i in
                                  parent_dir_soup.find_all("tr")]
            sub_dirs = []
            for tr in sub_dir_html_table:
                if (tr.find("SOL") != -1):
                    start_index = tr.find("SOL")
                    stop_index = tr.find("/", start_index)
                    dir_name = tr[start_index:stop_index + 1]
                    sub_dirs.append(dir_name)

            # Write to parent_subdir_structure/
            self.__write_all_dirs(parent_dir, sub_dirs,
                                  path=r"C:\Dev\python\ML_Data_Analysis\BMI\scrape\class_reorganization\parent_subdir_structure")

            # Append to lists
            row_for_all_dirs = [parent_dir]
            row_for_all_dirs.append(sub_dirs)  # [parent/,list(subdir)]
            self.__all_dirs.append(row_for_all_dirs)

        return self.__all_dirs

    def requests_some_directories(self, start_parent_dir, end_parent_dir=None):
        """Returns partial parent-sub list; args are inclusive."""

        # Default value for end_parent_dir
        if (end_parent_dir is None):
            end_parent_dir = self.__parent_dirs[-1]

        # Build partial list
        start_index = self.__parent_dirs.index(start_parent_dir)
        end_index = self.__parent_dirs.index(end_parent_dir)
        for i in range(start_index, end_index + 1):
            # Prevent blacklisting
            time.sleep(random.uniform(10, 30))

            # Request only those parent dirs in the range
            url = self.__url + self.__parent_dirs[i]
            r = requests.get(url).content
            soup = bs4.BeautifulSoup(r, "html.parser")

            # Build list of subdirectories
            html_table = [str(i) for i in soup.find_all("tr")]
            sub_dirs = []
            for tr in html_table:
                if (tr.find("SOL") != -1):
                    start_index = tr.find("SOL")
                    stop_index = tr.find("/", start_index)
                    dir_name = tr[start_index:stop_index + 1]
                    sub_dirs.append(dir_name)

            # Append to partial list
            row_for_some_dirs = [self.__parent_dirs[i]]
            row_for_some_dirs.append(sub_dirs)
            self.__some_dirs.append(row_for_some_dirs)

        return self.__some_dirs

    def set_parent_and_all_dirs(self, dir_list):
        """If dirs.txt on disk, use this to self.__all_dirs"""

        self.__all_dirs = dir_list
        self.__set_parent_dirs()

    def __set_parent_dirs(self):
        """Takes self.__all_dirs structure and makes parent dirs lst. """

        for row in self.__all_dirs:
            self.__parent_dirs.append(row[0])

    def set_all_data(self, files_on_disk):
        """Takes SolDataOnDisk object and sets self.__all_data."""

        self.__all_data = files_on_disk.get_all_data()

    def __write_all_dirs(self, parent, subdirs,
                         path=None):
        r"""Writes self.__all_dirs to file on disk.

        Format of .txt file is 
        'parent\tsubdir1,subdir2,subdirN\n'
        """

        with open(path + "/dirs.txt", "a+") as fobj:
            fobj.write(parent + "\t")
            for folder in subdirs:
                if (folder != subdirs[-1]):
                    fobj.write(folder + ",")
                else:
                    fobj.write(folder + "\n")

    def write_data(self, dir_list, path=None):
        """Download the REMS RDR data.

        1:  Create sol_data directory if it doesn't already exist.
        2:  Iterate through list of directories and get response content
            for base/parent/subdir since that will have the <table>
            with .TAB and .LBL files. The file to download is the 7th
            element (6th index) of the <table>
        3:  Download this by getting the content at this position 
            i.e. r1 = requests.get(base/parent/subdir/).content
                r2 = requests.get(base/parent/subdir/data.TAB).content
            Then write that HTTP response to disk using 'wb'
            Must be 'wb' since r_obj.content returns binary.
        4: Write the data to disk
        """

        # Set default path
        if (path is None):
            path = self.__write_path

        index = 0
        # Needed directories are in parent SOL_02225_02358
        for row in dir_list:
            base_url = self.__url + row[0]  # ~/parent_dir/
            for sub_dir in row[1]:  # ['SOL0000Nth/']
                # Prevent blacklisting
                time.sleep(random.uniform(10, 30))

                # HTTP request and BeautifulSoup
                folder_response = requests.get(base_url + sub_dir).content
                folder_soup = bs4.BeautifulSoup(folder_response,
                                                "html.parser")

                # Extract data file name from HTML table
                data_html_table = folder_soup.find_all("tr")
                data_fname = self.__extract_fname(data_html_table)

                # Check if file already downloaded
                if(data_fname not in self.__all_data):
                    # Get file response content and write to disk
                    data_url = base_url + sub_dir + data_fname
                    file_response = requests.get(data_url,
                                                 allow_redirects=True).content

                    # Write to disk
                    with open(path + "/" + data_fname, "wb") as fobj:
                        fobj.write(file_response)

            index += 1  # Update index of row in dir_list

    def __extract_fname(self, html_table):
        """Searchs HTML table for RMD file and returns its name."""

        # <tr> of <table> to string
        html_table = [str(tr) for tr in html_table]

        # Search the table for RMD & select latest .TAB
        for tr in html_table:
            if (tr.find("RMD") != -1 and (tr.find(".TAB") != -1)):
                start_index = tr.find("RME")
                stop_index = tr.find("TAB", start_index) + len("TAB")
                fname = tr[start_index:stop_index]

        # Return name of file to download
        return fname
