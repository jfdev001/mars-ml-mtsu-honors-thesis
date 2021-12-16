from scrape_PDS import ScrapePDS
from sol_data_on_disk import SolDataOnDisk

def main():
    # URL and instantiate objects
    url = "https://atmos.nmsu.edu/PDS/data/mslrem_1001/DATA/"
    data_path = r"C:\Dev\python\ML_Data_Analysis\BMI\data\raw_data"
    scraper = ScrapePDS(url, data_path)
    disk_data = SolDataOnDisk(data_path) 

    # Get all directories data structure on disk
    all_dirs = disk_data.get_dirs_on_disk()

    # Set variables in scraper
    scraper.set_parent_and_all_dirs(all_dirs)
    scraper.set_all_data(disk_data)

    # Get all_dirs structure for remaining data to scrape
    # Stopped in SOL_02483_02579/
    init_fix_list = []
    for row in range(-1, -4, -1):
        init_fix_list.append(all_dirs[row])
    
    # Reverse the list
    init_fix_list.reverse()

    # Rewrite this
    ele = init_fix_list[0][1][0]
    while (ele != "SOL02568/"):
        del(init_fix_list[0][1][0])
        ele = init_fix_list[0][1][0]

    # Collect data
    scraper.write_data(init_fix_list)

if __name__ == "__main__":
    main()