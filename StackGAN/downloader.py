from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8',
                                    dest_path='./dataset/flowers.hdf5',
                                    unzip=True)