from astropy.io import fits
from astropy.table import Table
import numpy as np

# def print_fits_file(filename):
#     # Open the FITS file
#     with fits.open(filename) as hdul:
#         # Print the header of the primary HDU
#         hdul[0].header.totextfile('stdout', overwrite=True)

#         # Print the data of the primary HDU
#         print(hdul[0].data)

#         # If there are other HDUs, you can similarly print their headers and data
#         # for hdu in hdul[1:]:
#         #     print("\n\n--- Next HDU ---\n\n")
#         #     hdu.header.totextfile('stdout', overwrite=True)
#         #     print(hdu.data)


def print_fits_content(filename):
    # Open the FITS file
    with fits.open(filename) as hdul:
        # Loop through each HDU and print its content
        for idx, hdu in enumerate(hdul):
            print(f"========== HDU {idx} ==========")
            print("\n---- Header ----\n")
            hdu.header.totextfile('stdout', overwrite=True)
            
            print("\n---- Data ----\n")
            print(type(hdu.data))
            print("\n\n")

output_path = 'fits_header.txt'

def print_fits_header(file_path):
    # Open the FITS file using astropy
    with fits.open(file_path) as hdul:
        # Access the header of the primary HDU (Header Data Unit)
        header_data = hdul[0].header

        # Print each header card
        for card in header_data.cards:
            print(card)
            
        with open(output_path, 'w') as txt_file:
            for card in header_data.cards:
                txt_file.write(str(card) + '\n')


# Use the function
#direc = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Research/EDIBLES/DR4_all/HD166937/RED_564/HD166937_w564_redl_20170508_O1.fits"

#direc = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Research/EDIBLES/DR4_all_merged/HD166937/RED_564/HD166937_w564_n10_20170508_L.fits"
direc = "/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Research/EDIBLES/DR4_all_merged/HD166937/RED_564/HD166937_w564_n10_20170508_U.fits"
#filename = direc
print_fits_header(direc)

# from astropy.io import fits
# fits_image_filename = fits.util.get_testdata_filepath(direc)

# # hdul = fits.open(fits_image_filename)
# # print(hdul)

# Table.read(direc)