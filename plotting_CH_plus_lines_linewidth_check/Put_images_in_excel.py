import os
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image
from PIL import Image as PILImage

# Create a new workbook and select the active worksheet
workbook = Workbook()
worksheet = workbook.active

# Directory where your images are stored

image_dir = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/plotting_CH_plus_lines_linewidth_check/continuum_subtracted'

# Loop through the files and add them to the Excel file
files = os.listdir(image_dir)

# Loop through the files and add them to the Excel file
for index, file in enumerate(files):
    if file.endswith(('jpeg', 'jpg', 'png')):  # Add other file types if needed
        img = Image(os.path.join(image_dir, file))
        
        # Adjust the size of the image if necessary (optional)
        pil_img = PILImage.open(os.path.join(image_dir, file))
        img.width, img.height = pil_img.size

        # Add the image to the worksheet
        worksheet.add_image(img, f'A{index + 1}')

        # Insert filename in the next column
        worksheet[f'B{index + 1}'] = file

        # Set the row height to match the image height (optional)
        worksheet.row_dimensions[index + 1].height = img.height



excel_file = '/Users/charmibhatt/Library/CloudStorage/OneDrive-TheUniversityofWesternOntario/UWO_onedrive/Local_GitHub/DIBs/plotting_CH_plus_lines_linewidth_check/continuum_subtracted/Dtermining_continuum_for_CH_plus.xlsx'

# Save the workbook
workbook.save(excel_file)

