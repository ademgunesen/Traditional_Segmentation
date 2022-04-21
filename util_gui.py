import utils as util
import os
import skimage.io as io
import csv
from gooey import Gooey, GooeyParser

def process_dir(img_in_dir, img_out_dir, preprocessing):
    print(f'Started to {preprocessing} preprocessing')
    entries = sorted(os.listdir(img_in_dir))
    for entry in entries:
        print(entry)
        img_in = io.imread(img_in_dir + "/" + entry)
        if preprocessing == "Adems":
            img_out = util.adems_processing(img_in)
        elif preprocessing == "Bens":
            img_out = util.bens_processing(img_in)
        io.imsave(img_out_dir + "/" + entry, img_out, check_contrast=False)
    
@Gooey(
    program_name="Image Processor",
    program_description="Process an image using a preprocessing function",
    image_dir='icons/'
    # tabbed_groups=True,
    # navigation='Tabbed'
)
def main():
    parser = GooeyParser()
    tab1 = parser.add_argument_group("Folder Selections")
    tab1.add_argument(
        "-i",
        "--input_folder", 
        required=True,
        help="Image folder to be processed",
        widget="DirChooser")

    tab1.add_argument(
        "-o",
        "--output_folder",
        required=True,
        help="Path for the processed image",
        widget="DirChooser")
        
    tab2 = parser.add_argument_group("Select preprocessing")
    
    tab2.add_argument(
        "--preprocessing",
        required=True,
        widget="Dropdown",
        choices=['Bens', 'Adems'],
        help='Select preprocessing'
    )
    args = parser.parse_args()
    
    process_dir(args.input_folder, args.output_folder, args.preprocessing)
    
if __name__ == "__main__":
    main()