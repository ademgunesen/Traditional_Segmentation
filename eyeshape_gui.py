import utils as util
import os
import sys
import skimage.io as io
from skimage.morphology import disk
from gooey import Gooey, GooeyParser

def print_progress(index, total):
    print(f"Progress {int((index + 1) / total * 100)}")
    sys.stdout.flush()

def process_dir(img_in_dir, img_out_dir, threshold, labelcsv):
    with open(labelcsv, 'r') as file:
        reader = csv.reader(file,skipinitialspace=True)
        next(reader)
        index = 0
        for row in reader:
            print(entry)
            print_progress(index, 12397)
            img_in = io.imread(img_in_dir + "/" + row[0])
            gray = util.rgb2gray(img_in)
            eye_shape = gray > threshold
            my_disk = disk(512)
            mask = eye_shape*my_disk[0:1024,0:1024]
            img_out = util.adems_processing(img_in)
            proc_out = img_out*mask[...,None]
            #util.show_images([img_in, gray, eye_shape, mask, img_out, proc_out])
            io.imsave(img_out_dir + "/" + row[0], proc_out, check_contrast=False)
            index = index + 1
            print(index)
    
@Gooey(
    program_name="Get Eyeshape",
    program_description="Process an image using a preprocessing function",
    progress_regex=r"^Progress (\d+)$"
    # tabbed_groups=True,
    # navigation='Tabbed'
)
def main():
    parser = GooeyParser()
    tab1 = parser.add_argument_group("Folder Selections")
    tab1.add_argument(
        "-c",
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
        
    tab1.add_argument(
        '-i',
        metavar='Label CSV',
        help='Select CSV file',
        widget='FileChooser')

    tab2 = parser.add_argument_group("Select treshold")
    
    tab2.add_argument(
        '-thresh', 
        '--threshold', 
        default=1,
        type=int, 
        help='Threshold value for the background')
    
    args = parser.parse_args()
    
    process_dir(args.input_folder, args.output_folder, args.threshold)
    
if __name__ == "__main__":
    main()