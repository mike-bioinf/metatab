import argparse



def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("-f", "--file-estimator", required=True,
                   help="Pickle file of the fitted estimator to use in prediction.")
    
    p.add_argument("-i", "--input-data", required=True, 
                   help="Path of the file/folder containing the data on which doing predictions.")
    
    p.add_argument("-m", "--input-mode", required=True, choices=["sets", "xy", "df"],
                   help="Defines the data input format. One of 'sets', 'xy', or 'df'.")
    
    p.add_argument("-y", "--target-feature", default=None,
                    help="Name of the target feature column. Must be provided if --input-mode is equal to 'df'.")
    
    p.add_argument("-x", "--x-uniform", action="store_true",
                   help="Uniform the test data feature space to the one seen by the estimator at fit level, before doing predictions")

    p.add_argument("-o", "--output-dir", default=".",        
                   help="""Path of the folder in which the output file is created.
                   If not provided the folder in which the program is run is used.
                   Note that one must provide only the output folder path without a filename.
                   The name of the created file is automatically inferred and must not be specified.""")
    
    p.add_argument("--create-outdir", action="store_true", help="Create the output directory if does not exists.")
    
    return p.parse_args(args)