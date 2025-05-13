import argparse


def parse_args(args):
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter) 

    p.add_argument("-m", "--model", required=True, help="Pickle file of the serialized model object.")

    p.add_argument("-i", "--input-mode", required=True, choices=['sets', 'df', 'xy'],
                   help="""Define the input data format. One of 'sets', 'xy' and 'df'.
                   In case of 'sets' only the test "named" files are loaded.""")
    
    p.add_argument("-y", "--target-feature", default=None, 
                    help="Name of the target feature column. Must be provided if --input-mode is equal to 'df'")
    
    return p.parse_args(args)