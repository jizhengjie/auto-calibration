import os, sys,time, json, logging, argparse
import numpy as np

from method import *
from configs import *
from utils import *

# Logging level
logging.basicConfig(level=logging.DEBUG)

# Set argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="sample_data/",
                    help="Relative path to the directory containing images to detect.")
parser.add_argument("--overwrite-log", default=False,
                    help="Overwrite log / append the log at the end.")


def main():
    FLAGS = parser.parse_args()

    # Load data
    logging.info("Loading data...")
    try:
        data_path = FLAGS.data_path
        data = load_data(data_path)
    except BaseException:
        logging.critical("Failed to load data.")
        return
    else:
        logging.info("Successfully loaded data.")

    # Run algorithm
    start = time.time()
    logging.info("Running calibration...")
    f = FullyAutomaticCalibration(data)
    output = f.run() # a dict of R and T 
    end = time.time()
    t = round(end-start, 3)
    logging.info("Finish calibration in "+str(t)+"s.")

    # Write output
    logging.info("Writing output to log file...")
    if FLAGS.overwrite_log:
        with open('log.json','w') as f:
            json.dump(output, f)
    else:
        with open('log.json','a') as f:
            json.dump(output, f)
    logging.info("Successfully wrote log.")


if __name__ == "__main__":
    main()
