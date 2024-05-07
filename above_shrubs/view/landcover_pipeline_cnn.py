import sys
import time
import logging
import argparse
from above_shrubs.model.landcover_pipeline import LandCoverPipeline


# -----------------------------------------------------------------------------
# main
#
# python landcover_pipeline_cli.py -c config.yaml \
#   -d data.csv -s preprocess train predict
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to perform CNN segmentation.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-d',
                        '--data-csv',
                        type=str,
                        required=False,
                        dest='data_csv',
                        help='Path to the data configuration file')

    # parser.add_argument('-f',
    #                    '--force-cleanup',
    #                    type=bool,
    #                    required=False,
    #                    default=False,
    #                    action='store_true',
    #                    dest='force_cleanup',
    #                    help='Cleanup of lock files from parallel prediction')

    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=[
                            'all', 'preprocess',
                            'train', 'predict', 'validate'],
                        choices=[
                            'all', 'preprocess',
                            'train', 'predict', 'validate'])

    args = parser.parse_args()

    # Setup timer to monitor script execution time
    timer = time.time()

    # Initialize pipeline object
    pipeline = LandCoverPipeline(args.config_file, args.data_csv)

    # Segmentation CNN pipeline steps
    if "preprocess" in args.pipeline_step or "all" in args.pipeline_step:
        pipeline.preprocess(enable_multiprocessing=True)
    if "train" in args.pipeline_step or "all" in args.pipeline_step:
        pipeline.train()
    if "predict" in args.pipeline_step or "all" in args.pipeline_step:
        pipeline.predict()
    # if "validate" in args.pipeline_step or "all" in args.pipeline_step:
    #    pipeline.validate()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
