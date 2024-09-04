import sys
import time
import logging
import argparse
from above_shrubs.model.chm_pipeline import CHMPipeline


# -----------------------------------------------------------------------------
# main
#
# python chm_pipeline_cli.py -c config.yaml -s preprocess train predict
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to perform CNN regression.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-f',
                        '--force-cleanup',
                        required=False,
                        default=False,
                        action='store_true',
                        dest='force_cleanup',
                        help='Cleanup of lock files in prediction')

    parser.add_argument('-m',
                        '--model-filename',
                        type=str,
                        required=False,
                        default=None,
                        dest='model_filename',
                        help='Path to model file')

    parser.add_argument('-o',
                        '--output-dir',
                        type=str,
                        default=None,
                        required=False,
                        dest='output_dir',
                        help='Path to output directory')

    parser.add_argument('-r',
                        '--regex-list',
                        type=str,
                        nargs='*',
                        required=False,
                        dest='inference_regex_list',
                        help='Inference regex list',
                        default=None)

    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=[
                            'all', 'setup', 'preprocess',
                            'train', 'predict', 'validate'],
                        choices=[
                            'all', 'setup', 'preprocess',
                            'train', 'predict', 'validate'])

    args = parser.parse_args()

    # Setup timer to monitor script execution time
    timer = time.time()

    # Initialize pipeline object
    pipeline = CHMPipeline(
        args.config_file,
        args.model_filename,
        args.output_dir,
        args.inference_regex_list
    )

    # Regression CHM pipeline steps
    if "setup" in args.pipeline_step or "all" in args.pipeline_step:
        pipeline.setup()
    if "preprocess" in args.pipeline_step or "all" in args.pipeline_step:
        pipeline.preprocess()
    if "train" in args.pipeline_step or "all" in args.pipeline_step:
        pipeline.train()
    if "predict" in args.pipeline_step or "all" in args.pipeline_step:
        pipeline.predict(args.force_cleanup)
    if "validate" in args.pipeline_step or "all" in args.pipeline_step:
        pipeline.validate()

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
