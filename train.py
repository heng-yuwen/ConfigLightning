from argparse import ArgumentParser
import experiments
import datasets
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which experiment, dataset to use
    parser.add_argument("--experiment_name", type=str, default="SparseSemanticSegmentor", help="SparseSemanticSegmentor"
                                                                                               " by default")
    parser.add_argument("--dataset_name", type=str, default="XJ3SegmentDataModule", help="XJ3Segment by default")

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    args, _ = parser.parse_known_args()
    experiment = experiments.get_experiment(args.experiment_name)
    dataset = datasets.get_dataset(args.dataset_name)
    LightningCLI(experiment, dataset)
