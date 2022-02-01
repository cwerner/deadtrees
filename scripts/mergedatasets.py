import argparse
import re
import shutil
import sys
from pathlib import Path

from deadtrees.data.deadtreedata import DeadtreeDatasetConfig, split_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indirs", type=Path, nargs="+")

    parser.add_argument(
        "--outdir",
        dest="outdir",
        type=Path,
        default=Path("data/dataset"),
        help="output directory for merged dataset",
    )

    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if len(args.indirs) < 2:

        print("At least two indirs are required!\n")
        parser.print_help()
        sys.exit(1)

    # find year in path str
    years = [re.search(r"\d{4}", str(d)) for d in args.indirs]
    years_extracted = [y.group() for y in years if y]

    if len(years_extracted) != len(args.indirs):
        print("Extracting year info from indirs failed!\n")
        parser.print_help()
        sys.exit(1)

    # create train, validation, test folders
    (args.outdir / "train").mkdir(parents=True, exist_ok=True)
    (args.outdir / "val").mkdir(parents=True, exist_ok=True)
    (args.outdir / "test").mkdir(parents=True, exist_ok=True)

    for year, indir in zip(years_extracted, args.indirs):

        def copy_to_dst(files, subdir):
            for infile in files:
                infile = Path(infile)
                f = infile.name.split("-0")
                outfile = args.outdir / subdir / f"{f[0]}-{year}-0{f[1]}"
                shutil.copyfile(str(infile), str(outfile))

        train_files, val_files, test_files = split_shards(
            sorted(indir.glob("*.tar")), DeadtreeDatasetConfig.fractions
        )

        copy_to_dst(train_files, "train")
        copy_to_dst(val_files, "val")
        copy_to_dst(test_files, "test")


if __name__ == "__main__":
    main()
