from train_image import main_image
from utils import parse_args


def main_segment(args):
    pass


def main():
    args = parse_args()
    if args.train_type == "image":
        main_image(args)
    elif args.train_type == "segment":
        main_segment(args)
    else:
        raise Exception("Train type not found.")


if __name__ == "__main__":
    main()
