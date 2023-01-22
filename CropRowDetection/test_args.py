from argparse import ArgumentParser
import argparse

if __name__ == "__main__":
    p = ArgumentParser(description="Detect crop rows!")
    p.add_argument("mode", choices=["video", "picture"], default="video", help= 'specify mode ')
    p.add_argument("-k", type=int, default=5, help="number of frames before complete process (video only)", required=False)
    p.add_argument("-n", type=int, default=5, help="number of crop rows", required=False)
    p.add_argument("--name", "-p", type=str, help="name of the picture", required=True)


    args = p.parse_args()

    if args.mode == "video":
        print("we're in video mode!")
        print("k is", args.k)
    elif args.mode == "picture":
        print("we're in picture mode!")
        print('nb crop tbd : ', args.n)