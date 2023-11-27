import numpy as np
import argparse
import cs
import js

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Path to Data', type=str)
    parser.add_argument('-s', help='Random Seed', type=int)
    parser.add_argument('-m', help='Similarity Measure', type=str, choices=['js', 'cs', 'dcs'])
    args = parser.parse_args()

    print(f'Running {args.m} with seed {args.s}...')
    # run the specified similarity measure
    if args.m == 'cs' or args.m == 'dcs':
        cs.main(args)
    elif args.m == 'js':
        js.main(args)