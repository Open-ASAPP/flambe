from flambe import log
import argparse


def my_script(arg1: str, arg2: str, kwarg1: int, kwarg2: str):
    """Test script"""
    for i in range(10):
        msg = f'arg1: {arg1}, '
        msg += f'arg2: {arg2}, '
        msg += f'kwarg1: {kwarg1}, '
        msg += f'kwarg2: {kwarg2}'
        log(msg, float(i), global_step=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1', type=str)
    parser.add_argument('arg2', type=str)
    parser.add_argument('--kwarg1', type=int)
    parser.add_argument('--kwarg2', type=str)
    args = parser.parse_args()
    my_script(args.arg1, args.arg2, args.kwarg1, args.kwarg2)
