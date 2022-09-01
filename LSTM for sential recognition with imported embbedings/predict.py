import sys
import os
from train import train
from evaluate import evaluate

def main():
    test_path = sys.argv[1]
    train()
    evaluate(test_path)
    os.remove('model_vocab.txt')



if __name__ == '__main__':
    main()
