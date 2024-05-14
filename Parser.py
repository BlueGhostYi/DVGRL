import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='DVGRL')

    parser.add_argument('--path', type=str, default='./Data/',help='Movielens-20m dataset location')
    
    parser.add_argument("--model", default="DVGRL", help="Model to train")
    
    parser.add_argument('--dataset', type=str, default='douban-book', help='Movielens-20m dataset location')
    
    parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate')
    
    parser.add_argument('--dims', type=int, default=50, help='embedding size')
    
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay coefficient')
    
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    
    parser.add_argument('--epochs', type=int, default=3000, help='upper epoch limit')

    parser.add_argument('--stop', type=int, default=20, help='early stop')
    
    parser.add_argument('--anneal_cap', type=float, default=1.0, help='largest annealing parameter')
    
    parser.add_argument('--seed', type=int, default=2023, help='random seed')

    parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='report interval')

    parser.add_argument('--sparsity', type=bool, default=False, help='Sparsity Level')

    parser.add_argument('--show_loss', type=bool, default=False, help='show loss in each epoch')

    parser.add_argument('--write', type=bool, default=True, help='write results in txt')
    
    return parser.parse_args()
