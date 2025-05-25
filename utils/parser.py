import argparse

def parse_args_kgsr():
    parser = argparse.ArgumentParser(description="Ours")
    parser.add_argument("--cuda", type=int, default=1, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="music", help="Choose a dataset:[music,book,movie]")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")
    parser.add_argument('--T', type=int, default=1, help='threshold C for interest-aware task')
    parser.add_argument('--C', type=int, default=2, help='threshold C for user interest entity set construction')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")

    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=2000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='test batch size')
    parser.add_argument('--activate', nargs='?', default="tanh", help='[tanh, sigmoid, relu]')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')

    parser.add_argument('--user_interest_graph_hop', type=int, default=3, help='[1,2,3,4]')
    parser.add_argument('--knowledge_graph_hop', type=int, default=1, help='[1,2,3,4]')
    parser.add_argument('--user_item_graph_hop', type=int, default=3, help='[1,2,3,4]')
    parser.add_argument('--l2', type=float, default=5e-2, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--cl_rate", type=float, default=0.1, help="cl_rate")
    parser.add_argument("--il_rate", type=float, default=0.1, help="il_rate")
    parser.add_argument("--tau", type=float, default=0.3, help="cl_tau")

    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    # ===== Test ===== #
    parser.add_argument('--test_batch', type=int, default=10, help='test_batch')
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50, 100]', help='Output sizes of every layer')

    return parser.parse_args()


