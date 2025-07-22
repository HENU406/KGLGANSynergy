from model import *


def parse_method(args, n, edge_num, edge_dim, c, d, device):
    if args.method == 'KGLGANSynergy':
        model = KGLGANSynergy(n, edge_num, edge_dim, d, args.hidden_channels, c, args.lgat_hidden_channels, args.lgat_head,
                         graph_weight=args.graph_weight, aggregate=args.aggregate,
                         trans_num_layers=args.trans_num_layers, trans_dropout=args.trans_dropout,
                         trans_num_heads=args.trans_num_heads, trans_use_bn=args.trans_use_bn,
                         trans_use_residual=args.trans_use_residual, trans_use_weight=args.trans_use_weight,
                         trans_use_act=args.trans_use_act,
                         gnn_num_layers=args.gnn_num_layers, gnn_dropout=args.gnn_dropout, gnn_use_bn=args.gnn_use_bn,
                         gnn_use_residual=args.gnn_use_residual, gnn_use_weight=args.gnn_use_weight,
                         gnn_use_init=args.gnn_use_init, gnn_use_act=args.gnn_use_act,
                         ).to(device)
    else:
        raise ValueError('Invalid method')
    return model

# OS
# def parser_add_main_args(parser):
#     # dataset and evaluation
#     parser.add_argument('--dataset', type=str, default='OS')
#     parser.add_argument('--data_dir', type=str, default='../data_set/OS/')
#     parser.add_argument('--device', type=int, default=0,
#                         help='which gpu to use if any (default: 0)')
#     parser.add_argument('--seed', type=int, default=123)
#     parser.add_argument('--cpu', action='store_true')
#     parser.add_argument('--epochs', type=int, default=1000)
#     parser.add_argument('--runs', type=int, default=1,
#                         help='number of distinct runs')
#     parser.add_argument('--split', type=int, default=5, help='cv5')
#
#     # gnn branch
#     parser.add_argument('--method', type=str, default='KGLGANSynergy')
#     parser.add_argument('--hidden_channels', type=int, default=128)
#     parser.add_argument('--use_graph', action='store_true', help='use input graph')
#     parser.add_argument('--aggregate', type=str, default='add', help='MCA aggregate type, add or avg.')
#     parser.add_argument('--graph_weight', type=float, default=0.5, help='graph weight.')
#     parser.add_argument('--lgat_hidden_channels', type=int, default=64, help='edge dim and lgat hidden channels')
#     parser.add_argument('--n_dim', type=int, default=64, help='node dim')
#     parser.add_argument('--lgat_head', type=int, default=4, help='lgat multiple head nums')
#
#     parser.add_argument('--gnn_use_bn', action='store_true', help='use batchnorm for each GNN layer')
#     parser.add_argument('--gnn_use_residual', action='store_true', help='use residual link for each GNN layer')
#     parser.add_argument('--gnn_use_weight', action='store_true', help='use weight for GNN convolution')
#     parser.add_argument('--gnn_use_init', action='store_true', help='use initial feat for each GNN layer')
#     parser.add_argument('--gnn_use_act', action='store_true', help='use activation for each GNN layer')
#     parser.add_argument('--gnn_num_layers', type=int, default=1, help='number of layers for GNN')
#     parser.add_argument('--gnn_dropout', type=float, default=0.0)
#     parser.add_argument('--gnn_weight_decay', type=float, default=1e-4)
#
#     # all-pair attention (Transformer) branch
#     parser.add_argument('--trans_num_heads', type=int, default=1, help='number of heads for attention')
#     parser.add_argument('--trans_use_weight', action='store_true', help='use weight for trans convolution')
#     parser.add_argument('--trans_use_bn', action='store_true', help='use layernorm for trans')
#     parser.add_argument('--trans_use_residual', action='store_true', help='use residual link for each trans layer')
#     parser.add_argument('--trans_use_act', action='store_true', help='use activation for each trans layer')
#     parser.add_argument('--trans_num_layers', type=int, default=2, help='number of layers for all-pair attention.')
#     parser.add_argument('--trans_dropout', type=float, default=0.0, help='gnn dropout.')
#     parser.add_argument('--trans_weight_decay', type=float, default=1e-6)
#
#     # training
#     parser.add_argument('--lr', type=float, default=0.0003)
#     parser.add_argument('--batch_size', type=int, default=32, help='batch size')
#     parser.add_argument('--patience', type=int, default=15, help='early stopping patience.')


# DrugCombDB
def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='DrugCombDB')
    parser.add_argument('--data_dir', type=str, default='../data_set/DrugCombDB/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--split', type=int, default=5, help='cv5')

    # gnn branch
    parser.add_argument('--method', type=str, default='KGLGANSynergy')
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--use_graph', action='store_true', help='use input graph')
    parser.add_argument('--aggregate', type=str, default='avg', help='MCA aggregate type, add or avg.')
    parser.add_argument('--graph_weight', type=float, default=0.5, help='graph weight.')
    parser.add_argument('--lgat_hidden_channels', type=int, default=256, help='edge dim and lgat hidden channels')
    parser.add_argument('--n_dim', type=int, default=256, help='node dim')
    parser.add_argument('--lgat_head', type=int, default=4, help='lgat multiple head nums')

    parser.add_argument('--gnn_use_bn', action='store_true', help='use batchnorm for each GNN layer')
    parser.add_argument('--gnn_use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--gnn_use_weight', action='store_true', help='use weight for GNN convolution')
    parser.add_argument('--gnn_use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--gnn_use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--gnn_num_layers', type=int, default=2, help='number of layers for GNN')
    parser.add_argument('--gnn_dropout', type=float, default=0.0)
    parser.add_argument('--gnn_weight_decay', type=float, default=1e-4)

    # all-pair attention (Transformer) branch
    parser.add_argument('--trans_num_heads', type=int, default=1, help='number of heads for attention')
    parser.add_argument('--trans_use_weight', action='store_true', help='use weight for trans convolution')
    parser.add_argument('--trans_use_bn', action='store_true', help='use layernorm for trans')
    parser.add_argument('--trans_use_residual', action='store_true', help='use residual link for each trans layer')
    parser.add_argument('--trans_use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--trans_num_layers', type=int, default=2, help='number of layers for all-pair attention.')
    parser.add_argument('--trans_dropout', type=float, default=0.0, help='gnn dropout.')
    parser.add_argument('--trans_weight_decay', type=float, default=1e-6)

    # training
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience.')
