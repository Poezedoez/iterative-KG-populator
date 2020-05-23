import sys
sys.path.append('../distantly-supervised-dataset/')
from DistantSupervisor import DistantSupervisor
from argparser import get_parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    supervisor = DistantSupervisor(
        data_path=args.data_path,
        ontology_version=4,
        output_path=args.output_path,
        timestamp_given=args.timestamp_given,
        cos_theta=args.cos_theta,
        filter_sentences=args.filter_sentences,
        token_pooling=args.token_pooling,
        mention_pooling=args.mention_pooling
    )
    supervisor.supervise(label_strategy=args.label_strategy, selection=tuple(args.selection))