import sys.path
sys.path.append('../distantly-supervised-dataset/')
from DistantlySupervisedDatasets import DistantlySupervisedDatasets, get_parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    dataset = DistantlySupervisedDatasets(args.ontology_entities_path, args.ontology_relations_path, args.document_path,
                                         args.entity_embedding_path, args.output_path, args.timestamp_given, args.cos_theta)
    dataset.create(label_function=args.label_function, selection=tuple(args.selection))