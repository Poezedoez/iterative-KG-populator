import sys.path
sys.path.append('../distantly-supervised-dataset/')
os.system
from DistantlySupervisedDatasets import DistantlySupervisedDatasets

class DSDWrapper:
    def __init__(
        self,
        data_path='data/scientific_documents.json',
        types='types.json',
        tokenizer=None,
        save_path=''
    ):


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a distantly supervised dataset of scientific documents')
    parser.add_argument('--ontology_entities_path', type=str, default="data/ontology_entities.csv",
                        help="path to the ontology entities file")
    parser.add_argument('--ontology_relations_path', type=str, default="data/ontology_relations.csv",
                        help="path to the ontology relations file")
    parser.add_argument('--document_path', type=str, help='path to the folder containing scientific documents',
                        default="data/ScientificDocuments/")
    parser.add_argument('--output_path', type=str, default="data/DistantlySupervisedDatasets/", help="output path")
    parser.add_argument('--entity_embedding_path', type=str, default="data/entity_embeddings.json",
                        help="path to file of precalculated lexical embeddings of the entities")
    parser.add_argument('--selection', type=int, nargs=2, default=None,
                        help="start and end of file range for train/test split")
    parser.add_argument('--label_function', type=int, default=2, choices=range(0, 3),
                        help="0 = string, 1 = embedding, 2 = string + embedding")
    parser.add_argument('--timestamp_given', default=False, action="store_true")
    parser.add_argument('--cos_theta', type=float, default=0.83,
                        help="similarity threshold for embedding based labeling")
    args = parser.parse_args()
    dataset = DistantlySupervisedDatasets(args.ontology_entities_path, args.ontology_relations_path, args.document_path,
                                         args.entity_embedding_path, args.output_path, args.timestamp_given, args.cos_theta)
    dataset.create(label_function=args.label_function, selection=tuple(args.selection))