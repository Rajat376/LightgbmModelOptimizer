from lightgbm import Booster
from .utils import StringFileReader, StringFileWriter
import json
from ._tree import _Tree


class Optimizer:
    def __init__(self, pool_size=-1):
        self.pool_size = pool_size
        if pool_size == -1:
            from multiprocessing import cpu_count
            self.pool_size = cpu_count()

    @staticmethod
    def get_trees_and_other_info(file: StringFileReader):
        trees = []
        file.reset()
        line = file.readline()
        while line:
            if line.startswith('Tree'):
                tree_content = {}
                index = line
                while line:
                    line = file.readline()
                    if line.startswith('Tree') or line.startswith("end of trees"):
                        break
                    content = line.strip()
                    if content.startswith('split_feature'):
                        tree_content['split_feature'] = content
                    if content.startswith('threshold'):
                        tree_content['threshold'] = content
                    if content.startswith('cat_boundaries'):
                        tree_content['cat_boundaries'] = content
                    if content.startswith('decision_type'):
                        tree_content['decision_type'] = content
                    if content.startswith('cat_threshold'):
                        tree_content['cat_threshold'] = content
                tree_content['num_feature'] = num_feature
                trees.append(_Tree(index,
                                   tree_content['split_feature'],
                                   tree_content['threshold'],
                                   tree_content['cat_boundaries'],
                                   tree_content['decision_type'],
                                   tree_content['cat_threshold'],
                                   tree_content['num_feature']
                                   )
                             )
            else:
                if line.startswith('[categorical_feature:'):
                    cat_ids = [int(x.strip()) for x in line.strip()[len("[categorical_feature:"):-1].split(",")]
                elif line.startswith("pandas_categorical"):
                    pandas_categorical = json.loads(line[len("pandas_categorical:"):])
                elif line.startswith("max_feature_idx"):
                    num_feature = int(line[len("max_feature_idx="):]) + 1
                line = file.readline()
        return trees, cat_ids, pandas_categorical, num_feature

    @staticmethod
    def get_optimized_model_string(trees: [_Tree], file: StringFileReader, new_pandas_cat: [[]]) -> str:
        Writer = StringFileWriter()
        trees = sorted(trees, key=lambda tree: tree.index)
        line = file.readline()
        while line:
            if line.startswith('Tree'):
                content = line.strip()
                tree_index = int(content.split("=")[1])
                Writer.write(line)
                while line:
                    line = file.readline()
                    if line.startswith('Tree') or line.startswith("end of trees"):
                        break
                    if line.startswith('cat_boundaries'):
                        Writer.writeline(trees[tree_index].new_cat_boundaries)
                    elif line.startswith('cat_threshold'):
                        Writer.writeline(trees[tree_index].new_cat_threshold)
                    else:
                        Writer.write(line)
            else:
                if line.startswith("pandas_categorical:"):
                    Writer.write("pandas_categorical:")
                    Writer.writeline(json.dumps(new_pandas_cat))
                    line = file.readline()
                elif line.startswith("tree_sizes"):
                    line = file.readline()
                else:
                    Writer.write(line)
                    line = file.readline()
        return str(Writer)

    @staticmethod
    def get_feature_id_mappings(results, num_feature) -> [dict]:
        used_feat_indices = [[] for _ in range(num_feature)]
        for result in results:
            for i in range(num_feature):
                used_feat_indices[i].extend(result[i])
        feature_ind_mappings = [{} for _ in range(num_feature)]

        for i in range(num_feature):
            feature_ind_mappings[i] = {feature_id: True for feature_id in used_feat_indices[i]}

        for i in range(num_feature):
            cnt = 0
            for feat_id in feature_ind_mappings[i].keys():
                feature_ind_mappings[i][feat_id] = cnt
                cnt += 1
        return feature_ind_mappings

    @staticmethod
    def get_modified_pandas_categorical(pandas_categorical: [[]], cat_ids: [int], feature_id_mappings: [{}]) -> [
        []]:
        new_pandas_cat = [[] for _ in range(len(cat_ids))]
        for i, j in enumerate(cat_ids):
            new_pandas_cat[i] = [None for _ in range(len(feature_id_mappings[j]))]
            for feat_id, new_feat_id in feature_id_mappings[j].items():
                new_pandas_cat[i][new_feat_id] = pandas_categorical[i][feat_id]
        return new_pandas_cat

    @staticmethod
    def get_used_feat_val_map(x):
        return x.get_used_feat_val_map()

    @staticmethod
    def get_optimized_tree_string(tree, feature_id_mappings):
        tree.get_optimized_tree_string(feature_id_mappings)
        return tree

    def get_optimized_model_string_from_file_reader(self, file:StringFileReader) -> str:
        trees, cat_ids, pandas_categorical, num_feature = self.get_trees_and_other_info(file)
        file.reset()
        if self.pool_size == 1:
            results = [self.get_used_feat_val_map(x) for x in trees]
        else:
            from multiprocessing import Pool
            with Pool(self.pool_size) as p:
                results = p.map(self.get_used_feat_val_map, trees)
        feature_id_mappings = self.get_feature_id_mappings(results, num_feature)
        if self.pool_size == 1:
            trees = [self.get_optimized_tree_string(x[0], x[1]) for x in [(tree, feature_id_mappings) for tree in trees]]
        else:
            from multiprocessing import Pool
            with Pool(self.pool_size) as p:
                trees = p.starmap(self.get_optimized_tree_string, [(tree, feature_id_mappings) for tree in trees])
        new_pandas_cat = self.get_modified_pandas_categorical(pandas_categorical, cat_ids, feature_id_mappings)
        optimized_model_str = self.get_optimized_model_string(trees, file, new_pandas_cat)
        return optimized_model_str

    def optimize_model_string(self, model_str: str) -> str:
        file_reader = StringFileReader(model_str)
        return self.get_optimized_model_string_from_file_reader(file_reader)

    def optimize_booster(self, raw_model: Booster) -> Booster:
        model_str = raw_model.model_to_string()
        optimized_model_str = self.optimize_model_string(model_str)
        optimized_booster = Booster(model_str=optimized_model_str)
        return optimized_booster

    def optimize_model_file(self, source_path: str, destination_path: str = None) -> None:
        if destination_path is None:
            destination_path = source_path
        model_file = open(source_path, 'r')
        model_file_reader = StringFileReader(model_file.read())
        optimized_model_str = self.get_optimized_model_string_from_file_reader(model_file_reader)
        with open(destination_path, 'w') as file:
            file.write(optimized_model_str)
