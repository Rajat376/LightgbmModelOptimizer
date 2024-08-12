class _Tree:
    # required keys = split_feature,threshold,cat_boundaries,decision_type,cat_threshold,num_feature

    def __init__(self, index, split_feature, threshold, cat_boundaries, decision_type, cat_threshold, num_feature):
        self.new_cat_boundaries = None
        self.new_cat_threshold = None
        self.index = index
        self.split_feature = split_feature
        self.threshold = threshold
        self.cat_boundaries = cat_boundaries
        self.decision_type = decision_type
        self.cat_threshold = cat_threshold
        self.num_feature = num_feature
        self.are_elements_extracted = False

    def extract_tree_elements_from_model_string(self):
        self.split_feature = [int(x) for x in self.split_feature.split("=")[1].split(" ")]
        self.threshold = [float(x) for x in self.threshold.split("=")[1].split(" ")]
        self.cat_boundaries = [int(x) for x in self.cat_boundaries.split("=")[1].split(" ")]
        self.decision_type = [int(x) for x in self.decision_type.split("=")[1].split(" ")]
        self.cat_threshold = [int(x) for x in self.cat_threshold.split("=")[1].split(" ")]
        self.index = int(self.index.strip().split("=")[1])
        self.are_elements_extracted = True

    @staticmethod
    def is_decision_categorical(decision_type) -> bool:
        if (decision_type & 1) > 0:
            return True
        else:
            return False

    def get_used_feat_val_map(self) -> [[]]:
        if not self.are_elements_extracted:
            self.extract_tree_elements_from_model_string()
        used_feat_indices = [[] for _ in range(self.num_feature)]
        for node_index, decision_type in enumerate(self.decision_type):
            if self.is_decision_categorical(decision_type):
                cat_boundary_index = int(self.threshold[node_index])
                for cat_threshold_index in range(self.cat_boundaries[cat_boundary_index], self.cat_boundaries[cat_boundary_index + 1]):
                    if self.cat_threshold[cat_threshold_index] == 0:
                        continue
                    for bit in range(32):
                        if (self.cat_threshold[cat_threshold_index] & (1 << bit)) > 0:
                            feat_val_index = (cat_threshold_index - self.cat_boundaries[cat_boundary_index]) * 32 + bit
                            used_feat_indices[self.split_feature[node_index]].append(feat_val_index)
        return used_feat_indices

    def get_optimized_tree_string(self, feat_val_mappings):
        if not self.are_elements_extracted:
            self.extract_tree_elements_from_model_string()
        self.new_cat_boundaries = [0]
        self.new_cat_threshold = []
        for node_index, decision_type in enumerate(self.decision_type):
            if self.is_decision_categorical(decision_type):
                cat_boundary_index = int(self.threshold[node_index])
                feat_val_mapping = feat_val_mappings[self.split_feature[node_index]]
                total = int((len(feat_val_mapping) - 1) / 32 + 1)
                self.new_cat_boundaries.append(self.new_cat_boundaries[cat_boundary_index] + total)
                used_feat_val_indices = set()
                for cat_threshold_index in range(self.cat_boundaries[cat_boundary_index], self.cat_boundaries[cat_boundary_index + 1]):
                    if self.cat_threshold[cat_threshold_index] == 0:
                        continue
                    for bit in range(32):
                        if (self.cat_threshold[cat_threshold_index] & (1 << bit)) > 0:
                            feat_val_index = (cat_threshold_index - self.cat_boundaries[cat_boundary_index]) * 32 + bit
                            used_feat_val_indices.add(feat_val_mapping[feat_val_index])
                for cat_threshold_index in range(total):
                    num = 0
                    for bit in range(32):
                        feat_val_index = bit + cat_threshold_index * 32
                        if feat_val_index in used_feat_val_indices:
                            num += (1 << bit)
                    self.new_cat_threshold.append(num)
        self.new_cat_boundaries = 'cat_boundaries=' + ' '.join(map(str, self.new_cat_boundaries))
        self.new_cat_threshold = 'cat_threshold=' + ' '.join(map(str, self.new_cat_threshold))
