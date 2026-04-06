class _Tree:
    #required keys = split_feature,threshold,cat_bounderies,decision_type,cat_threshold,num_feature

    def __init__(self, index, split_feature, threshold, cat_bounderies, decision_type, cat_theshold, num_feature):
        self.index = index
        self.split_feature = split_feature
        self.threshold = threshold
        self.cat_boundaries = cat_bounderies
        self.decision_type = decision_type
        self.cat_threshold = cat_theshold
        self.num_feature = num_feature

    def preprocess(self):
        self.split_feature=[int(x) for x in self.split_feature.split("=")[1].split(" ")]
        self.threshold=[float(x) for x in self.threshold.split("=")[1].split(" ")]
        if self.cat_boundaries is not  None:
            self.cat_boundaries=[int(x) for x in self.cat_boundaries.split("=")[1].split(" ")]
        self.decision_type=[int(x) for x in self.decision_type.split("=")[1].split(" ")]
        if self.cat_threshold is not None:
            self.cat_threshold=[int(x) for x in self.cat_threshold.split("=")[1].split(" ")]
        self.index=int(self.index.strip().split("=")[1])

    def isDecisionCategorical(self,decision_type)->bool:
        if (decision_type&1) >0:
            return True
        else:
            return False

    def process_data(self)->[[]]:
        self.preprocess()
        feature_map=[[]for _ in range(self.num_feature) ]
        for idx,decision_type in enumerate(self.decision_type):
            if self.isDecisionCategorical(decision_type):
                cat_index=int(self.threshold[idx])
                for i in range(self.cat_boundaries[cat_index],self.cat_boundaries[cat_index+1]):
                    if self.cat_threshold[i]==0:
                        continue
                    for j in range(32):
                        if (self.cat_threshold[i]&(1<<j)) > 0:
                            feature_index=(i-self.cat_boundaries[cat_index])*32+j
                            feature_map[self.split_feature[idx]].append(feature_index)
        return feature_map

    def post_process(self,feature_mappings):
        self.preprocess()
        self.new_cat_boundaries=[0]
        self.new_cat_threshold=[]
        for idx,decision_type in enumerate(self.decision_type):
            if self.isDecisionCategorical(decision_type):
                cat_index=int(self.threshold[idx])
                feat_map=feature_mappings[self.split_feature[idx]]
                total=int((len(feat_map)-1)/32+1)
                self.new_cat_boundaries.append(self.new_cat_boundaries[cat_index]+total)
                used_indices=set()
                for i in range(self.cat_boundaries[cat_index],self.cat_boundaries[cat_index+1]):
                    if self.cat_threshold[i]==0:
                        continue
                    for j in range(32):
                        if (self.cat_threshold[i]&(1<<j)) > 0:
                            feature_index=(i-self.cat_boundaries[cat_index])*32+j
                            used_indices.add(feat_map[feature_index])
                for i in range(total):
                    num=0
                    for j in range(32):
                        ind=j+i*32
                        if ind in used_indices:
                            num+=(1<<j)
                    self.new_cat_threshold.append(num)
        if self.cat_boundaries is not None:
            self.new_cat_boundaries='cat_boundaries='+' '.join(map(str, self.new_cat_boundaries))
        if self.new_cat_threshold is not None:
            self.new_cat_threshold='cat_threshold='+ ' '.join(map(str, self.new_cat_threshold))
