import heapq
import numpy as np

from .utils import read_array
from .feature_detect import get_way, way_feature
from .config import NUM_FEATURES, SIMILARITY, WAY_INDEX, DIMENSIONS, TOP_SEARCH


class FaissIndex(object):

    def __init__(self, index, way_index, id_to_vector):
        assert index
        # assert id_to_vector

        self.index = index    # lsh, L2
        self.way_index = way_index
        self.id_to_vector = id_to_vector
        self.way = get_way(way_index)

    def search_by_ids(self, ids, k):
        vectors = [self.id_to_vector(id_)[1] for id_ in ids]
        results = self.__search__(ids, vectors, k + 1)

        return results

    def search_by_vectors(self, vectors, k):
        vectors = read_array(vectors, DIMENSIONS[self.way_index])
        # ====== trick code start ===========
        count = vectors.shape[0]
        vectors = np.vstack((vectors, vectors))
        vectors = vectors[0:count, :]
        print(vectors.shape)
        # ====== trick code end ===========
        ids = [None]
        results = self.__search__(ids, [vectors], k)
        return results

    def search_by_image(self, image_f, k):
        ids = [None]
        _, vectors = way_feature(self.way, image_f)
        results = self.__search__(ids, [vectors], k)

        return results

    def __search__(self, ids, vectors, topN):
        def neighbor_dict_with_path(id_, file_path, score):
            return {'id': id_, 'file_path': file_path, 'score': score}

        def neighbor_dict(id_, score):
            return {'id': id_, 'score': score}

        def result_dict_str(id_, neighbors):
            return {'id': id_, 'neighbors': neighbors}

        results = []
        need_hit = SIMILARITY   # change to dynamic

        for id_, feature in zip(ids, vectors):
            scores, neighbors = self.index.search(feature, k=topN) if feature.size > 0 else ([], [])  #  search: np.matrix, and vector?
            # print(neighbors)
            # print(scores[0, :], np.max(scores))   # here topN may be should be big, consider many cig_box has same blod
            n = neighbors.shape[0]    # this img has n reprentation vector
            # print('num blob vector', n)
            result_dict = {}

            for i in range(n):
                l = np.unique(neighbors[i]).tolist()    # unique make e[0] / n <= 1
                # print('unique neighbors', l)
                for r_id in l:
                    if r_id != -1:    # many -1,, 
                        score = result_dict.get(r_id, 0)
                        score += 1
                        result_dict[r_id] = score    # get every vector's matched id and summed num score
            # print('jjjj',result_dict)
            h = []
            for k in result_dict:    # if len(result_dict) small  and < topN * one_num, is good
                v = result_dict[k]
                if v >= need_hit:    # need_hit is greater than topN when feature is big, is a filter thresold for diff matched id 
                    if len(h) < topN * 4:   # this topN is threshold for all vec in one image's mathed id, should be small
                        heapq.heappush(h, (v, k))    # image result len(set(k)) is small, and max(v) >> second_max(v) seems good
                    else:
                        heapq.heappushpop(h, (v, k))

            # result_list = heapq.nlargest(topN, h, key=lambda x: x[0])    # h is not empty, topN shold change to 2 or 3
            result_list = heapq.nlargest(1, h, key=lambda x: x[0])     # top 4 acc: 0.837, top2 acc:0.789, time:0.073/img, top1:0.728, 
            # print(result_list, 'xxxxx')
            neighbors_scores = []
            for e in result_list:
                confidence = e[0] * 100 / n
                if self.id_to_vector:
                    file_path = self.id_to_vector(e[1])[0]    # id_to_vector: train index from path file
                    neighbors_scores.append(neighbor_dict_with_path(e[1], file_path, str(confidence)))
                else:
                    neighbors_scores.append(neighbor_dict(e[1], str(confidence)))
            results.append(result_dict_str(id_, neighbors_scores))

        return results

