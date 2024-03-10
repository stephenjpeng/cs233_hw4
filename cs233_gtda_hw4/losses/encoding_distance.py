from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors


class EncodingDistance:
    def __init__(self, golden_part_dist_file='../data/golden_dists.npz'):
        golden_data = np.load(golden_part_dist_file, allow_pickle=True)
        self.golden_part_dist = golden_data['golden_part_dist']
        self.golden_names = golden_data['golden_names']

        # Load/organize golden part-aware distances.
        self.sn_id_to_parts = defaultdict(list)
        self.id_to_part_loc = dict()
        
        for i, name in enumerate(self.golden_names):
            # Extract shape-net model ids of golden, map them to their parts.
            sn_id, _, part_id, _, _ = name.split('_')
            self.sn_id_to_parts[sn_id].append(part_id)
        
            # Map shape-net model id and part_id to location in distance matrix, (the order is the same).
            self.id_to_part_loc[(sn_id, part_id)] = i

    def calculate(self, latent_codes, test_names):
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(latent_codes)

        encoding_distances = np.zeros(len(test_names))
        num_shared_parts = np.zeros(len(test_names))
        latent_distances = np.empty(len(test_names))
        for i, sn_name in enumerate(test_names):
            parts_of_model = set(self.sn_id_to_parts[sn_name])

            nn_distances, nn_indices = nn.kneighbors([latent_codes[i]],  return_distance=True)
            latent_distances[i] = nn_distances[0, 1]
            matched_neighbor = test_names[nn_indices[0, 1]] # Students find the model's name of the Nearest-Neighbor
            parts_of_neighbor = set(self.sn_id_to_parts[matched_neighbor])

            # compute the requested distances.
            # Use id_to_part_loc for each model/part combination

            parts_in_both = parts_of_model.intersection(parts_of_neighbor)
            for k in parts_in_both:
                encoding_distances[i] += self.golden_part_dist[
                        self.id_to_part_loc[(sn_name, k)],
                        self.id_to_part_loc[(matched_neighbor, k)]]
                num_shared_parts[i] += 1

            parts_only_model = parts_of_model.difference(parts_of_neighbor)
            parts_only_neighbor = parts_of_neighbor.difference(parts_of_model)

            # for k in parts_only_B:
            #     encoding_distances[i] += max([golden_part_dist[id_to_part_loc[(matched_neighbor, k)], id_to_part_loc[(sn_name, u)]] for u in parts_of_model])

            cand_distances = [0] * 4
            for u in parts_of_model:
                for k in parts_only_neighbor:
                    cand_distances[int(u) - 1] += self.golden_part_dist[
                            self.id_to_part_loc[(matched_neighbor, k)],
                            self.id_to_part_loc[(sn_name, u)]]
            encoding_distances[i] += max(cand_distances)
        return {
            'enc_dist': encoding_distances.sum(),
            'shared_pts': num_shared_parts.mean(),
            'latent_dist': latent_distances.mean()
        }
