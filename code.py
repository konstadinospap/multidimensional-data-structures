import hashlib
import time

import numpy as np
import pandas as pd

#gia to KDTREE

class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
                        if not points:
                            return None

                        k = len(points[0]) - 1  # Αφαιρούμε 1 γιατί το "Surname" δεν είναι διάσταση

                        axis = depth % k
                        sorted_points = sorted(points, key=lambda x: x[axis])

                        median = len(sorted_points) // 2
                        return KDNode(
                            point=sorted_points[median],
                            left=build_kdtree(sorted_points[:median], depth + 1),
                            right=build_kdtree(sorted_points[median + 1:], depth + 1)
                        )

def closest_point(root, target, depth=0, best=None):
                if root is None:
                    return best

                k = len(target) - 1  # Αφαιρούμε 1 γιατί το "Surname" δεν είναι διάσταση

                axis = depth % k

                next_best = None
                next_branch = None

                if target[axis] < root.point[axis]:
                    next_branch = root.left
                else:
                    next_branch = root.right

                next_best = closest_point(next_branch, target, depth + 1, next_best)

                if best is None or distance(target, next_best) < distance(target, best):
                    best = next_best

                if abs(target[axis] - root.point[axis]) < distance(target, best):
                    next_branch = root.right if next_branch == root.left else root.left
                    next_best = closest_point(next_branch, target, depth + 1, next_best)

                    if best is None or distance(target, next_best) < distance(target, best):
                        best = next_best

                return best

def distance(point1, point2):
                return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

def query_kd_tree(root, surname_range, awards_threshold, dblp_range):
   
    results = []

  
    query_result = query_kd_tree_recursive(root, surname_range, awards_threshold, dblp_range, results)

    return query_result

def query_kd_tree_recursive(node, surname_range, awards_threshold, dblp_range, results, depth=0):
    if node is not None:
        k = len(node.point) - 1  
        axis = depth % k

        if surname_range[0] <= node.point[0] <= surname_range[1] and \
           node.point[1] > awards_threshold and \
           dblp_range[0] <= node.point[2] <= dblp_range[1]:
            results.append(node.point)

        
        next_branch = None
        if surname_range[0] <= node.point[0] <= surname_range[1]:
            next_branch = node.left if node.point[axis] > surname_range[1] else node.right
        else:
            next_branch = node.right

       
        query_kd_tree_recursive(next_branch, surname_range, awards_threshold, dblp_range, results, depth + 1)

    return results


#Αντίστοιχη δομή στο RangeTree
class RangeTreeNode:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_range_tree(points, depth=0):
    if not points:
        return None

    k = len(points[0]) - 1  # Αφαιρούμε το "Surname" που δεν αποτελεί διάσταση

    axis = depth % k
    sorted_points = sorted(points, key=lambda x: x[axis])

    median = len(sorted_points) // 2
    return RangeTreeNode(
        point=sorted_points[median],
        left=build_range_tree(sorted_points[:median], depth + 1),
        right=build_range_tree(sorted_points[median + 1:], depth + 1)
    )

def query_range_tree(root, surname_range, awards_threshold, dblp_range):
    results = []

  
    query_range_tree_recursive(root, surname_range, awards_threshold, dblp_range, results)

    return results

def query_range_tree_recursive(node, surname_range, awards_threshold, dblp_range, results, depth=0):
    if node is not None:
        k = len(node.point) - 1  
        axis = depth % k

        
        if surname_range[0] <= node.point[0] <= surname_range[1] and \
           node.point[1] > awards_threshold and \
           dblp_range[0] <= node.point[2] <= dblp_range[1]:
            results.append(node.point)

       
        next_branch = None
        if surname_range[0] <= node.point[0] <= surname_range[1]:
            next_branch = node.left if node.point[axis] > surname_range[1] else node.right
        else:
            next_branch = node.right

      
        query_range_tree_recursive(next_branch, surname_range, awards_threshold, dblp_range, results, depth + 1)

    return results



# Η δομή του RTreeNode

class RTreeNode:
    def __init__(self, points=None, children=None, is_leaf=True):
        self.points = points or []
        self.children = children or []
        self.is_leaf = is_leaf

def build_rtree(points, max_children=4, depth=0):
    if len(points) <= max_children:
        return RTreeNode(points=points, is_leaf=True)

    axis = depth % len(points[0])
    points.sort(key=lambda x: x[axis])

    median = len(points) // 2

    left_child = build_rtree(points[:median], max_children, depth + 1)
    right_child = build_rtree(points[median:], max_children, depth + 1)

    return RTreeNode(children=[left_child, right_child], is_leaf=False)

def query_rtree(root, surname_range, awards_threshold, dblp_range):
    results = []
    query_rtree_recursive(root, surname_range, awards_threshold, dblp_range, results)
    return results

def query_rtree_recursive(node, surname_range, awards_threshold, dblp_range, results):
    if node.is_leaf:
        for point in node.points:
            if surname_range[0] <= point[0] <= surname_range[1] and \
               point[1] > awards_threshold and \
               dblp_range[0] <= point[2] <= dblp_range[1]:
                results.append(point)
    else:
        for child in node.children:
            if surname_range[1] < child.points[0][0] or \
               surname_range[0] > child.points[-1][0]:
                continue
            query_rtree_recursive(child, surname_range, awards_threshold, dblp_range, results)



#Η δομή QuadTree

class QuadTreeNode:
    def __init__(self, bounding_box, points=None, children=None):
        self.bounding_box = bounding_box
        self.points = points if points else []
        self.children = children if children else []

def build_quadtree(points, depth=0, max_items_per_node=4):
    if not points:
        return None

    k = len(points[0]) - 1  # Αφαιρούμε το "Surname" που δεν αποτελεί διάσταση

    axis = depth % k
    sorted_points = sorted(points, key=lambda x: x[axis])

    median = len(sorted_points) // 2
    bounding_box = calculate_bounding_box(sorted_points)

    if len(sorted_points) <= max_items_per_node:
        return QuadTreeNode(bounding_box, points=sorted_points)

    left_child = build_quadtree(sorted_points[:median], depth + 1, max_items_per_node)
    right_child = build_quadtree(sorted_points[median:], depth + 1, max_items_per_node)

    return QuadTreeNode(bounding_box, children=[left_child, right_child])

def calculate_bounding_box(points):
    min_values = [min(p[i] for p in points) for i in range(len(points[0]))]
    max_values = [max(p[i] for p in points) for i in range(len(points[0]))]
    return list(zip(min_values, max_values))

def query_quadtree(root, surname_range, awards_threshold, dblp_range):
    results = []
    query_quadtree_recursive(root, surname_range, awards_threshold, dblp_range, results)

    return results

def query_quadtree_recursive(node, surname_range, awards_threshold, dblp_range, results):
    if node is not None:
        k = len(node.bounding_box) // 2  

       
        if all(min_val <= max_val for min_val, max_val in zip(surname_range, node.bounding_box[0])) and \
           all(min_val <= max_val for min_val, max_val in zip(node.bounding_box[1], surname_range)) and \
           node.points:
           
            for point in node.points:
                if point[0] >= surname_range[0] and point[0] <= surname_range[1] and \
                   point[1] > awards_threshold and \
                   point[2] >= dblp_range[0] and point[2] <= dblp_range[1]:
                    results.append(point)

      
        for child in node.children:
            query_quadtree_recursive(child, surname_range, awards_threshold, dblp_range, results)


#o algorithmos LSH για κείμενα
class LSH:
    def __init__(self, num_hashes, num_buckets):
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.hashes = [self._generate_hash_function() for _ in range(num_hashes)]
        self.buckets = {i: set() for i in range(num_buckets)}

    def _generate_hash_function(self):
        a = np.random.randint(low=1, high=self.num_buckets)
        b = np.random.randint(low=0, high=self.num_buckets)
        return lambda x: (a * hash(x) + b) % self.num_buckets

    def _hash(self, vector):
        return [hash_function(vector) for hash_function in self.hashes]

    def index(self, text, document_id):
        hashed_vector = self._hash(text)
        for i, bucket_index in enumerate(hashed_vector):
            self.buckets[bucket_index].add((document_id, i))

    def query(self, text):
        hashed_vector = self._hash(text)
        candidate_set = set()
        for bucket_index in hashed_vector:
            candidate_set.update(self.buckets[bucket_index])

        return candidate_set

class LSH:
    def __init__(self, num_hashes, num_buckets):
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.hashes = [self._generate_hash_function() for _ in range(num_hashes)]
        self.buckets = {i: set() for i in range(num_buckets)}

    def _generate_hash_function(self):
        a = np.random.randint(low=1, high=self.num_buckets)
        b = np.random.randint(low=0, high=self.num_buckets)
        return lambda x: (a * hash(x) + b) % self.num_buckets

    def _hash(self, vector):
        return [hash_function(vector) for hash_function in self.hashes]

    def index(self, text, document_id):
        hashed_vector = self._hash(text)
        for i, bucket_index in enumerate(hashed_vector):
            self.buckets[bucket_index].add((document_id, i))

    def query(self, text):
        hashed_vector = self._hash(text)
        candidate_set = set()
        for bucket_index in hashed_vector:
            candidate_set.update(self.buckets[bucket_index])

        return candidate_set


# Κυρίως πρόγραμμα

# διάβασμα του CSV
df=pd.read_csv("coffee_analysis.csv")
selected_fields = ['name','roaster', 'roast']
points=df[selected_fields].values.tolist()
root = build_kdtree(points)

documents=df["loc_country"].values.tolist()

# Καθορισμός των κριτηρίων ερωτήματος
surname_range = ['A', 'D']
awards_threshold = 10000
dblp_range = [0, 40000]

# Εκτέλεση του ερωτήματος
start = time.time()

#ektelesi tou LSH


query_results = query_kd_tree(root, surname_range, awards_threshold, dblp_range)
#query_results = query_range_tree(root, surname_range, awards_threshold, dblp_range)
#query_results = query_rtree(root, surname_range, awards_threshold, dblp_range)
#query_results = query_quadtree(root, surname_range, awards_threshold, dblp_range)

#ektelesi tou lsh
lsh = LSH(num_hashes=5, num_buckets=5)
for i, doc in enumerate(documents):
    lsh.index(doc, i)
    
for result in query_results:
    print(result)

end = time.time()
# Χρόνος αποτελεσμάτων
print("Time:",end - start)









