import math

deg_to_rad = math.pi / 180.0

def haversine_distance(lat1, lon1, lat2, lon2):
    dlon = abs(lon1 - lon2)
    dlat = abs(lat1 - lat2)
    
    a = math.pow(math.sin(dlat/2 * deg_to_rad), 2) + \
        math.cos(lat1 * deg_to_rad) * math.cos(lat2 * deg_to_rad) * \
        math.pow(math.sin(dlon/2 * deg_to_rad), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = 6371.0 * c
    return round(d)

def attractiveness(distance, quality):
    return float(quality) / (distance + 1.0)

def to_percent(utility, total):
    if total == 0:
        return 0.0
    return utility / total * 100.0

class Problem:
    def __init__(self):
        self.demands = []
        self.J = []
        self.QJ = []
        self.L = []
        self.QL = []
        self.QJByLoc = {}
        self.QLByLoc = {}
        self.DM = []
    
    def build_distance_matrix(self):
        n = len(self.demands)
        self.DM = []
        for i in range(n):
            self.DM.append([0.0] * (i + 1))
            for j in range(i + 1):
                self.DM[i][j] = haversine_distance(
                    self.demands[i].lat, self.demands[i].lon,
                    self.demands[j].lat, self.demands[j].lon
                )
    
    def distance(self, i, j):
        if i >= j:
            return self.DM[i][j]
        return self.DM[j][i]

class DemandPoint:
    def __init__(self, lat, lon, weight):
        self.lat = lat
        self.lon = lon
        self.weight = weight

def load_problem(problem_file, demands_file):
    p = Problem()
    load_facilities(p, problem_file)
    load_demands(p, demands_file)
    p.build_distance_matrix()
    return p

def load_facilities(p, filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    begin_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "-----BEGIN-----":
            begin_idx = i
            break
    
    if begin_idx is None:
        raise ValueError("-----BEGIN----- not found")
    
    tokens = []
    for line in lines[begin_idx + 1:]:
        tokens.extend(line.strip().split())
    
    idx = 0
    n_j = int(tokens[idx]); idx += 1
    p.J = []
    p.QJ = []
    for _ in range(n_j):
        p.J.append(int(tokens[idx])); idx += 1
        p.QJ.append(float(tokens[idx])); idx += 1
    
    n_l = int(tokens[idx]); idx += 1
    p.L = []
    p.QL = []
    for _ in range(n_l):
        p.L.append(int(tokens[idx])); idx += 1
        p.QL.append(float(tokens[idx])); idx += 1
    
    p.QJByLoc = {}
    for i, loc in enumerate(p.J):
        p.QJByLoc[loc] = p.QJ[i]
    p.QLByLoc = {}
    for i, loc in enumerate(p.L):
        p.QLByLoc[loc] = p.QL[i]

def load_demands(p, filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    tokens = []
    for line in lines:
        tokens.extend(line.strip().split())
    
    idx = 0
    n = int(tokens[idx]); idx += 1
    p.demands = []
    for _ in range(n):
        lat = float(tokens[idx]); idx += 1
        lon = float(tokens[idx]); idx += 1
        weight = float(tokens[idx]); idx += 1
        p.demands.append(DemandPoint(lat, lon, weight))

def huff_utility(p, X):
    total = 0.0
    utility = 0.0
    
    for i, dp in enumerate(p.demands):
        total += dp.weight
        
        attrJ = 0.0
        for jIdx, jLoc in enumerate(p.J):
            attrJ += attractiveness(p.distance(i, jLoc), p.QJ[jIdx])
        
        attrX = 0.0
        for xLoc in X:
            attrX += attractiveness(p.distance(i, xLoc), p.QLByLoc[xLoc])
        
        denom = attrJ + attrX
        if denom > 0:
            utility += dp.weight * attrX / denom
    
    return to_percent(utility, total)

def partially_binary_utility(p, X):
    total = 0.0
    utility = 0.0
    
    for i, dp in enumerate(p.demands):
        total += dp.weight
        
        bestJ = 0.0
        for jIdx, jLoc in enumerate(p.J):
            attr = attractiveness(p.distance(i, jLoc), p.QJ[jIdx])
            if attr > bestJ:
                bestJ = attr
        
        bestX = 0.0
        for xLoc in X:
            attr = attractiveness(p.distance(i, xLoc), p.QLByLoc[xLoc])
            if attr > bestX:
                bestX = attr
        
        denom = bestJ + bestX
        if denom > 0:
            utility += dp.weight * bestX / denom
    
    return to_percent(utility, total)

def binary_utility(p, X):
    total = 0.0
    utility = 0.0
    
    for i, dp in enumerate(p.demands):
        total += dp.weight
        
        bestJ = -1.0
        for jIdx, jLoc in enumerate(p.J):
            attr = attractiveness(p.distance(i, jLoc), p.QJ[jIdx])
            if attr > bestJ:
                bestJ = attr
        
        bestX = -1.0
        for xLoc in X:
            attr = attractiveness(p.distance(i, xLoc), p.QLByLoc[xLoc])
            if attr > bestX:
                bestX = attr
        
        if bestX > bestJ:
            utility += dp.weight
        elif bestX == bestJ:
            utility += dp.weight / 2.0
    
    return to_percent(utility, total)

def pareto_huff_utility(p, X):
    total = 0.0
    utility = 0.0
    
    for i, dp in enumerate(p.demands):
        total += dp.weight
        
        facilities = []
        for jIdx, jLoc in enumerate(p.J):
            d = p.distance(i, jLoc)
            facilities.append({
                'distance': d,
                'quality': p.QJ[jIdx],
                'attr': attractiveness(d, p.QJ[jIdx]),
                'ours': False
            })
        for xLoc in X:
            d = p.distance(i, xLoc)
            facilities.append({
                'distance': d,
                'quality': p.QLByLoc[xLoc],
                'attr': attractiveness(d, p.QLByLoc[xLoc]),
                'ours': True
            })
        
        pareto_mask = [True] * len(facilities)
        
        for a in range(len(facilities)):
            for b in range(len(facilities)):
                if a == b:
                    continue
                dominated = (facilities[b]['distance'] <= facilities[a]['distance'] and
                           facilities[b]['quality'] >= facilities[a]['quality'] and
                           (facilities[b]['distance'] < facilities[a]['distance'] or
                            facilities[b]['quality'] > facilities[a]['quality']))
                if dominated:
                    pareto_mask[a] = False
                    break
        
        total_attr = 0.0
        our_attr = 0.0
        for idx, keep in enumerate(pareto_mask):
            if not keep:
                continue
            total_attr += facilities[idx]['attr']
            if facilities[idx]['ours']:
                our_attr += facilities[idx]['attr']
        
        if total_attr > 0:
            utility += dp.weight * our_attr / total_attr
    
    return to_percent(utility, total)

if __name__ == "__main__":
    p = load_problem("../CFLP.dat", "../demands.dat")
    
    facility_set_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    facility_set_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 49]
    
    print("Evaluating facility set {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}:")
    print(f"  HuffModel: {huff_utility(p, facility_set_1):.6f}%")
    print(f"  PartiallyBinaryModel: {partially_binary_utility(p, facility_set_1):.6f}%")
    print(f"  BinaryModel: {binary_utility(p, facility_set_1):.6f}%")
    print(f"  ParetoHuffModel: {pareto_huff_utility(p, facility_set_1):.6f}%")
    
    print("\nEvaluating facility set {0, 1, 2, 3, 4, 5, 6, 7, 8, 49}:")
    print(f"  HuffModel: {huff_utility(p, facility_set_2):.6f}%")
    print(f"  PartiallyBinaryModel: {partially_binary_utility(p, facility_set_2):.6f}%")
    print(f"  BinaryModel: {binary_utility(p, facility_set_2):.6f}%")
    print(f"  ParetoHuffModel: {pareto_huff_utility(p, facility_set_2):.6f}%")
