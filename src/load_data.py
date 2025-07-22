import torch
from utils import osUtils as ou


def readKGData(data_dir='../data_set/OS/'):
    data_type = 'kg_final2.txt'
    path = data_dir + data_type
    print('Read knowledge graph data...')
    entity_set = set()
    # relation_set = set()
    triples = [[], []]
    for h, r, t in ou.readTriple(path, sep=','):
        entity_set.add(int(h))
        entity_set.add(int(t))
        # relation_set.add(int(r))
        # triples.append([int(h), int(r), int(t)])
        triples[0].append(int(h))
        triples[1].append(int(t))
    # return list(entity_set), triples
    return torch.Tensor(list(entity_set)), torch.tensor(triples)


def readRecData(data_dir='../data_set/OS/'):
    data_type = 'comb_final.txt'
    path = data_dir + data_type
    print('Read Drug Combination Synergy Data...')
    # drug_set1, drug_set2, cell_set = set(), set(), set()
    triples = []
    for d1, d2, c, s, flod in ou.readTriple(path, sep=','):
        # drug_set1.add(int(d1))
        # drug_set2.add(int(d2))
        # cell_set.add(int(i))
        triples.append((int(d1), int(d2), int(c), int(s), int(flod)))

    return triples
