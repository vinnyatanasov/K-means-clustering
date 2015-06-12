# 
# Takes care of creating feature space, creating feature vectors and plotting graphs
# works with the K-means clustering algorithm
#
# Vincent Atanasov, m4va
#

import numpy
import matplotlib.pyplot as plot


# creates feature space and returns dict of features
def process_features(file_path):
    feats = set()
    with open(file_path) as F:
        for line in F:
            for word in line.strip().split()[1:]:
                feats.add(word)
    
    feats_index = {}
    for (fid, fval) in enumerate(list(feats)):
        feats_index[fval] = fid
    
    return feats_index


# makes vectors from data - we ignore the first element (as it's the label) when creating vector
# then add the label to create a tuple in the format (vector, label)
def make_vects(file_path, feats, feat_size):
    data = []
    with open(file_path) as F:
        for line in F:
            vect = numpy.zeros(feat_size)
            line = line.strip().split()
            label = line[0]
            for word in line[1:]:
                vect[feats[word]] = 1
            data.append((vect, label))
    
    return data


def plot_results(values, label):
    # x axis
    xaxis = xrange(2, 21, 1)
    plot.xlabel("K value")
    plot.xticks([x for x in xaxis])
    
    # y axis
    plot.ylabel("Measure")
    plot.ylim([min(values)-0.15, max(values)+0.15])
    
    plot.plot(xaxis, values, label=label, linewidth=2)
    plot.legend(loc=2)
    plot.grid(True)
    plot.show()
