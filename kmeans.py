#
# K-means clustering algorithm
#
# Vincent Atanasov, m4va
#

import numpy, random, argparse
import data as d


# controls the program flow - runs for varying k values, or just one
def run(k_value, num_runs, medoid_mode, full_mode):
    # generate feature space and create data vectors
    feats = d.process_features("data/data.txt")
    data = d.make_vects("data/data.txt", feats, len(feats))
    
    # run the algorithm with varying values of k
    if (full_mode):
        r = [] # results
        for k in xrange(2, 21):
            print "k = {0}\n".format(k)
            r.append(kmeans(k, num_runs, data, medoid_mode))
        
        # plot results for each dimension of the results array
        l = ["Precision", "Recall", "F-score"]
        for i in xrange(3):
            values = []
            for row in r:
                values.append(row[i])
            d.plot_results(values, l[i])
        
    # or just run the algorithm with one k value
    else:
        print "k = {0}\n".format(k_value) 
        # kmeans returns results, but we don't need to do anything with them here
        r = kmeans(k_value, num_runs, data, medoid_mode)
    
    print "End.\n"


# runs the k-means algorithm with the supplied k value, number of iterations, data
# and a boolean indicating whether or not to use medoid mode in the selection of cluster centres
def kmeans(k, num_runs, data, medoid_mode):
    # store results to average
    r = []
    
    # run algorithm for desired number of iterations
    for i in xrange(num_runs):
        # generate initial centroids
        centroids = generate_centroids(k, data)
        
        # instantiate clusters dict in format {cluster_index: [data_index, data_index...]}
        clusters = {}
        for j in xrange(k):
            clusters[j] = []
        
        print "Run[{0}]".format(i)
        print "Clustering..."
        # run until convergence, with a maximum of 20 iterations
        for j in xrange(20):
            #print "Iteration #{0}".format(j)
            
            # count number of changes for to track convergence
            changes = 0
            
            # loop over data instances and assign each to some cluster based on distance measure
            for index, item in enumerate(data):
                # store all distances (to each centroid) - we then choose the smallest
                centroid_dists = []
                for centroid in centroids:
                    centroid_dists.append(compute_distance(item[0], centroid))
                
                # m is the index of the closest cluster centre
                m = centroid_dists.index(min(centroid_dists))
                
                # get closest cluster list
                cluster_list = clusters.get(m, [])
                
                # if item not already in list, then remove it from old one and add it to new one
                if (index not in cluster_list):
                    # on first iteration, we don't need to remove because they haven't been placed yet
                    if (j > 0):
                        # remove from old cluster
                        for cluster in clusters.itervalues():
                            if (index in cluster):
                                cluster.remove(index)
                                break
                    
                    # put in new cluster
                    cluster_list.append(index)
                    clusters[m] = cluster_list
                    
                    # increment changes counter
                    changes+=1
            
            # compute new cluster centres
            for key, value in clusters.iteritems():
                # compute the mean in cluster
                count = len(value)
                sum_vects = sum(data[x][0] for x in value)
                mean = sum_vects / count if count !=0 else 0
                
                # if medoid mode, we need to compute the nearest item to the mean
                if (medoid_mode):
                    tests = {}
                    # loop over each item in cluster and compute distance to mean
                    for item in value:
                        tests[item] = compute_distance(data[item][0], mean)
                    # here, m is the index of the closest data item to the mean
                    m = min(tests, key=tests.get)
                    # assign nearest item as new centroid
                    centroids[key] = data[m][0]
                # else, we just use the mean as the centroid
                else:
                    centroids[key] = mean
            
            #print "changes:", changes
            
            # if no points have changed cluster, we've converged on a minima, so exit loop
            if (changes == 0):
                print "Converged."
                break
        
        # evaluate clusters and append to results
        r.append(evaluate(k, data, clusters))
    
    return average_results(r)


# generates initial centroids using probabilities based on the smallest
# distance to pre-selected centroids - larger minimum gives higher probability of selection
def generate_centroids(k, data):
    centroids = []
    n = [] # just to hold the indices for reference
    
    # select initial point random and add to centroids array
    rn = random.randint(0, len(data)-1)
    n.append(rn)
    centroids.append(data[rn][0])
    
    # each item has array of distances to each centroid - so don't calculate them all on each iteration
    dists = {}
    # also store min distance (that's what we'll use for probabilities)
    min_dists = []
    
    # until we have k centroids
    i = 1
    while (i < k):
        for index, item in enumerate(data):
            # take minimum distance to a centroid and use to get probability
            # - if it has a small distance to one centroid, then we don't really want to use it...
            # the ones with the largest minimum are the ones we want to choose more often, which is achieved here
            # here we calculate the distance to the last centroid - all others have been calculated already
            d = compute_distance(item[0], centroids[-1])
            
            # add distance to correct list
            this_list = dists.get(index, [])
            this_list.append(d)
            dists[index] = this_list
            
            # update min to min_dists array - if index doesn't exist (on first iteration) then we append
            m = min(dists[index])
            try:
                min_dists[index] = m
            except:
                min_dists.append(m)
        
        # compute probabilities based on mimimum distances - larger distances = higher probabilities
        probs = numpy.array([x/sum(min_dists) for x in min_dists])
        cum_probs = probs.cumsum()
        
        # get a random float threshold [0,1] and find where we're over it
        # those with 0 distances (chosen already) can never get picked here
        rf = random.random()
        chosen_index = numpy.where(cum_probs >= rf)[0][0]
        
        # append new centroid to list
        centroids.append(data[chosen_index][0])
        n.append(chosen_index)
        i+=1
    
    return centroids


# computes distance between two vectors (x and y)
def compute_distance(x, y):
    # compute euclidean distance using normalised vectors
    normx = x / numpy.linalg.norm(x)
    normy = y / numpy.linalg.norm(y)
    dist = numpy.linalg.norm(normx - normy)
    return dist


# evaluates clusters
# first we count what is in each cluster, then we choose the highest count to be the cluster label
# then we merge any clusters that share same label, before calculating precision, recall and f-score
def evaluate(k, data, clusters):
    # dict to count what's in each cluster and label
    # e.g. {0: ['cluster_label', {label: count, label: count...}], ...}
    results = {}
    for i in xrange(k):
        results[i] = [None, {}]
    
    # dict to hold labels for each cluster - so we know which labels have > 1 cluster (to merge)
    labels = {}
    # loop through each cluster
    for key, value in clusters.iteritems():
        # loop over items in cluster
        for i in value:
            label = data[i][1]
            # add one to the corresponding counter in results dictionary
            results[key][1][label] = results[key][1].get(label, 0) + 1
        
        # count the label with the most occurances to use for the cluster label
        cluster_label = max(results[key][1], key=results[key][1].get)
        results[key][0] = cluster_label
        
        # add current cluster key to label list it belongs to
        temp_list = labels.get(cluster_label, [])
        temp_list.append(key)
        labels[cluster_label] = temp_list
    
    # merge same label clusters
    for j in labels.itervalues():
        # we only care about those with more than one in list
        if len(j) > 1:
            # if more than one cluster associated with list, then we merge (sum) them together
            # go through the indices in list that have same label (could be > 2, remember)
            n = {}
            for x in j:
                # get corresponding list from results dict
                a = results[x][1]
                # merge the lists
                n = {y: a.get(y, 0) + n.get(y, 0) for y in set(a).union(n)}
            
            # replace first old dict with new dict in the results and remove others completely
            count = 0;
            for x in j:
                # if first one, replace with new dictionary
                # else, just remove it
                if count == 0:
                    results[x][1] = n
                else:
                    del results[x]
                count += 1
    
    # return macros measures
    return calculate_measures(results)


# computes evaluation measures for each cluster and returns macro measures
def calculate_measures(results):
    precision_arr = []
    recall_arr = []
    fscore_arr = []
    for i, j in results.iteritems():
        # get label
        l = j[0]
        # get number associated with correct label
        correct = j[1][l]
        # count total in cluster
        total = sum(j[1].itervalues())
        # precision = number correct / total in cluster
        precision = correct/float(total)
        # recall = number correct / total number of possible correct (51)
        recall = correct/51.0
        # fscore = 2PR / P+R
        fscore = (2 * precision * recall) / (precision + recall)
        # append scores to respective arrays
        precision_arr.append(precision)
        recall_arr.append(recall)
        fscore_arr.append(fscore)
    
    # calculate macro averaged measures and print
    m_precision = sum(precision_arr)/float(len(precision_arr))
    m_recall = sum(recall_arr)/float(len(recall_arr))
    m_fscore = sum(fscore_arr)/float(len(fscore_arr))
    print "Macro precision:", m_precision
    print "Macro recall:", m_recall
    print "Macro F-score:", m_fscore, "\n"
    
    return (m_precision, m_recall, m_fscore)


# averages precision, recall and F-score
def average_results(r):
    # compute averages of results
    sum_precision = 0
    sum_recall = 0
    sum_fscore = 0
    for row in r:
        sum_precision += row[0]
        sum_recall += row[1]
        sum_fscore += row[2]
    avg_precision = sum_precision/len(r)
    avg_recall = sum_recall/len(r)
    avg_fscore = sum_fscore/len(r)
    
    # print averaged results
    print "Averaged results"
    print "Precision:", avg_precision
    print "Recall:", avg_recall
    print "F-score:", avg_fscore
    print "---------------------------\n"
    
    # return the averaged measures
    return (avg_precision, avg_recall, avg_fscore)


if __name__ == "__main__":
    # command line argument for k value, number of times to iterate and program modes
    parser = argparse.ArgumentParser(description="K-means", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--k_value", help="k is the number of clusters you want to initialise with", default=6, type=int)
    parser.add_argument("-n", "--num_runs", help="how many times to loop the program and take average results of", default=10, type=int)
    parser.add_argument("-m", "--medoid", help="include argument for medoid mode (data points as cluster centres) or ignore for default mode (means as cluster centres)", action='store_true')
    parser.add_argument("-f", "--full_mode", help="include argument to run program with varying values of k (from 2 to 20)", action='store_true')
    args = parser.parse_args()
    
    print "--K-MEANS CLUSTERING ALGORITHM--\n"
    
    # call main method with user preferences
    run(args.k_value, args.num_runs, args.medoid, args.full_mode)
    