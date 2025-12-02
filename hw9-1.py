#! /usr/bin/env python3

import argparse
import csv
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# compute euclidean, manhattan, jaccard, cosine distance matrices
class DistanceMatrices:

    def __init__(self,sequence,kmer_size):
        self.sequence = sequence
        self.size = kmer_size

    # parses FASTA file into k-mers and records count per k-mer
    def parse(self):
        kmer_counts = {}

        try:
            # opens FASTA file to read, errors out if not in FASTA format
            with open(self.sequence,"r") as file:
                if file.readline()[0] != ">":
                    raise ValueError("must enter '.fasta' format file")

                # header lines are skipped
                for line in file:
                    if line.startswith(">"):
                        continue

                    # parses sequence into k-mers based on specified k-mer size
                    for index, nucleotide in enumerate(line):
                        kmer = line.rstrip()[index:index+self.size]

                        # determine end of sequence if k-mer length is < k-mer size
                        if len(kmer) < self.size:
                            break

                        # skips kmers with non-nucleotide values; adds valid k-mers and their counts to dictionary
                        nuc_only = True
                        for nuc in kmer:
                            if nuc not in "AGCT":
                                nuc_only = False
                                break
                        if nuc_only == True:
                            kmer_counts.setdefault(kmer,0)
                            kmer_counts[kmer] += 1

            return kmer_counts
        except:
            raise IOError("Could not open FASTA file")

    # normalizes each k-mer count, calls parse()
    def normalize(self):
        dict = self.parse()
        counts_list = []

        # appends each k-mer count from parse() into list
        for count in dict.values():
            counts_list.append(count)

        # adds k-mer and normalized frequency (k-mer count divided by sum of all k-mers) to new dictionary
        normalized_counts = {}
        for kmer in dict.keys():
            normalized_counts[kmer] = dict[kmer] / sum(counts_list)

        return normalized_counts

    # computes differences between all k-mers of two FASTA file sequences, calls normalize()
    def x_minus_y(self,other):
        seq_one = self.normalize()
        seq_two = other.normalize()
        set_one = set(seq_one.keys())
        set_two = set(seq_two.keys())
        differences = []

        # appends differences between all k-mers to list (k-mers in both sets, k-mers unique to set 1 and set 2)
        for kmer in set_one.intersection(set_two):
            differences.append(seq_one[kmer] - seq_two[kmer])
        for kmer in set_one.difference(set_two):
            differences.append(seq_one[kmer] - 0)
        for kmer in set_two.difference(set_one):
            differences.append(0 - seq_two[kmer])

        return differences

    # computes euclidean distance matrix between two sequences, calls x_minus_y()
    def euclidean(self,other):
        euclidean = []

        # return euclidean value: square root of the sum of all differences squared
        for number in self.x_minus_y(other):
            euclidean.append(number**2)
        return math.sqrt(sum(euclidean))

    # computes manhattan distance matrix between two sequences, calls x_minus_y()
    def manhattan(self,other):
        manhattan = []

        # return manhattan value: sum of the absolute values of all differences
        for number in self.x_minus_y(other):
            manhattan.append(abs(number))
        return sum(manhattan)

    # computes cosine distance matrix between two sequences, calls parse()
    def cosine(self,other):
        seq1 = self.parse()
        seq2 = other.parse()
        set_one = set(seq1.keys())
        set_two = set(seq2.keys())
        s1_values = []
        s2_values = []
        products = []

        # sets numerator as the sum of the product of each k-mer count pair to list
        for item in set_one.intersection(set_two):
            products.append(seq1[item] * seq2[item])
        numerator = sum(products)

        # sets x denominator as the square root of the sum of all sequence 1 k-mer counts squared
        for num in seq1.values():
            s1_values.append(num**2)
        x_denominator = math.sqrt(sum(s1_values))

        # sets y denominator as the square root of the sum of all sequence 2 k-mer counts square
        for num in seq2.values():
           s2_values.append(num**2)
        y_denominator = math.sqrt(sum(s2_values))

        # returns cosine dissimilarity : 1 minus the calculated numerator divided by the product of the x and y denominator
        denominator = x_denominator * y_denominator
        cosine_dissimilarity = 1 - (numerator / denominator)

        # sets cosine dissimilarity values to 0 if very close to 0
        if math.isclose(cosine_dissimilarity,0.0,abs_tol=1e-09) :
            cosine_dissimilarity = 0
        return cosine_dissimilarity

    # computes jaccard dissimilarity index value based on prescence/absence of k-mers
    def jaccard(self,other):
        set_one = set(self.parse().keys())
        set_two = set(other.parse().keys())

        # returns jaccard dissimilarity index: 1 minus the intersection of the two sequences' k-mers divided by the union
        jaccard_result = (len(set_one.intersection(set_two)))/len((set_one.union(set_two)))
        return 1 - jaccard_result

# returns 2 required arguments of k-mer size and FASTA file path, and True/False argument for override
def get_args():
    parser = argparse.ArgumentParser("instructions for use")
    parser.add_argument("--size","-s",required=True,type=int,help="specify kmer size for parsing")
    parser.add_argument("--path","-p",required=True,nargs="+",help="path to FASTA file for kmer comparison")
    parser.add_argument("--override","-o",action="store_true",help="view raw distance matrix calculations if > 3 files")
    return parser.parse_args()

# takes filename, write or append instructions and line to write as parameters
def write_to_table(filename,w_or_a,line):
    try:
        with open(filename,w_or_a,newline="") as handle:
            writer = csv.writer(handle,delimiter="\t")
            writer.writerow(line)
    except:
        return "Error: could not write/append data to file"

# creates TSV file with all k-mer counts, calls parse()
def count_table(seq_names):
    all_parse = []

    # creates list with all sequences' returned dictionaries from parse()
    for iteration in range(len(seq_names)):
        all_parse.append(instances["seq"+str(iteration)].parse())
    all_kmers = set()

    # sets row 1 header as "sample" and all k-mers in sorted order and writes to count_table.tsv file by calling write_to_table()
    for inst in all_parse:
        all_kmers = all_kmers.union(inst.keys())
    header =["sample"] + sorted(all_kmers)
    write_to_table("count_table.tsv","w",header)

    # appends each FASTA files k-mer counts to seperate sequence specific list
    for idx,file in enumerate(all_parse) :
        counts = []
        for kmer in sorted(all_kmers):
            if kmer not in file.keys():
                counts.append(0)
            else:
                counts.append(file[kmer])

        # appends each sequence name and their sorted counts to the corresponding kmers to count_table.tsv file
        line = [seq_names[idx]] + counts
        write_to_table("count_table.tsv","a",line)
    return "kmer count table saved to directory for PCA use"

# creates TSV file with all distance matrices for each possible combination of sequences
def dm_output(seq_names):

    # writes first row/header to dm_output.tsv using write_to_table()
    header = ["sample files","euclidean","manhattan","cosine","jaccard"]
    write_to_table("dm_output.tsv","w",header)
    pairs = []
    sample_pairs = []

    # appends all combinations of instances to a list and all possible pairs of sequence names to a list
    for first in range(len(seq_names)):
        for second in range(first+1,len(seq_names)):
            pairs.append([instances["seq"+str(first)],instances["seq"+str(second)]])
            sample_pairs.append([seq_names[first],seq_names[second]])

    # for each pair of sequences, appends line to dm_output.tsv file with both sequence names and their distance matrix comparisons
    for idx,pair in enumerate(pairs):
        samples = sample_pairs[idx][0].split(".")[0] + " x " + sample_pairs[idx][1].split(".")[0]
        line = [samples] + [pair[0].euclidean(pair[1])] + [pair[0].manhattan(pair[1])] + [pair[0].cosine(pair[1])] + [pair[0].jaccard(pair[1])]
        write_to_table("dm_output.tsv","a",line)
    return "distance matrix table saved to directory"

# creates pairwise comparison table
def pairwise(seq_names):
    cosine_results = {}
    compare = []
    inst_list = []

    # appends all instances of sequences to list and all numerical combinations/pairs to a list
    for iteration in range(len(seq_names)):
        inst_list.append(instances["seq"+str(iteration)])
        compare.append(str(iteration))

    # creates dictionary with key as numerical combination names and value as the cosine value
    for first in range(len(seq_names)):
        for second in range(len(seq_names)):
            result = inst_list[first].cosine(inst_list[second])
            cosine_results[compare[first]+compare[second]] = result

    data = {}

    # creates list with all cosine matrix values in pairwise comparison table order, in row format
    for first in range(len(seq_names)) :
        cosine_values = []
        for second in range(len(seq_names)) :
            key = compare[first] + compare[second]
            cosine_values.append(cosine_results[key])

    # uses pandas module to create dataframe with sequence names as row indexers
        data[seq_names[first]] = cosine_values
    df = pd.DataFrame(data)
    df.index = seq_names

    # prints pandas dataframe to tsv file and returns the table to terminal
    df.to_csv("pairwise_compare.tsv",sep="\t")
    print("pairwise comparison table saved to tsv file in directory")
    return(df)

# creates heatmap utilizing pairwise() table, via seaborn module
def heatmap(seq_names):
    sns.heatmap(pairwise(seq_names))

    # saves heatmap as png in tight layout w/ more axes space format
    plt.tight_layout()
    plt.savefig("heatmap.png")
    return "heatmap saved to directory"

# creates PCA plot using pandas module and sci-kit learn module
def pca(seq_names):
    # use pandas module to read TSV k-mer count table file and sets PCA indexes
    print(count_table(seq_names))
    df = pd.read_csv("count_table.tsv",sep="\t",index_col=0)
    pca = PCA(n_components=2)

    # computes PCA values for k-mer counts and adds values back into dataframe table
    pca_result = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_result, index=df.index,columns=["PC1","PC2"])

    # creates scatter plot with pca dataframe as plotted values, adds data points to table, plot title, and x/y labels
    # with the PCA variance ratios
    fig, ax = plt.subplots()
    ax.scatter(pca_df["PC1"],pca_df["PC2"])
    for idx,sample in enumerate(pca_df.index):
        x = pca_df.loc[sample,"PC1"]
        y = pca_df.loc[sample,"PC2"]
        ax.scatter(x,y)
        ax.text(x,y,seq_names[idx])
    ax.set_title("PCA of K-mer Count Table")
    ax.set_xlabel("PC1 "+str(pca.explained_variance_ratio_[0]*100))
    ax.set_ylabel("PC2 "+str(pca.explained_variance_ratio_[1]*100))

    # saves PCA plot to png file in tight layout with more axes space in image
    plt.tight_layout()
    plt.savefig("pca.png")
    return "PCA saved to directory as png"

args = get_args()
instances = {}

# creates dictionary with key as the numbered sequence (seq0-seqx) and the value as the instance of 
# DistanceMatrices class of that sequence
for iteration in range(len(args.path)):
    inst_name = "seq" + str(iteration)
    instances[inst_name] = DistanceMatrices(args.path[iteration],args.size)

# runs dm_output() with file paths as parameter if 2 FASTA files or override argument is given
if len(args.path) == 2 or args.override==True:
    print(dm_output(args.path))
# runs heatmap() and pca() with file paths as parameter is 3 of more FASTA files given
if len(args.path) >= 3:
    print(heatmap(args.path),pca(args.path))
# if less than two file paths are given, error is raised
if len(args.path) < 2:
    raise ValueError("must provide at least two FASTA files")


