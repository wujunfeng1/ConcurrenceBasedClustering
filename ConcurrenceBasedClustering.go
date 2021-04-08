package ConcurrenceBasedClustering

// =============================================================================
// Basic Concepts:
// 	This package is used for clustering of nodes based on concurrences of pairs
//	of nodes. In this package, there are quality models for evaluation of such
//	clusterings, as well cluster initializers and cluster optimizers.
// References:
//	[Shared Near Neighbors] Jarvis, R. A., & Patrick, E. A. (1973). Clustering
//		using a similarity measure based on shared near neighbors. IEEE
//		Transactions on computers, 100(11), 1025-1034.
//	[DBSCAN] Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996, August). A
//		density-based algorithm for discovering clusters in large spatial
//		databases with noise. In Kdd (Vol. 96, No. 34, pp. 226-231).
//	[ROCK] Guha, S., Rastogi, R., & Shim, K. (2000). ROCK: A robust clustering
//		algorithm for categorical attributes. Information systems, 25(5), 345-
//		366.
//	[Centric Local Outliers] Yu, J. X., Qian, W., Lu, H., & Zhou, A. (2006).
//		Finding centric local outliers in categorical/numerical spaces.
//		Knowledge and Information Systems, 9(3), 309-338.
//	[Louvain Algorithm & Modularity] Blondel, V. D., Guillaume, J. L., Lambiotte
//		, R., & Lefebvre, E. (2008). Fast unfolding of communities in large
//		networks. Journal of statistical mechanics: theory and experiment,
//		2008(10), P10008.
//	[Constant Potts Model] Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011).
//		Narrow scope for resolution-limit-free community detection. Physical
//		Review E, 84(1), 016114.
//	[Leiden Algorithm] Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From
//		Louvain to Leiden: guaranteeing well-connected communities. Scientific
//		reports, 9(1), 1-12.
//	[Label Propagation Algorithm] Zhu, X., & Ghahramani, Z. (2002). Learning
//		from labeled and unlabeled data with label propagation.
//	[Girvan Newman Algorithm] Girvan, M., & Newman, M. E. (2002). Community
//		structure in social and biological networks. Proceedings of the national
//		academy of sciences, 99(12), 7821-7826.
//	[Clique Percolation Method] Palla, G., Derényi, I., Farkas, I., & Vicsek, T.
//		(2005). Uncovering the overlapping community structure of complex
//		networks in nature and society. nature, 435(7043), 814-818.
//	[Advanced Clique Percolation Method] Salatino, A. A., Osborne, F., & Motta,
//		E. (2018, May). AUGUR: forecasting the emergence of new research topics.
//		In Proceedings of the 18th ACM/IEEE on Joint Conference on Digital
//		Libraries (pp. 303-312).
//	[Sequential Clique Percolation Method] Kumpula, J. M., Kivelä, M., Kaski, K.
//		, & Saramäki, J. (2008). Sequential algorithm for fast clique
//		percolation. Physical review E, 78(2), 026109.
//	[SLINK] Sibson, R. (1973). SLINK: an optimally efficient algorithm for the
//		single-link cluster method. The computer journal, 16(1), 30-34.
//	[CLINK] Defays, D. (1977). An efficient algorithm for a complete link method
//		. The Computer Journal, 20(4), 364-366.
//	[WPDM] Yang, S., Huang, G., & Ofoghi, B. (2019, May). Short Text Similarity
//		Measurement Using Context from Bag of Word Pairs and Word Co-occurrence.
//		In International Conference on Data Service (pp. 221-231). Springer,
//		Singapore.
// =============================================================================

import (
	"fmt"
	"log"

	//"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
	"time"
)

// =============================================================================
// func init
// brief description: init the package
func init() {
	rand.Seed(time.Now().UnixNano())
}

// =============================================================================
// struct ConcurrenceModel
// brief description: This is a struct for concurrence models
type ConcurrenceModel struct {
	// ------------------------------------------------------------------------
	// basic fields:
	n             int
	cardinalities []int
	concurrences  []map[int]float64

	// ------------------------------------------------------------------------
	// statistical fields
	sumConcurrences   float64
	sumConcurrencesOf []float64
}

// =============================================================================
// func NewConcurrenceModel
// brief description: create a new ConcurrenceModel object
func NewConcurrenceModel(neighbors [][]int, sims [][]float64, cardinalities []int) ConcurrenceModel {
	n := len(neighbors)
	if n != len(sims) || n != len(cardinalities) {
		log.Fatalln("input size don't match in NewConcurrenceModel")
	}
	concurrences := make([]map[int]float64, n)
	for i := 0; i < n; i++ {
		if len(neighbors[i]) != len(sims[i]) {
			log.Fatalln(fmt.Sprintf("len(neighbors[%d]) = %d != len(sims[%d]) = %d",
				i, len(neighbors[i]), i, len(sims[i])))
		}
		concurrences[i] = map[int]float64{}
		for j := 0; j < len(sims[i]); j++ {
			neighbor := neighbors[i][j]
			if neighbor == i {
				continue
			}
			concurrences[i][neighbor] = sims[i][j]
		}
	}
	sumConcurrencesOf := GetSumConcurrencesOf(concurrences, cardinalities)
	sumConcurrences := 0.0
	for i := 0; i < n; i++ {
		sumConcurrences += sumConcurrencesOf[i]
	}
	return ConcurrenceModel{
		n:                 n,
		concurrences:      concurrences,
		cardinalities:     cardinalities,
		sumConcurrences:   sumConcurrences,
		sumConcurrencesOf: sumConcurrencesOf,
	}
}

// =============================================================================
// func getSumConcurrencesOf
// brief description: Compute a vector that the i-th component of the vector is
//	the sum of concurrences connected to node i.
// input:
//	n: the number of nodes
//	concurrences: a matrix that its element (i,j) is the frequency of the
//		concurrence between node i and node j. If no such concurrence exists,
//		then the element is 0.
// output:
//	the vector mentioned in brief description.
func GetSumConcurrencesOf(concurrences []map[int]float64, cardinalities []int) []float64 {
	// -------------------------------------------------------------------------
	// step 1:
	n := len(concurrences)
	if n != len(cardinalities) {
		log.Fatalln("lengthes of concurrences and cardinalities don't match")
	}
	sumConcurrencesOf := make([]float64, n)
	for u := 0; u < n; u++ {
		mySum := 0.0
		weightsOfU := concurrences[u]
		for v, weightUV := range weightsOfU {
			mySum += weightUV * float64(cardinalities[u]*cardinalities[v])
		}
		sumConcurrencesOf[u] = mySum
	}

	// -------------------------------------------------------------------------
	// step 2: return the result
	return sumConcurrencesOf
}

// =============================================================================
// func (cm ConcurrenceModel) GetN
func (cm ConcurrenceModel) GetN() int {
	return cm.n
}

// =============================================================================
// func (cm ConcurrenceModel) GetConcurrencesOf
// brief description: get the concurrences related to a node
// input:
//	i: a point ID
// output:
//	the frequency of the concurrence of i if exists, 0 otherwise
func (cm ConcurrenceModel) GetConcurrencesOf(i int) map[int]float64 {
	return cm.concurrences[i]
}

// =============================================================================
// func (cm ConcurrenceModel) GetConcurrence
// brief description: get concurrence between i and j
// input:
//	i, j: two point IDs
// output:
//	the frequency of the concurrence between i and j if the edge exists, 0
//	otherwise
func (cm ConcurrenceModel) GetConcurrence(i, j int) float64 {
	weightIJ, exists := cm.concurrences[i][j]
	if exists {
		return weightIJ
	} else {
		return 0.0
	}
}

// =============================================================================
// func (cm ConcurrenceModel) Aggregate
// brief description: aggregates concurrences according to communities
// input:
//	communities: a list of clusters.
// output:
//	the aggregated ConcurrenceModel
func (cm ConcurrenceModel) Aggregate(communities []map[int]bool) ConcurrenceModel {
	// -------------------------------------------------------------------------
	// step 1: set newN and create an empty newConcurrences
	newN := len(communities)
	newConcurrences := make([]map[int]float64, newN)
	newCardinalities := make([]int, newN)
	for i := 0; i < newN; i++ {
		newConcurrences[i] = map[int]float64{}
		newCardinalities[i] = 1
	}

	// -------------------------------------------------------------------------
	// step 2: scans through the communities to fill newConcurrences
	for i1 := 0; i1+1 < newN; i1++ {
		c1 := communities[i1]
		for i2 := i1 + 1; i2 < newN; i2++ {
			c2 := communities[i2]
			weightI1I2 := 0.0
			for pt1, _ := range c1 {
				weightsOfPt1 := cm.concurrences[pt1]
				for pt2, _ := range c2 {
					weightPt1Pt2, exists := weightsOfPt1[pt2]
					if exists {
						weightI1I2 += weightPt1Pt2 *
							float64(cm.cardinalities[pt1]*cm.cardinalities[pt2])
					}
				}
			}
			if weightI1I2 > 0.0 {
				newConcurrences[i1][i2] = weightI1I2
				newConcurrences[i2][i1] = weightI1I2
			}
		}
	}

	// -------------------------------------------------------------------------
	// step 3: create a new ConcurrenceModel using these data
	newSumConcurrencesOf := GetSumConcurrencesOf(newConcurrences, newCardinalities)
	newSumConcurrences := 0.0
	for i := 0; i < cm.n; i++ {
		newSumConcurrences += newSumConcurrencesOf[i]
	}
	newCM := ConcurrenceModel{
		n:                 newN,
		concurrences:      newConcurrences,
		cardinalities:     newCardinalities,
		sumConcurrences:   newSumConcurrences,
		sumConcurrencesOf: newSumConcurrencesOf,
	}

	// -------------------------------------------------------------------------
	// step 4: return the new ConcurrenceModel
	return newCM
}

// =============================================================================
// func (cm ConcurrenceModel) connects
// brief description: check whether the concurrence graph connects two nodes.
// input:
//	u, v: two node IDs
// output:
//	true if it connects them, false otherwise
func (cm ConcurrenceModel) Connects(u, v int) bool {
	return cm.GetConcurrence(u, v) > 0.0
}

// =============================================================================
// func (cm ConcurrenceModel) connectsWell
// brief description: check whether the concurrence graph connects a subset well
//	from a set of node IDs.
// input:
//	subset: a subset of idSet
//	set: a set of node IDs
//	r: a threshold
// output:
//	true if it connects well, false otherwise
func (cm ConcurrenceModel) ConnectsWell(subset, set map[int]bool, r float64) bool {
	// -------------------------------------------------------------------------
	// step 1: find the complement of subset in set
	complement := map[int]bool{}
	for v, _ := range set {
		_, inSubset := subset[v]
		if inSubset {
			continue
		}
		complement[v] = true
	}

	// -------------------------------------------------------------------------
	// step 2: sum the weights between the subset and the complement
	x := 0.0
	for u, _ := range subset {
		weightsOfU := cm.concurrences[u]
		for v, _ := range complement {
			weightUV, exists := weightsOfU[v]
			if exists {
				x += weightUV * float64(cm.cardinalities[u]*cm.cardinalities[v])
			}
		}
	}

	// step 3: sum the cardinalities of the set and the subset
	card0 := 0
	for u, _ := range set {
		card0 += cm.cardinalities[u]
	}
	card1 := 0
	for u, _ := range subset {
		card1 += cm.cardinalities[u]
	}

	// -------------------------------------------------------------------------
	// step 4: return the result
	return x >= r*float64(card0*card1)
}

// =============================================================================
// interface QualityModel
// brief description: This is an interface for quality models
type QualityModel interface {
	// The first four methods are parts of ConcurrenceModel. Therefore, for
	// those structs merged with ConcurreneModel, they already have these four
	// methods
	GetN() int
	ConnectsWell(subset, set map[int]bool, r float64) bool
	Connects(u, v int) bool
	GetNeighbors(u int) map[int]float64

	// This method is simiar to that of ConcurrenceModel. The difference is the
	// return value.
	Aggregate(communities []map[int]bool) QualityModel

	// The last two methods are new to QualityModel. The implementations of this
	// interface must implement them.
	Quality(communities []map[int]bool) float64
	DeltaQuality(communities []map[int]bool, u, oldCu, newCu int) float64
}

// =============================================================================
// struct Modularity
// brief introduction: this is an implementation of the famous Modularity
// 	quality model for network clustering
type Modularity struct {
	r float64
	ConcurrenceModel
}

// =============================================================================
// func NewModularity
// brief description: create a new Modularity
// input:
//	r: a threshold of modularity
func NewModularity(r float64, cm ConcurrenceModel) Modularity {
	return Modularity{
		r:                r,
		ConcurrenceModel: cm,
	}
}

// =============================================================================
// func (qm *Modularity) Aggregate
func (qm Modularity) Aggregate(communities []map[int]bool) QualityModel {
	return QualityModel(Modularity{qm.r, qm.ConcurrenceModel.Aggregate(communities)})
}

// =============================================================================
// func (qm *Modularity) Quality
// brief description: this implements Quality for interface QualityModel
// input:
//	communities: a list of clusters.
// output:
//	the value of Modularity
func (qm Modularity) Quality(communities []map[int]bool) float64 {
	// -------------------------------------------------------------------------
	// step 1: compute 1/m and r/m
	oneOverM := 1.0 / float64(qm.sumConcurrences)
	rOverM := qm.r * oneOverM

	// -------------------------------------------------------------------------
	// step 2: compute modularity using the following equation:
	// modularity = 1/m sum_{i,j} (w_{i,j} - k_i * k_j * r/m) delta(c_i, c_j),
	// where:
	//	1/m = oneOverM,
	//	w_{i,j} = concurrence[i][j],
	//	k_u = nodewiseSumWeights[u],
	//	delta(s,t) = 0 if s != t, 1 if s == t.
	//	c_u = the community ID of u, i.e., communities[c][u] == true
	result := 0.0
	for _, c := range communities {
		for i, _ := range c {
			ki := qm.sumConcurrencesOf[i]
			for j, _ := range c {
				if i == j {
					continue
				}
				kj := qm.sumConcurrencesOf[j]
				result += qm.GetConcurrence(i, j)*float64(qm.cardinalities[i]*qm.cardinalities[j]) -
					rOverM*ki*kj
			}
		}
	}
	result *= oneOverM

	// -------------------------------------------------------------------------
	// step 3: return the result
	return result
}

// =============================================================================
// func (qm *Modularity) DeltaQuality
// brief description: this implements DeltaQuality for interface QualityModel
// input:
//	communities: a list of clusters.
//	u: a node ID, 0 <= u < n.
//	oldCu: the ID of the cluster u currently locates in.
//	newCu: the ID of the cluster u wants to move in.
// output:
//	The change amount of modularity.
// output:
//	the value of Modularity
func (qm Modularity) DeltaQuality(communities []map[int]bool, u, oldCu, newCu int) float64 {
	// -------------------------------------------------------------------------
	// step 1: check whether oldCu and newCu are the same one.
	// no change if oldCu == newCu
	if oldCu == newCu {
		return 0.0
	}

	// -------------------------------------------------------------------------
	// step 2: compute 1/m and r/m
	oneOverM := 1.0 / qm.sumConcurrences
	rOverM := qm.r * oneOverM

	// -------------------------------------------------------------------------
	// step 3: compute delta modularity. Note that:
	// modularity = 1/m sum_{i,j} (w_{i,j} - k_i * k_j * 1/m) delta(c_i, c_j),
	// where:
	//	1/m = oneOverM,
	//	w_{i,j} = concurrence[i][j],
	//	k_u = nodewiseSumWeights[u],
	//	delta(s,t) = 0 if s != t, 1 if s == t.
	//	c_u = the community ID of u, i.e., communities[c][u] == true
	// therfore:
	// delta modularity =
	//	1/m sum_{j in community newCu} (w_{u,j} - k_u * k_j * r/m)
	//	- 1/m sum_{j in community oldCu, j != i} (w_{u,j} - k_u * k_j * r/m)
	// (3.1) fetch weights of u and k_u
	weightsOfU := qm.GetConcurrencesOf(u)
	ku := qm.sumConcurrencesOf[u]

	// (3.2) add to result the change at the new community of u
	result := 0.0
	newCommunityOfU := communities[newCu]
	for j, _ := range newCommunityOfU {
		weightUJ, exists := weightsOfU[j]
		if !exists {
			weightUJ = 0.0
		}
		kj := qm.sumConcurrencesOf[j]
		result += weightUJ*float64(qm.cardinalities[u]*qm.cardinalities[j]) - rOverM*ku*kj
	}

	// (3.3) subtract from result the change at the old community of u
	oldCommunityOfU := communities[oldCu]
	for j, _ := range oldCommunityOfU {
		if j == u {
			continue
		}
		weightUJ, exists := weightsOfU[j]
		if !exists {
			weightUJ = 0.0
		}
		kj := qm.sumConcurrencesOf[j]
		result -= weightUJ*float64(qm.cardinalities[u]*qm.cardinalities[j]) - rOverM*ku*kj
	}
	result *= oneOverM

	// -------------------------------------------------------------------------
	// step 4: return the result
	return result
}

// =============================================================================
// struct CPM
// brief introduction: this is an implementation of the famous Constant Potts
// 	quality model for network clustering
type CPM struct {
	r float64
	ConcurrenceModel
}

func (qm CPM) GetNeighbors(u int) map[int]float64 {
	return qm.concurrences[u]
}

// =============================================================================
// func NewCPM
// brief description: create a new CPM
// input:
//	r: a threshold of CPM
func NewCPM(r float64, cm ConcurrenceModel) CPM {
	return CPM{
		r:                r,
		ConcurrenceModel: cm,
	}
}

func (qm Modularity) GetNeighbors(u int) map[int]float64 {
	return qm.concurrences[u]
}

// =============================================================================
// func (qm CPM) Aggregate
func (qm CPM) Aggregate(communities []map[int]bool) QualityModel {
	return QualityModel(CPM{qm.r, qm.ConcurrenceModel.Aggregate(communities)})
}

// =============================================================================
// func (qm *CPM) Quality
// brief description: this implements Quality for interface QualityModel
// input:
//	communities: a list of clusters.
// output:
//	the value of Modularity
func (qm CPM) Quality(communities []map[int]bool) float64 {
	// -------------------------------------------------------------------------
	// step 1: compute CPM using the following equation:
	// CPM = sum_c (w_c - r size_c^2),
	// where:
	//	c is a community,
	//	size_c is the number of nodes in c,
	//	w_c is the sum of weight(i,j) for all i, j in c.
	result := 0.0
	for _, c := range communities {
		sizeC := 0
		for i, _ := range c {
			sizeC += qm.cardinalities[i]
		}

		sumWeightsOfC := 0.0
		for i, _ := range c {
			weightsOfI := qm.GetConcurrencesOf(i)
			for j, _ := range c {
				weightIJ, exists := weightsOfI[j]
				if exists {
					sumWeightsOfC += weightIJ * float64(qm.cardinalities[i]*qm.cardinalities[j])
				}
			}
		}

		result += sumWeightsOfC - qm.r*float64(sizeC*sizeC)
	}

	// -------------------------------------------------------------------------
	// step 3: return the result
	return result
}

// =============================================================================
// func (qm *CPM) DeltaQuality
// brief description: this implements DeltaQuality for interface QualityModel
// input:
//	communities: a list of clusters.
//	u: a node ID, 0 <= u < n.
//	oldCu: the ID of the cluster u currently locates in.
//	newCu: the ID of the cluster u wants to move in.
// output:
//	The change amount of modularity.
// output:
//	the value of Modularity
func (qm CPM) DeltaQuality(communities []map[int]bool, u, oldCu, newCu int) float64 {
	// -------------------------------------------------------------------------
	// step 1: check whether oldCu and newCu are the same one.
	// no change if oldCu == newCu
	if oldCu == newCu {
		return 0.0
	}

	// -------------------------------------------------------------------------
	// step 2: compute delta CPM.
	// CPM = sum_c (w_c - r size_c^2),
	// where:
	//	c is a community,
	//	size_c is the number of nodes in c,
	//	w_c is the sum of weight(i,j) for all i, j in c.
	// Therefore:
	// delta CPM = delta w_oldCu + delta w_newCu
	//	- r ((size_oldCu-card)^2 - size_oldCu^2)
	//	- r ((size_newCu+card)^2 - size_newCu^2)
	//	= delta w_oldCu + delta w_newCu - r (-2 size_oldCu * card + card^2)
	//	- r (2 size_newCu * card + card^2)
	//	= delta w_oldCu + delta w_newCu - 2 r * card * (size_newCu - size_oldCu + card)

	// (2.1) fetch weights and card of u
	weightsOfU := qm.GetConcurrencesOf(u)
	cardU := qm.cardinalities[u]

	// (2.2) compute delta w_oldCu and sizeOldCu
	deltaWOldCu := 0.0
	sizeOldCu := 0
	oldCommunityOfU := communities[oldCu]
	for j, _ := range oldCommunityOfU {
		sizeOldCu += qm.cardinalities[j]
		if j == u {
			continue
		}
		weightUJ, exists := weightsOfU[j]
		if exists {
			deltaWOldCu -= weightUJ * float64(cardU*qm.cardinalities[j])
		}
	}

	// (2.3) compute delta w_newCu and sizeNewCu
	deltaWNewCu := 0.0
	sizeNewCu := 0
	newCommunityOfU := communities[newCu]
	for j, _ := range newCommunityOfU {
		sizeNewCu += qm.cardinalities[j]
		weightUJ, exists := weightsOfU[j]
		if exists {
			deltaWNewCu += weightUJ * float64(cardU*qm.cardinalities[j])
		}
	}

	// (2.4) compute the result
	result := deltaWOldCu + deltaWNewCu - 2.0*qm.r*float64(cardU*(sizeNewCu-sizeOldCu+cardU))

	// -------------------------------------------------------------------------
	// step 3: return the result
	return result
}

// =============================================================================
// func getCorePoints
// brief description: This is part of an implementation to the famous DBScan
//	algorithm: looking for all core points.
// input:
//	simMat: the similarity matrix. It must be symmetric, all elements 0~1, and
//		the diagonal elements are all 1.
//	eps: the radius of neighborhood.
//	minPts: Only if the neighborhood of a point contains at least minPt points
//		(the center point of the neighborhood included), the neighborhood is
//		called dense. Only dense neighborhoods are connected to communities.
// output:
//	A map of core points to their neighborhood densities.
func (cm ConcurrenceModel) getCorePoints(eps float64, minPts int) map[int]int {
	// -------------------------------------------------------------------------
	// step 1: compute the density of all points' neighborhoods
	n := cm.n
	densities := make([]int, n)
	for pt := 0; pt < n; pt++ {
		rowPt := cm.concurrences[pt]
		density := cm.cardinalities[pt]
		for neighbor, similarity := range rowPt {
			if similarity+eps >= 1.0 {
				density += cm.cardinalities[neighbor]
			}
		}
		densities[pt] = density
	}

	// -------------------------------------------------------------------------
	// step 2: generate a list of points with dense neighborhoods
	corePts := map[int]int{}
	for pt, density := range densities {
		if density >= minPts {
			corePts[pt] = density
		}
	}

	// -------------------------------------------------------------------------
	// step 3: return the result
	return corePts
}

// =============================================================================
// func getNeighbors
// brief description: This is part of an implementation to the famous DBScan
//	algorithm: generating a list of core members and another list of noncore
//	neighbors for each core points.
// input:
//	eps: the radius of neighborhood.
//	minPts: Only if the neighborhood of a point contains at least minPt points
//		(the center point of the neighborhood included), the neighborhood is
//		called dense. Only dense neighborhoods are connected to communities.
//	corePts: a map of core points to their neighborhood densities.
// output:
//	output 1: a list of the core neighbors for each core point.
//	output 2: a list of the noncore neighbors for each core point.
func (cm ConcurrenceModel) getNeighbors(eps float64, minPts int, corePts map[int]int) (
	coreNeighbors map[int]map[int]bool, noncoreNeighbors map[int]map[int]bool) {
	coreNeighbors = map[int]map[int]bool{}
	noncoreNeighbors = map[int]map[int]bool{}
	for pt, _ := range corePts {
		// create the rows of the results
		coreRow := map[int]bool{}
		coreNeighbors[pt] = coreRow
		noncoreRow := map[int]bool{}
		noncoreNeighbors[pt] = noncoreRow

		// read the row of similarity matrix
		simRow := cm.concurrences[pt]

		// scan through the row we just read
		for neighbor, similarity := range simRow {
			// skip pt itself
			if neighbor == pt {
				continue
			}
			// find points that locate within pt's neighborhood
			if similarity+eps >= 1.0 {
				_, isCorePoint := corePts[neighbor]
				if isCorePoint {
					coreRow[neighbor] = true
				} else {
					noncoreRow[neighbor] = true
				}
			}
		}
	}
	return
}

// =============================================================================
// func (cm ConcurrenceModel) DBScan
// brief description: This is an implementation to the famous DBScan algorithm.
// input:
//	eps: the radius of neighborhood.
//	minPts: Only if the neighborhood of a point contains at least minPt points
//		(the center point of the neighborhood included), the neighborhood is
//		called dense. Only dense neighborhoods are connected to communities.
//	simType: the type of similarity, 0 for simple induced similarity, 1 for normalized
//		similarity, 2 for jaccard similarity, 4 for weighted jaccard similarity, 4 for
//		normalized jaccard similarity
// output:
//	A list of clusters.
func (cm ConcurrenceModel) DBScan(eps float64, minPts int) ([]map[int]bool, []int) {
	// -------------------------------------------------------------------------
	// step 1: initialize auxiliary data structures
	communityIDs := make([]int, cm.n)
	communities := []map[int]bool{}
	for i := 0; i < cm.n; i++ {
		communityIDs[i] = -1
	}

	// -------------------------------------------------------------------------
	// step 3: find all core points and their neighborhood densities
	corePts := cm.getCorePoints(eps, minPts)

	// -------------------------------------------------------------------------
	// step 4: find neighbors for each core point
	coreNeighbors, noncoreNeighbors := cm.getNeighbors(eps, minPts, corePts)

	// -------------------------------------------------------------------------
	// step 5: loop until all core points are in communities
	n := cm.n
	for {
		// (5.1) prepare an ID for the new community
		c := len(communities)

		// (5.2) find the densist unassigned core point as the center point of
		// the new cluster
		centerPt := n
		centerDensity := 0
		for pt, density := range corePts {
			// skip those points that have already been assigned into community
			if communityIDs[pt] >= 0 {
				continue
			}

			// check whether with the currently most dense neighborhood
			if density > centerDensity {
				centerPt = pt
				centerDensity = density
			}
		}

		// (5.3) stop the loop if not new centerPt is found
		if centerPt == n {
			break
		}

		// (5.4) officially create the community
		newCommunity := map[int]bool{centerPt: true}
		communities = append(communities, newCommunity)
		communityIDs[centerPt] = c

		// (5.5) iteratively append neighbors to the new community
		boundary := map[int]bool{centerPt: true}
		for len(boundary) > 0 {
			newBoundary := map[int]bool{}
			for bpt, _ := range boundary {
				bptNoncoreNeighbors, exists := noncoreNeighbors[bpt]
				if exists {
					for neighbor, _ := range bptNoncoreNeighbors {
						// skip those already in a community
						if communityIDs[neighbor] >= 0 {
							continue
						}
						newCommunity[neighbor] = true
						communityIDs[neighbor] = c
					}
				}
				bptCoreNeighbors, exists := coreNeighbors[bpt]
				if !exists {
					continue
				}
				for neighbor, _ := range bptCoreNeighbors {
					// skip those already in a community
					if communityIDs[neighbor] >= 0 {
						continue
					}
					newBoundary[neighbor] = true
					newCommunity[neighbor] = true
					communityIDs[neighbor] = c
				}
			}
			boundary = newBoundary
		}
	}

	// -------------------------------------------------------------------------
	// step 6: add isolated points into the result
	for pt := 0; pt < cm.n; pt++ {
		if communityIDs[pt] < 0 {
			newCommunity := map[int]bool{pt: true}
			communityIDs[pt] = len(communities)
			communities = append(communities, newCommunity)
		}
	}

	// -------------------------------------------------------------------------
	// step 7: return the result
	return communities, communityIDs
}

// =============================================================================
// func flattenCommunities
// brief description: expand the aggregated concurrence graph's communities at
//	the original concurrence graph.
// input:
//	aggCommunities: the aggregated concurrence graph's communities
//	communities: the original concurrence graph's communities
// output:
//	the flatten communities
func flattenCommunities(aggCommunities, communities []map[int]bool,
) []map[int]bool {
	result := []map[int]bool{}
	for _, aggC := range aggCommunities {
		newC := map[int]bool{}
		for idxC, _ := range aggC {
			c := communities[idxC]
			for pt, _ := range c {
				newC[pt] = true
			}
		}
		result = append(result, newC)
	}
	return result
}

// =============================================================================
// func Louvain
// brief description: Louvain algorithm for partition optimization of
//	concurrence graphs.
// input:
//	qm: a quality model.
//	communities: a list of clusters.
//	opts: an optional list of options
// output:
//	the optimized communities that maximizes quality
// note:
//	If the input communities is empty, this function will act as the classical
//	Louvain algorithm that uses single point communities as the initial
//	communities.
func Louvain(qm QualityModel, communities []map[int]bool, communityIDs []int, maxIters int,
) ([]map[int]bool, []int) {
	// -------------------------------------------------------------------------
	// step 1: initialize communities and communityIDs if they are nil
	n := qm.GetN()
	if communities == nil || communityIDs == nil {
		communities = make([]map[int]bool, n)
		communityIDs = make([]int, n)
		for i := 0; i < n; i++ {
			communities[i] = map[int]bool{i: true}
			communityIDs[i] = i
		}
	}

	// -------------------------------------------------------------------------
	// step 2: iteratively scan through the points to find out what is the best
	// community for a point. If all points are in their best communities, stop
	// the iteration.
	numCPUs := runtime.NumCPU()
	var wg sync.WaitGroup
	type MergeRequest struct {
		dst  int
		gain float64
	}
	type MergeDecision struct {
		src  int
		gain float64
	}
	mergeRequests := make([]MergeRequest, n)
	mergeOrders := make([]int, n)
	numIters := 0
	for iter := 0; iter < maxIters; iter++ {
		// (2.1) compute merge requests
		m := len(communities)
		wg.Add(numCPUs)
		for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
			go func(idxCPU int) {
				u0 := n * idxCPU / numCPUs
				u1 := n * (idxCPU + 1) / numCPUs
				gains := make([]float64, m)
				for u := u0; u < u1; u++ {
					mergeRequests[u] = MergeRequest{dst: -1, gain: 0.0}
					mergeOrders[u] = u
					oldCu := communityIDs[u]
					neighbors := qm.GetNeighbors(u)
					sumGains := 0.0
					if len(neighbors) < m {
						visitedCommunities := map[int]float64{}
						for neighbor, _ := range neighbors {
							newCu := communityIDs[neighbor]
							if newCu == oldCu {
								continue
							}
							_, visited := visitedCommunities[newCu]
							if visited {
								continue
							}

							deltaQ := qm.DeltaQuality(communities, u, oldCu, newCu)
							if deltaQ > 0.0 {
								visitedCommunities[newCu] = deltaQ
								sumGains += deltaQ
							} else {
								visitedCommunities[newCu] = 0.0
							}
						}

						if sumGains > 0.0 {
							x := rand.Float64() * sumGains
							sum := 0.0
							for c, gain := range visitedCommunities {
								sum += gain
								if sum >= x {
									//fmt.Printf("u = %d, c = %d, sum = %v, x = %v\n", u, c, sum, x)

									mergeRequests[u].dst = c
									mergeRequests[u].gain = gain
									break
								}
							}
						}
					} else {
						for newCu := 0; newCu < m; newCu++ {
							if newCu == oldCu {
								gains[newCu] = 0.0
								continue
							}
							deltaQ := qm.DeltaQuality(communities, u, oldCu, newCu)
							if deltaQ > 0.0 {
								gains[newCu] = deltaQ
								sumGains += deltaQ
							} else {
								gains[newCu] = 0.0
							}

							// if deltaQ > mergeRequests[u].gain {
							// 	mergeRequests[u].dst = newCu
							// 	mergeRequests[u].gain = deltaQ
							// }
						}

						if sumGains > 0.0 {
							x := rand.Float64() * sumGains
							sum := 0.0
							for c := 0; c < m; c++ {
								sum += gains[c]
								//fmt.Printf("u = %d, sum = %v, x = %v\n", u, sum, x)
								if sum >= x {
									mergeRequests[u].dst = c
									mergeRequests[u].gain = gains[c]
									break
								}
							}
						}
					}
				}
				wg.Done()
			}(idxCPU)
		}
		wg.Wait()

		// (2.2) sort merge requests
		sort.Slice(mergeOrders, func(i, j int) bool {
			return mergeRequests[mergeOrders[i]].gain > mergeRequests[mergeOrders[j]].gain
		})

		// (2.3) exit the loop if no merge is required
		bestMerge := mergeRequests[mergeOrders[0]]
		if bestMerge.dst < 0 || bestMerge.gain <= 0.0 {
			break
		}

		// (2.4) compute merge decisions
		mergeDecisions := make([]MergeDecision, m)
		for i := 0; i < m; i++ {
			mergeDecisions[i] = MergeDecision{src: -1, gain: 0.0}
		}
		mergeDecisions[bestMerge.dst].src = mergeOrders[0]
		mergeDecisions[bestMerge.dst].gain = bestMerge.gain
		mergeDecisions[communityIDs[mergeOrders[0]]].src = -2
		//fmt.Printf("move %d to cluster %d for gain %v\n", mergeOrders[0], bestMerge.dst, bestMerge.gain)
		totalGain := bestMerge.gain
		for i := 1; i < n; i++ {
			// skip those in communities that have already changed
			uI := mergeOrders[i]
			oldCuI := communityIDs[uI]
			if mergeDecisions[oldCuI].src >= 0 || mergeDecisions[oldCuI].src < -1 {
				//fmt.Printf("cannot move %d out of cluster %d\n", i, oldCuI)
				continue
			}

			// skip those does not request to move
			mergeI := mergeRequests[uI]
			newCuI := mergeI.dst
			if newCuI < 0 {
				continue
			}

			// skip those want to enter communities that have already changed
			if mergeDecisions[newCuI].src >= 0 || mergeDecisions[newCuI].src < -1 {
				//fmt.Printf("cannot move %d of cluster %d into cluster %d\n", i, oldCuI, newCuI)
				continue
			}

			// confirm the request
			mergeDecisions[newCuI].src = uI
			mergeDecisions[newCuI].gain = mergeI.gain
			mergeDecisions[oldCuI].src = -2
			//fmt.Printf("move %d of cluster %d into cluster %d for gain %v\n", uI, oldCuI, newCuI, mergeI.gain)
			totalGain += mergeI.gain
		}

		// (4.3) move points
		numMoves := 0
		for i := 0; i < m; i++ {
			if mergeDecisions[i].src >= 0 {
				u := mergeDecisions[i].src
				oldCu := communityIDs[u]
				communityIDs[u] = i
				communities[i][u] = true
				delete(communities[oldCu], u)
				numMoves++
			}
		}

		// (4.4) remove empty communities
		lastC := m - 1
		for len(communities[lastC]) == 0 {
			lastC--
		}
		//fmt.Printf("lastC = %d\n", lastC)
		for c := 0; c <= lastC; c++ {
			community := communities[c]
			if len(community) == 0 {
				communities[c] = communities[lastC]
				communities[lastC] = community
				//oldLastC := lastC
				for len(communities[lastC]) == 0 && lastC > c {
					lastC--
				}
				//fmt.Printf("%d is empty, switch with %d, lastC = %d\n", c, oldLastC, lastC)
			}
		}
		communities = communities[:lastC+1]
		for c, community := range communities {
			for u, _ := range community {
				communityIDs[u] = c
			}
		}

		// (4.5) report statistics
		fmt.Printf("iter %d: move %d points, gain %v\n", numIters, numMoves, totalGain)
		numIters++
	}

	// -------------------------------------------------------------------------
	// step 7: return the result
	return communities, communityIDs
}

// // =============================================================================
// // func refineForLeiden
// // brief description: refine communities for Leiden algorithm
// // input:
// //	qm: a quality model.
// //	communities: a list of clusters
// //	gamma: the threshold for qm.connectsWell
// //	theta: a threshold for sampling probablities
// // output:
// //	refinedCommunities, refinement
// //	refinedCommunities: the result communities refined from input communities.
// //	refinement: for each input community, list which refined communities it
// //		contains.
// func refineForLeiden(qm QualityModel, communities []map[uint]bool,
// 	gamma, theta float64) ([]map[uint]bool, []map[uint]bool) {
// 	if gamma <= 0.0 || theta <= 0.0 {
// 		log.Fatal("gamma and theta must be > 0.")
// 	}

// 	// -------------------------------------------------------------------------
// 	// step 1: initialize result with singleton communities
// 	n := qm.GetN()
// 	refinedCommunties := make([]map[uint]bool, n)
// 	for i := uint(0); i < n; i++ {
// 		refinedCommunties[i] = map[uint]bool{i: true}
// 	}

// 	// -------------------------------------------------------------------------
// 	// step 2: find out for each point which input community it is in.
// 	inputCommunityID := make([]int, n)
// 	for idxC, c := range communities {
// 		for u, _ := range c {
// 			inputCommunityID[u] = idxC
// 		}
// 	}

// 	// -------------------------------------------------------------------------
// 	// step 3: iteratively merge communities in result based on five rules:
// 	//	1. 	A community in result can only be merged with a sub-community in
// 	//		one of the input communties.
// 	//	2.	A communtiy in result is merged only if the merge increases the
// 	//		quality of the result.
// 	//	3.	Two communities are merged only if both of them are well connected
// 	//		to input communities with threshold gamma and at least one of them
// 	//		is singleton.
// 	//	4.	Tow communities are merged only if they are connected.
// 	//	5.	When a community can be merged with multiple communities, we select
// 	//		which to be merged with randomly with sampling probablities set as
// 	//		proportional to 1/theta * qualityGain
// 	for {
// 		done := true
// 		for i, refinedCi := range refinedCommunties {
// 			// ----------------------------------------------------------------
// 			// (3.1) skip non-singleton communities in result
// 			if len(refinedCi) > 1 {
// 				continue
// 			}

// 			// ----------------------------------------------------------------
// 			// (3.2) skip those refinedCi not connected well with any of the
// 			// input communities
// 			inputC := communities[inputCommunityID[i]]
// 			if !qm.ConnectsWell(refinedCi, inputC, gamma) {
// 				continue
// 			}

// 			// ----------------------------------------------------------------
// 			// (3.3) find those result communities in the same input community
// 			// as refinedC that has at least one node connected to refinedCi.
// 			// This enforces rule 1 and 4.
// 			connected := []uint{}
// 			u := uint(i)
// 			for j, _ := range inputC {
// 				refinedCj := refinedCommunties[j]
// 				// skip empty result communities
// 				if len(refinedCj) == 0 {
// 					continue
// 				}

// 				// Check whether there is at least one node connected to
// 				// refinedCi. If there is, append j to connected
// 				for v, _ := range refinedCj {
// 					if qm.Connects(u, v) {
// 						connected = append(connected, j)
// 						break
// 					}
// 				}
// 			}

// 			// ----------------------------------------------------------------
// 			// (3.4) scan throughs connected to search for those resultCi can
// 			// be merged with.
// 			logProb := map[uint]float64{}
// 			sumLogProb := 0.0
// 			for _, j := range connected {
// 				refinedCj := refinedCommunties[j]

// 				// Skip this refinedCj if it is not connected well to inputC.
// 				// This completes the enforcement of rule 3 with (3.1) & (3.2).
// 				if !qm.ConnectsWell(refinedCj, inputC, gamma) {
// 					continue
// 				}

// 				// Compute the quality gain when merging resultCi and resultCj
// 				deltaQuality := qm.DeltaQuality(refinedCommunties, u, u, j)

// 				// Skip this if the quality gain is not positive
// 				if deltaQuality <= 0.0 {
// 					continue
// 				}

// 				// record a sampling probability
// 				myLogProb := deltaQuality / theta
// 				logProb[j] = myLogProb
// 				sumLogProb += myLogProb
// 			}

// 			// ----------------------------------------------------------------
// 			// (3.5) if none resultCi be merged with, skip this resultCi
// 			if len(logProb) == 0 {
// 				continue
// 			}

// 			// ----------------------------------------------------------------
// 			// (3.6) normalize the sampling probabilities
// 			for j, value := range logProb {
// 				logProb[j] = value - sumLogProb
// 			}

// 			// ----------------------------------------------------------------
// 			// (3.7) sample a resultCj using logProb
// 			// first, get a random number x within [0.0, 1.0)
// 			x := rand.Float64()
// 			// then, scan through logProb to find the sample
// 			y := 0.0
// 			sample := uint(0)
// 			for j, value := range logProb {
// 				prob := math.Exp(value)
// 				y += prob
// 				if y >= x {
// 					sample = j
// 					break
// 				}
// 			}

// 			// ----------------------------------------------------------------
// 			// (3.8) now merge resultCi and sample
// 			refinedCommunties[i] = map[uint]bool{}
// 			refinedCommunties[sample][u] = true
// 			done = false
// 		}

// 		// ---------------------------------------------------------------------
// 		// end the loop if no merge happens
// 		if done {
// 			break
// 		}
// 	}

// 	// -------------------------------------------------------------------------
// 	// step 4: remove empty communties in result and record non-empty ones in
// 	// the refinement mapping
// 	oldRefinedCommunities := refinedCommunties
// 	refinedCommunties = []map[uint]bool{}
// 	refinement := make([]map[uint]bool, len(communities))
// 	for i := 0; i < len(communities); i++ {
// 		refinement[i] = map[uint]bool{}
// 	}
// 	for i, c := range oldRefinedCommunities {
// 		if len(c) > 0 {
// 			newI := uint(len(refinedCommunties))
// 			refinement[inputCommunityID[i]][newI] = true
// 			refinedCommunties = append(refinedCommunties, c)
// 		}
// 	}

// 	// -------------------------------------------------------------------------
// 	// step 5: return the result
// 	return refinedCommunties, refinement
// }

// // =============================================================================
// // func Leiden
// // brief description: Leiden algorithm for partition optimization of
// //	concurrence graphs.
// // input:
// //	qm: a quality model.
// //	communities: a list of clusters.
// //	opts: an optional list of options
// // output:
// //	the optimized communities that maximizes quality
// // note:
// //	If the input communities is empty, this function will act as the classical
// //	Leiden algorithm that uses single point communities as the initial
// //	communities.
// func Leiden(qm QualityModel, communities []map[uint]bool, gamma, theta float64,
// 	opts ...string) []map[uint]bool {
// 	// step 1: parsing options
// 	useSeqSelector := true
// 	multiResolution := true
// 	shuffle := false
// 	for _, opt := range opts {
// 		switch opt {
// 		case "priority selector":
// 			useSeqSelector = false
// 		case "sequential selector":
// 			useSeqSelector = true
// 		case "single resolution":
// 			multiResolution = false
// 		case "multiple resolution":
// 			multiResolution = true
// 		case "shuffle":
// 			shuffle = true
// 		case "no shuffle":
// 			shuffle = false
// 		}
// 	}

// 	// -------------------------------------------------------------------------
// 	// step 2: complete communities with isolated points added as single point
// 	// communities.
// 	result := qm.GetCompleteCommunities(communities)
// 	n := qm.GetN()

// 	// -------------------------------------------------------------------------
// 	// step 3: get the community ID for each point
// 	communityIDs := make([]uint, n)
// 	for communityID, community := range result {
// 		for point, _ := range community {
// 			communityIDs[point] = uint(communityID)
// 		}
// 	}

// 	// -------------------------------------------------------------------------
// 	// step 4: iteratively scan through the points to find out what is the best
// 	// community for a point. If all points are in their best communities, stop
// 	// the iteration.
// 	m := uint(len(result))
// 	for {
// 		// (4.1) create the access order of points
// 		points := make([]uint, n)
// 		for i := 0; i < int(n); i++ {
// 			points[i] = uint(i)
// 		}

// 		// (4.2) optionally, shuffle the access order of points
// 		if shuffle {
// 			rand.Shuffle(int(n), func(i, j int) {
// 				points[i], points[j] = points[j], points[i]
// 			})
// 		}

// 		// (4.3) move points
// 		if useSeqSelector {
// 			done := true
// 			for _, u := range points {
// 				oldCu := communityIDs[u]
// 				bestDeltaQuality := 0.0
// 				bestNewCu := oldCu
// 				for newCu := uint(0); newCu < m; newCu++ {
// 					deltaQuality := qm.DeltaQuality(result, u, oldCu, newCu)
// 					if deltaQuality > bestDeltaQuality {
// 						bestDeltaQuality = deltaQuality
// 						bestNewCu = newCu
// 					}
// 				}

// 				if bestDeltaQuality > 0.0 {
// 					delete(result[oldCu], u)
// 					result[bestNewCu][u] = true
// 					communityIDs[u] = bestNewCu
// 					done = false
// 				}
// 			}
// 			if done {
// 				break
// 			}
// 		} else {
// 			bestDeltaQuality := 0.0
// 			bestU := uint(0)
// 			oldCBestU := communityIDs[0]
// 			bestNewCu := oldCBestU
// 			for _, u := range points {
// 				oldCu := communityIDs[u]
// 				for newCu := uint(0); newCu < m; newCu++ {
// 					deltaQuality := qm.DeltaQuality(result, u, oldCu, newCu)
// 					if deltaQuality > bestDeltaQuality {
// 						bestDeltaQuality = deltaQuality
// 						bestU = u
// 						oldCBestU = oldCu
// 						bestNewCu = newCu
// 					}
// 				}
// 			}
// 			if bestDeltaQuality == 0.0 {
// 				break
// 			}
// 			delete(result[oldCBestU], bestU)
// 			result[bestNewCu][bestU] = true
// 			communityIDs[bestU] = bestNewCu
// 		}
// 	}

// 	// -------------------------------------------------------------------------
// 	// step 5: remove empty communities
// 	oldResult := result
// 	result = []map[uint]bool{}
// 	for _, c := range oldResult {
// 		if len(c) > 0 {
// 			result = append(result, c)
// 		}
// 	}

// 	// -------------------------------------------------------------------------
// 	// step 6: if required, do the multi-resolution part
// 	if multiResolution {
// 		// ---------------------------------------------------------------------
// 		// (6.1) refine the result
// 		refinedCommunities, refinement := refineForLeiden(qm, result, gamma, theta)

// 		// ---------------------------------------------------------------------
// 		// (6.2) create aggregate network from refined
// 		newQM := qm.Aggregate(refinedCommunities)

// 		// ---------------------------------------------------------------------
// 		// (6.2) compute aggregated result from the aggregate network
// 		aggResult := Leiden(newQM, refinement, gamma, theta, opts...)

// 		// -------------------------------------------------------------------------
// 		// (6.3) flatten the aggResult with refinedCommunities into result
// 		result = flattenCommunities(aggResult, refinedCommunities)
// 	}

// 	// -------------------------------------------------------------------------
// 	// step 7: return the result
// 	return result
// }
