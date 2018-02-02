# kaldi diarization

这里主要关注的是kaldi diarization的recipe中，得到了plda的score之后，如何对一个音频中所有的segments进行聚类并得到最后的结果(DER)

### 1. 文件目录结构
下面**不是**diarization完整的文件目录结构，而是与plda打分之后的步骤相关的文件目录
```
diarization                             顶层diarization目录
└── baseline                            项目目录，如baseline           
    ├── run.sh                          训练脚本
    ├── diarization                     diarization脚本目录
    │   ├── cluster.sh                  聚类脚本
    │   └── make_rttm.py                生成rttm类型的结果文件
    ├── data                            数据目录
    │   ├── callhome1                   callhome数据被分为两部分
    │   └── callhome2                   1和2各占callhome一半的数据
    ├── exp                             实验训练目录
    │   ├── ivectors_callhome1          callhome1的训练目录
    │   │   ├── plda_scores             plda score的存放目录
    │   │   ├── plda_scores_t($n)       threshold为$n的结果目录
    │   │   └── plda_scores_num_spk     supervised(已知说话人数量)的结果目录
    │   ├── ivectors_callhome2          同上callhome1
    │   │   ├── plda_scores
    │   │   ├── plda_scores_t($n)
    │   │   └── plda_scores_num_spk
    └── ├── tuning                      threshold tuning目录
        │   ├── callhome1_t-($n)        callhome1中threshold为$n的DER       
        │   ├── callhome1_best          callhome1中结果最好的threshold
        │   ├── callhome2_t-($n)        同上callhome1
        │   └── callhome2_best
        └── result                      最终结果目录(unsupervise和supervise的DER)
```

### 2. 脚本调用

diarization实验的主调用脚本是例子中的run.sh，其中stage5是得到plda的分数，stage6是对callhome数据集进行无监督(说话人个数未知)的聚类并计算DER，stage7是对callhome数据集进行有监督的聚类并计算DER

#### unsupervised clustering
```
1. 调用utils/filter_scp.pl 
    input:  rttm格式的reference文件(rttm会在后面提到)
    output: 两个rttm文件
    op:     将reference一分为二，分别对应callhome1和callhome2

2. 对两部分数据分别设置不同的threshold(-0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3)进行AHC聚类并计算DER，
   对于每一个threshold: (AHC算法以及kaldi代码的实现后面会具体分析，这里只涉及大致的脚本调用)

    2.1 调用diarization/cluster.sh
            input:  threshold和plda scores
            output: segment对应的label，rttm
            op:     对所有的segment进行聚类并计算DER

    2.2 调用md-eval.pl
            input:  reference对应的rttm文件和clustering之后得到的rttm文件
            output: 评测结果，diarization只需要用到DER
            op:     通过比较label文件和结果文件得DER

3. 比较使用不同threshold时的DER，找到最低的DER对应的threshold(对于callhome1和callhome2应该得到两个)

4. 进行heldout validation，用callhome2中找到的最好的threshold去对callhome1进行聚类，
   相应的用callhome1中找到的最好的threshold对callhome2进行聚类，用到的脚本还是cluster.sh

5. 用得到的聚类结果计算最终的unsupervised clustering的DER，保存目录为exp/result
```

#### supervised clustering
supervised clustering与上面unsupervised clustering相比，减少了确定threshold的那一步，因为在有监督的情况下，已知说话人的个数，也就是说聚类中心的个数是确定的，不需要额外的threshold去判断聚类是否结束(算法实现上的区别会在后面分析)

所以只需要对两部分数据分别聚类并计算最终的DER即可，用到的脚本仍是diarization/cluster.sh，同时去掉threshold的参数，加上保存说话人个数的文件作为参数。 


#### RTTM
NIST RTTM文件可以由diarization/make_rttm.py脚本得到，输入为segments和label，可以处理segments之间的overlap，RTTM具体格式如下:
```
<type> <file> <chnl> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>

where:
<type> = "SPEAKER"
<file> = <recording-id>
<chnl> = "0"
<tbeg> = start time of segment
<tdur> = duration of segment
<ortho> = "<NA>"
<stype> = "<NA>"
<name> = <speaker-id>
<conf> = "<NA>"
<slat> = "<NA>"
```
关于RTTM的详细内容可参考[KWS15 KEYWORD SEARCH EVALUATION PLAN](https://www.nist.gov/sites/default/files/documents/itl/iad/mig/KWS15-evalplan-v05.pdf)

### 3. Agglomerative Hierarchical Clustering(层次聚类)
####　伪代码(对应kaldi代码中的算法)
我写的可能有点长。。。
```cpp
# algorithm: use plda scores to do agglomerative hierarchical clustering
# ----------------------------------------------------------------------
# input:  1. plda scores of all pairs of segments for each recording
#         2. threshold & minimum cluster or number of speakers
# ouptut: the label of each segments after clustering

// Count of clusters that have been created. Also used to give clusters unique IDs
count = 0
// number of active clusters 
num_active = 0
// Map from cluster IDs to cost between them
cost_map_
// IDs of unmerged clusters
active_clusters_
// Map from cluster ID to cluster object address
cluster_map_
// Priority queue using greater (lowest costs are highest priority).
// Elements contain pairs of cluster IDs and their cost.
queue_ = priority_queue

function Cluster(cost_, num_cluster, min_cluster, threshold)
    Initialize(cost_, num_cluster, threshold)
    while num_active > min_cluster && !queue_.empty()
        pair(i, j) = queue_.top()
        queue_.pop()
        if i not in active_clusters_ && j not in active_clusters_
            MergeClusters(i, j)
    label = 0
    for cluster in active_clusters_
        ++label
        for segment in cluster
            assignments[segment] = label
    return assignments
end function

function Initialize(cost_, num_cluster)
    num_active = num_cluster
    for i=0 to num_cluster
        vector segments_ids_
        cluster = NewCluster(++count, -1, -1, segments_ids_)
        cluster_map_[count] = cluster
        active_clusters_.insert(count)
        for j=i+1 to num_cluster
            cost = cost_(i, j)
            cost_map_[pair(i+1, j+1)] = cost(i, j)
                if cost <= threshold
                    queue_.push(pair(cost, i+i, j+1))
end function

function GetCost(i, j)
    if i < j
        return cost_map_(i, j)
    else
        return cost_map_(j, i)
end function

function NewCluster(id, parent1, parent2, size, segments_ids_)
    new cluster
    cluster->id = id
    cluster->parent1 = parent1
    cluster->parnet2 = parent2
    cluster->size = size
    cluster->segments_ids_ = segments_ids_
    return cluster
end function

function MergeClusters(i, j)
    cluster1 = cluster_map_[i]
    cluster2 = cluster_map_[j]
    cluster1->id = ++count
    cluster1->parent1 = i
    cluster1->parent2 = j
    cluster1->size += cluster2->size
    cluster1->segments_ids_.insert(cluster2->segments_ids_)
    active_clusters_.erase(i)
    active_clusters_.erase(j)
    for cluster in active_clusters_
        new_cost = GetCost(cluster->id, i) + GetCost(cluster->id,j)
        cost_map_[pair(cluster-id, count)] = new_cost
        norm - cluster1->size * cluster->size
        if new_cost/norm <= threshold
            queue_.push(pair(new_cost/norm, cluster->id, count))
    active_clusters_.insert(count)
    cluster_map_[count] = cluster1
    delete cluster2
    num_cluster--
end function
```

#### kaldi算法具体的代码实现(与伪代码对应)
主要源码文件目录
```
# 外层数据读取及调用
kaldi/src/ivectorbin/agglomerative-cluster.cc
# 算法实现
kaldi/src/ivector/agglomerative-clustering.cc
kaldi/src/ivector/agglomerative-clustering.h
```

1. **score or cost**

    在最外层cluster.sh调用的时候可以设置--read-costs参数，来指定所有segments两两之间关系的参数是表示“score”还是“cost”，前者表示两个segment之间的分数越大表示越相似，而后者表示分数越小越相似，在代码中如果接收的参数表示score，会对多有的分数*(-1)，也就是说统一转化为cost来处理

2. **initialization**

    算法实现中，每一个cluster的数据结构包括了id, 2个父结点id，当前类中包括的segment的数量和列表。此外，算法记录了当前active cluster的数量和列表，active cluster的意思就是活跃的还未被合并的类，同时创建了一个优先队列(小顶堆)，把所有距离小于threshold的cluster pair放入队列中

3. **clustering**

    聚类的主要思想就是每次找到最相似的两个类进行合并，聚类结束的条件是当前cluster的个数小于设置的聚类中心的个数，或者是队列为空， 聚类循环中的操作是每次取队首的元素，也就是当前距离最短的两个类，如果它们都属于active cluster，就把它们合并，重复直到达到结束条件，然后将聚类之后的结果输出

4. **merge cluster**

    当需要合并两个cluster，i 和 j 时，为了节省内存，是直接把后一个cluster更新到第一个cluster上，并在active cluster列表中除去 i 和 j，最后删除后一个cluster，使记录当前active cluster个数的变量 -1

    合并的操作具体是首先把 cluster i 的id自增，表示这是一个新的cluster new，并加入到active cluster中，然后设置 new 的两个父结点分别为 i 和 j，将 cluster j 中的segments加入到 i 中，表示 new 中含有的segment，然后计算所有其他 active cluster 和 cluster new 的距离，这个距离就是和 new 的两个父结点之间的距离的和，在计算距离的同时，如果 距离/(2个cluster的segments size的乘积) 小于 threshold，就把这两个cluster组成的pair放到队列中

5. **supervise & unsupervise**

    supervise和unsupervise在实现上的区别在于

    - unsupervise需要手动设置一个threshold，然后将最小的聚类中心个数设为1，这时聚类结束条件的第一个(当前cluster的个数小于聚类中的个数)是没有作用的，需要依赖第二个条件(存储相似cluster的队列为空，没有可以合并的cluster)来结束聚类。

    - supervise已知了说话人的数量，也就是聚类中心的数量确定，在代码中设置threshold为limit::max，最小聚类中心的个数设置为说话人的个数，所有的cluster pair都可以进入队列，也就是说会一直合并，依赖第一个结束条件来结束聚类。