# speaker diarization 调研报告

## 任务定义
speaker diarization主要是一个针对多人对话场景中的说话人分割和聚类的任务，对于一段输入的音频流，首先要将其分割为多个只包含单个源的片段，然后再把对应于相同源的片段聚为一类，其中的"源"包括不同的说话人，音乐或者是背景噪声，也就是所谓的"鸡尾酒问题"

speaker diarization不同于说话人识别和语音识别，说话人识别是确定说话人的身份(who is speaking)，语音识别是获得说话人的语音内容(what spoken)，diarization是针对"who spoke when"的问题，主要目标就是正确的分割音频并将音频片段与相应的说话人对应，而不关注说话人的真实身份，所以这个任务包括了分割和聚类，也就是先找到音频流中说话人切换的change points，然后就是依据说话人的特征将音频片段进行聚类。

## 传统处理流程及系统分类
diarization系统的输入输出如下图所示，输入的是音频文件的特征，一般采用MFCC特征，然后进行分割，最后输出的是聚类之后的结果，也就是每一个段对应所对应的类别。
<p align="left"><img width="40%" src="pic/input_output.png" /></p>

diarization系统的处理流程大致如下
<p align="left"><img width="40%" src="pic/structure.png" /></p>

其中主要包括segmentation和clustering两个部分：

### Segmentation
<p align="left"><img width="40%" src="pic/segmentation.png" /></p>

segmentation的话就是对于输入的音频特征的序列，首先可以用固定长度将其划分为连续的小窗口，窗口的长度可以根据任务的要求或者是数据的特点来进行调整，相邻的窗口之间可以有overlap，然后就从头开始依次计算每两个窗口之间的距离，然后设定一个阈值，当两个窗口之间的距离超过阈值时，可以认为是发现了一个change point，在此处进行分割。这种计算两个窗口之间距离的方法有很多，常用的有BIC和KL2 distance

### Clustering
<p align="left"><img width="40%" src="pic/clustering.png" /></p>

clustering的话就是把之前分割的语音片段进行聚类，把属于同一个说话人或者是同一类的片段聚到一起去，根据聚类方法的不同可以把diarization的系统分为两种：
- **bottom-up**
	自底向上的方法是在聚类的时候初始化较多的类(一般多于说话人的数量)，然后通过类的不断合并来达到聚类的效果
- **top-down**
	自顶向下的方法是初始化很少的类(通常为1)，然后通过类的分离来得到最后的结果
<p align="left"><img width="40%" src="pic/classify.png" /></p>

这两类系统都是通过迭代直至收敛来找到一个最优的类的个数。如果最后得到的类的个数大于真实类的个数，那么系统就是聚类不足的(under-clustering)，反之就是聚类过度(over-clustering)。现在比较流行的是**bottom-up**的系统，它可以得到更好的结果，top-down系统的特点是速度较快，计算复杂度低。

在bottom-up的系统中常用的聚类方法是Agglomerative hierarchical clustering(AHC)，可以理解为层次聚类方法
<p align="left"><img width="40%" src="pic/AHC.png" /></p>

层次聚类方法的过程：
1. 初始化，将分割得到的每一个片段都设置为一个单一的类，设置表示参数(后面具体说明)
2. 计算所有类两两之间的距离(这里距离的计算后面会具体说明)
3. 找到距离最近的那两个类，并将它们合并成一个新的类
4. 更新类的参数并重新计算这个类与其余类的距离
5. 重复步骤3,4直到满足结束条件(这里的结束条件一般是类的个数减少到一定的数量)

### Optional
其余的就是一些optional的操作，包括可以在音频输入的时候加一个vad，目的是消除背景噪声和静音段的影响；或者是在segmentation之后加上一个比较粗略的分类，比如说可以将所有的片段分为male和female两部分，这样做的目的是在clustering的过程中可以使用不同的参数来进行自适应，可以提高聚类的准确性；还有就是在clustering之后可以再做re-segmentation来提高性能。


## 评价标准
diarization系统的评价标准主要是Diarization Error Rate (DER)，这个错误率包括了三种错误：
- missed speech (MS)
- false alarm (FA)
- speaker error (SE)
<p align="left"><img width="40%" src="pic/der.png" /></p>

最终的DER的计算公式就是三种错误率的和: **DER = MS + FA + SE**
部分论文中使用EER，DCF？来评价结果，还有论文提出了一些新的边界评估框架

## 当前主要方法和改进
因为用的数据集不尽相同，所以没有方法之间详细的结果对比
### PLDA for i-vector
<p align="left"><img width="40%" src="pic/i-vector_plda.png" /></p>

这个方法的话主要就是在segmentation之后，对于每个segment计算一个i-vector用于表示这个片段，然后用PLDA的方法取代原来的cosine打分，来计算两个片段之间的距离，最后使用层次聚类的方法得到结果。

其中每个片段的长度大概是1-2s，然后迭代结束的条件是用无监督的标注方法得到的。在callhome数据集上的结果如图所示:
<p align="left"><img width="40%" src="pic/der_of_plda_ivector.png" /></p>

### DNN Embedding
<p align="left"><img width="40%" src="pic/embedding_plda.png" /></p>

这个embedding的方法就是对于分割之后的每个segment，用定长的embedding向量来表示，避免了i-vector的计算
<p align="left"><img width="40%" src="pic/dnn_scoring.png" /></p>

而所有的embedding向量是通过一个DNN的结构得到的，如下图所示，这个DNN的输入是每个segment的特征序列，输出的是一个固定维度(400)的embedding向量，同时还会输出一个对称矩阵S和一个偏置b，这两个输出会在计算两个embedding向量距离的时候用到，DNN中的NIN Layer表示network in network，是一种将大网络拆分的操作，可以加快DNN的速度。
两个embedding向量之间的距离定义如下，与PLDA相似:
$$L(x, y)=x^Ty-x^TSx-y^TSy+b$$
在callhome上面最好的结果是DER=9.9%

### Embedding from DNN hidden layer
<p align="left"><img width="40%" src="pic/speaker_embedding.png" /></p>

这个方法中embedding的方式和上面所说的那个不同，根据论文中所说的，考虑到在做说话人识别相关任务的时候，训练的DNN模型在隐层中压缩了很多相关的特征，所以可以从隐层神经元的激活状态中得到一个特征向量，如上图所示，具体的做法就是利用DNN结构中的某一个隐层来作为speaker embedding的向量，这个DNN的输入是从GMM-UBM得到的61440维的超向量$s_g$
$$s_g=\frac{1}{\sum_t \gamma_g(t)}\sum_t \gamma_g(t)(x_t-\mu_g)$$
其中$\gamma_g(t)$是第$t$帧属于第$g$个高斯分量的后验概率，$x_t$表示第$t$帧，$\mu_g$表示GMM-UBM的均值。
<p align="left"><img width="40%" src="pic/embedding_feature.png" /></p>

上面是所提取的embedding向量的例子，颜色越亮表示数值越高，每个说活人有两段提取的特征，可以每个说话人各自的特征还是有很多相似之处的
在ETAPE数据集上的结果:
<p align="left"><img width="40%" src="pic/der_of_hidden_embedding.png" /></p>

### Cross-show diarization
<p align="left"><img width="40%" src="pic/cross_show.png" /></p>

这种cross-show的方法是主要是在聚类上做了改变，论文中认为传统的AHC，也就是层次聚类的方法无法保证得到一个最优解，所以将聚类的问题转化成ILP (Integer Linear Programming)的形式，这里涉及到一些公式的推导，具体细节看论文，大概的意思就是这种聚类的方法与传统的bottom-up的方法不同，它是定义了一些辅助的隐变量，然后写出一个全局的目标函数，通过最小化类的个数和类内方差来进行计算。而cross-show的方法就是包含了两层的结构，第一层就是分离的操作，进行正常的分割和聚类，第二层利用IPL在全局上再进行一次重新的聚类。

这种方法在REPERE数据集上可以使得DER下降0.82%

## 难点和挑战
1. 在数据中会有多人同时说话(overlapping)的情况，是当前一个比较明显的问题，由overlapping带来的错误在最终的错误率中占有较大的比重，这种场景的增加会使得DER明显提高。现在的diarization系统对于overlapping也有一些初步的解决方法，例如通过HMM的状态转换来找出音频中overlapping的部分，然后处理overlapping部分的片段时保留多个结果
<p align="left"><img width="40%" src="pic/overlapping.png" /></p>

这些解决方法对于结果有一定的提高，但是有较为明显的局限性，需要假设所有overlapping的部分都是2个说话人重叠。

2. 数据集的微小变化会对diarization系统产生很大的影响，鲁棒性有待提高。
3. 目前diarization的方法不够成熟，普适性较差，无法适用于多个领域(新闻播报，客服电话等)

## 数据集
1. CALLHOME conversational telephone speech corpus. We evaluated our systems using the CALLHOME corpus, which is a CTS collection between familiar speakers. Within each conversation, all speakers are recorded in a single channel. There are anywhere between 2 and 7 speakers (with the majority of conversations involving between 2 and 4), and the corpus also is distributed across six languages: Arabic, English, German, Japanese, Mandarin, and
Spanish.
2. TBL is TV broadcast data which consists of 22 programmes from a talk–show with single distant microphone
(SDM) and IHM channels: four speakers as one host and three guests. The recordings have been split into a training
set of 12 programmes for DNN training only, and a test set of 10 episodes which has a total of 40 speakers and 8749 segments in 5.3 hours of speech time. The audio was manually transcribed to an accuracy of 0.1s
3. NIST Rich Transcription evaluation in 2007 & sre10 https://www.nist.gov/itl
4. IFLY-DIAR-II database which is drawn from Chinese talk shows, and the sample rate is 16 kHz. The duration of the recordings in the IFLY-DIAR-II database vary from 20 minutes to one hour. The number of speakers in each recording ranges from 2 to 9, and there are generally one host and several guests. The speaking style is spontaneous and causal, and short conversation turns and overlapped speech are often encountered. Furthermore, the speech is corrupted by music, laughter, applause, or other noises. The training set contains 171 recordings (86 hours), the development set consists of 90 conversations (47 hours), and the test set contains 367 audio files (193 hours).
5. REPERE 2013 data ESTER
6. development MGB Challenge data set

## 参考论文
