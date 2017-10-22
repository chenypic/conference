# CVPR2017



[Fine-Tuning Convolutional Neural Networks for Biomedical Image Analysis: Actively and Incrementally](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Fine-Tuning_Convolutional_Neural_CVPR_2017_paper.html)

- Intense interest in applying convolutional neural networks (CNNs) in biomedical image analysis is wide spread, but its success is impeded by the lack of large annotated datasets in biomedical imaging. Annotating biomedical images is not only tedious and time consuming, but also demanding of costly, specialty - oriented knowledge and skills, which are not easily accessible. To dramatically reduce annotation cost, this paper presents a novel method called AIFT (active, incremental fine-tuning) to naturally integrate active learning and transfer learning into a single framework. AIFT starts directly with a pre-trained CNN to seek "worthy" samples from the unannotated for annotation, and the (fine-tuned) CNN is further fine-tuned continuously by incorporating newly annotated samples in each iteration to enhance the CNN's performance incrementally. We have evaluated our method in three different biomedical imaging applications, demonstrating that the cost of annotation can be cut by at least half. This performance is attributed to the several advantages derived from the advanced active and incremental capability of our AIFT method.
- AIFT (active, incremental fine-tuning) 整合主动学习 (active learning) 和迁移学习 (transfer learning).




这篇主要针对医学图像处理领域标注数据匮乏的问题，如何通过卷积神经网络（CNN）的Fine-tune和主动学习（Active Learning）来解决。使用CNN进行生物医学图像分析在最近几年得到了比较多的关注，但面临的一个问题是缺乏大量的标注数据，相比imagenet，对医学图像进行标注需要大量的专业背景知识，为了节约标注的成本和时间，这篇论文提供了一个新型的方法AIFT（Active，Incremental Fine-Tuning），把主动学习和迁移学习集成到一个框架。AIFT算法开始是直接使用一个预训练从未标注数据里找一些比较值得标注的样本，然后模型持续的加入新标注的数据，一直做微调。

AIFT方法是在CAD（计算机辅助诊断）系统的环境下使用，CAD可以生成候选集U，都是未标注数据，其中每一个候选样本（candidate）通过数据增强可以生成一系列的patches，由于这些patches来自于同一个候选样本，所以它们的标签跟该候选样本一致。

**AIFT方法的主要创新点体现在如下几个方面：**

- **持续性的fine-tuning**

一开始标注数据集L是空的，我们拿一个已经训练好了的CNN（比如AlexNet），让它在未标注数据集U中选b个候选集来找医生标注，这新标注的候选集将会放到标注数据集L中，来持续的增量式fine-tune那个CNN直到合格，通过实验发现，持续的fine-tuning CNN相比在原始的预训练中重复性的fine-tuning CNN，可以让数据集收敛更快。

- ** 通过Active learning选择候选样本**

主动学习的关键是找到一个标准来评判候选样本是否值得标注，在当前CNN中，一个候选样本生成的所有patches都应该是有差不多的预测。所以我们可以先通过这个CNN来对每个候选样本的每个patch进行预测，然后对每个候选样本，通过计算patch的熵和patch之间KL距离来衡量这个候选样本。如果熵越高，说明包含更多的信息，如果KL距离越大，说明patch间的不一致性大，所以这两个指标越高，越有可能对当前的CNN优化越大。对每个矩阵都可以生成一个包含patch的KL距离和熵的邻接矩阵R。

- **通过少数服从多数来处理噪音**

我们普遍都会使用一些自动的数据增强的方法，来提高CNN的表现，但是不可避免的给某些候选样本生成了一些难的样本，给数据集注入了一些噪音。所以为了显著的提高我们方法的鲁棒性，我们依照于当前CNN的预测，对每个候选样本只选择一部分的patch来计算熵和多样性。首先对每个候选样本的所有patch，计算平均的预测概率，如果平均概率大于0.5，我们只选择概率最高的部分patch，如果概率小于0.5，选最低的部分patch，再基于已经选择的patch，来构建得分矩阵R。

- **预测出的结果有不同的模式**

对每个候选样本进行计算所有补丁的概率分布直方图，对于概率的分布有以下几种模式：

1、patch大部分集中在0.5，不确定性很高，大多数的主动学习算法都喜欢这种候选集。

2、比a还更好，预测从0-1分布均匀，导致了更高的不确定性，因为所有的patch都是通过同一个候选集数据增强得到，他们理论上应该要有差不多的预测。这种类型的候选集有明显优化CNN模型的潜力。

3、预测分布聚集在两端，导致了更高的多样性，但是很有可能和patch的噪声有关，这是主动学习中最不喜欢的样本，因为有可能在fine-tuning的时候迷惑CNN。

4、预测分布集中在一端（0或1），包含更高的确定性，这类数据的标注优先级要降低，因为当前模型已经能够很好的预测它们了。

5、在某些补丁的预测中有更高的确定性，并且有些还和离群点有关联，这类候选集是有价值的，因为能够平滑的改善CNN的表现，尽管不能有显著的贡献，但对当前CNN模型不会有任何伤害。

**应用的创新：**

上述方法被应用在了结肠镜视频帧分类和肺栓塞检测上，得到了比较好的效果。前者只用了800个候选样本就达到了最好的表现，只用了5%的候选样本就代表了剩下的候选样本，因为连续的视频帧通常都差不多。后者使用了1000个样本就达到了AlexNet做Fine-tune使用2200个随机样本的效果。

**该工作的主要优势包括如下几点：**

1、从一个完全未标注的数据集开始，不需要初始的种子标注数据。

2、通过持续的fine-tuning而不是重复的重新训练来一步一步改善学习器。

3、通过挖掘每一个候选样本的补丁的一致性来选择值得标注的候选集。

4、自动处理噪音

5、只对每个候选集中小数量的补丁计算熵和KL距离，节约了计算。

总结下来，该工作提出的方法显著的减低标注的工作量，并且有指导的选择哪些数据需要标注，同时降低了数据增强带来的噪声影响。这个方向在医学图像处理领域有非常大的价值，相信会得到越来越多的关注。





参考资料：

1. https://www.leiphone.com/news/201707/GrFoDuRwYNpttISb.html



