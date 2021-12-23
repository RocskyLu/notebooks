# Transformer

#### RNN

+ ##### RNN

  + RNN非常擅长于处理输入是一个sequence的情况
  + RNN很不容易并行化
  + CNN取代RNN其本质可以理解成滑动窗思想，考虑的内容非常有限，

+ ##### LSTM

+ ##### GRU

#### Transformer

+ ##### Self Attention

  + d就是channel数
  + attention 是k'*q, 输出是v * A， 要是反过来，就是q' * k， A * v
  + position encoding就是长度为N的one-hot编码经过dxN的矩阵变换得到的，所以直接加上即可
  + CNN是特殊的self-attention，只能近距离的attention
  + layer norm，每一个样本单独计算； batch norm，每一个channel单独计算
  + Add表示残差连接，防止网络退化
  + Decoder中第一个attention是masked， 第二个是普通attention

+ ##### Multi-Head Attention

  + Multi-head可以理解为attention做多次，每次注意的点不一样

+ ##### Masked Multi-Head Attention

  + attention只会关注已经产生的sequence，而不会关注没有产生出来的东西
  + mask操作在scale之后，softmax之前， 下三角的mask则是q' * k
  + 

+ ##### 理解

  + ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BKey%2C+Value%7D) 来自Transformer Encoder的输出，所以可以看做**句子(Sequence)/图片(image)**的**内容信息(content，比如句意是："我有一只猫"，图片内容是："有几辆车，几个人等等")**
  + ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BQuery%7D) 表达了一种诉求：希望得到什么，可以看做**引导信息(guide)**
  + 通过Multi-Head Self-attention结合在一起的过程就相当于是**把我们需要的内容信息指导表达出来**

#### Vision Transformer

+ ##### DETR

  + **将transformers运用到了object detection领域，取代了现在的模型需要手工设计的工作，并且取得了不错的结果。**在object detection上DETR准确率和运行时间上和Faster RCNN相当；将模型 generalize 到 panoptic segmentation 任务上，DETR表现甚至还超过了其他的baseline。DETR第一个使用End to End的方式解决检测问题，解决的方法是把检测问题视作是一个set prediction problem
  + CNN + Transformer，显式地对一个序列中的所有elements两两之间的interactions进行建模，使得这类transformer的结构非常适合带约束的set prediction的问题
  +  DETR的特点是：一次预测，端到端训练，set loss function和二分匹配
  + **第一点不同的是**，原版Transformer只考虑 ![[公式]](https://www.zhihu.com/equation?tex=x) 方向的位置编码，但是DETR考虑了 ![[公式]](https://www.zhihu.com/equation?tex=xy) 方向的位置编码，因为图像特征是2-D特征
  + **另一点不同的是，原版Transformer**只在Encoder之前使用了Positional Encoding，而且是**在输入上进行Positional Encoding，再把输入经过transformation matrix变为Query，Key和Value这几个张量**， 但是DETR**在Encoder的每一个Multi-head Self-attention之前都使用了Positional Encoding，且**只对Query和Key使用了Positional Encoding，即：只把维度为![[公式]](https://www.zhihu.com/equation?tex=%28HW%2CB%2C256%29) 维的位置编码与维度为![[公式]](https://www.zhihu.com/equation?tex=%28HW%2CB%2C256%29) 维的Query和Key相加，而不与Value相加
  + Decoder， **decodes the N objects in parallel at each decoder layer**

+ 



