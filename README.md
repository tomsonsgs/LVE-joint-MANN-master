# LVE-joint-MANN-master
1，本任务为ECPE任务，即情感-诱因对抽取任务，本文极少数数据预处理代码参考了ECPE原始论文的代码

2，glove词向量文件可自行下载，但这个对效果提升有限，可以不使用而注释掉

3，ECPE原始论文的核心为两阶段模型，即第一阶段预测情感诱因句子，第二阶段模型中两两组合之前预测的情感诱因候选得到情感-诱因对候选，再进一步细筛情感诱因对

3，原始两阶段模型比直接的单阶段配对模型提升明显，本代码提出了改进后的两阶段模型，比原始两阶段模型更强，具体思路见代码

4，以十折中的第二折数据集为例，原始代码F1值可以跑到61.5，本代码可跑到63.9

5，尝试过层级的BERT（每个句子对应一个）对情感分类有一定提升，但对ECPE提升不大，可尝试文档直接拼接成单文本而不是依照多个句子，这样bert或许可以刻画情感句子和诱因句子的内部联系以达到提升的效果

6，本文提出的多层次注意力机制对ECPE有一定提升，主要在于刻画了诱因句子映射到情感句子的内部联系，加强了配对效果

7，训练步骤为先训练ecmodel，即情感诱因句子识别模型，判断是否为诱因句子或者情感句子，接着训练配对模型并利用到ecmodel的预测结果转化的绝对标签信息

8，对下一步研究的建议：1，利用BERT编码整个单文本而不是基于多个句子 2，最好基于本代码提出的改进两阶段模型  3，考虑其他比句子配对的二元分类loss更好的loss函数
