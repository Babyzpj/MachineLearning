# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: PengjunZhu <1512568691@qq.com>
#
# Function:
#
# time:
#

# -*- coding: utf-8 -*-
#
# Copyright @2017 R&D, CINS Inc. (cins.com)
#
# Author: PengjunZhu <1512568691@qq.com>
#
# Function: 使用tf-idf实现获取文档摘要，并使用评估方法ROUGE-2、ROUGE-3进行评估
#      思路：使用IFIDF使用文档中问个句子的权值。然后抽取权值最大的句子作为摘要
# time: 2018.4.07

from __future__ import division
from dealMethods import readLcstsFile, cutWords


import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import jieba

"""
    具体实现步骤：
       1、对文档进行分句
       2、计算每个句子的TF值、IDF值、TF*IDF值
       3、再跟进IFIDF值对文档排序，抽取前三句作为摘要
"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    corpusAllList = ["我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
              "他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
              "小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
              "我 爱 北京 天安门"]  # 第四类文本的切词结果

    # path = 'D:\DataSet\PaperDataSet2018\LCSTS_curpus_3k.txt'
    # summaryAllList, corpusAllList = readLcstsFile(path)

    # curpusList = [cutWords(item) for item in corpusAllList]

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpusAllList))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重


    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"
        for j in range(len(word)):

            if weight[i][j] <= 0:
                continue
            print word[j], weight[i][j]

