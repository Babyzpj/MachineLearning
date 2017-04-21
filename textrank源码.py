# encoding=utf-8  
import jieba  
import networkx as nx  
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer  
  
  
def cut_sentence(sentence):  
    """ 
    �־� 
    :param sentence: 
    :return: 
    """  
    if not isinstance(sentence, unicode):  
        sentence = sentence.decode('utf-8')  
    delimiters = frozenset(u'������')  
    buf = []  
    for ch in sentence:  
        buf.append(ch)  
        if delimiters.__contains__(ch):  
            yield ''.join(buf)  
            buf = []  
    if buf:  
        yield ''.join(buf)  
  
  
def load_stopwords(path='/home/fhqplzj/data/orion/stopwords.txt'):  
    """ 
    ����ͣ�ô� 
    :param path: 
    :return: 
    """  
    with open(path) as f:  
        stopwords = filter(lambda x: x, map(lambda x: x.strip().decode('utf-8'), f.readlines()))  
    stopwords.extend([' ', '\t', '\n'])  
    return frozenset(stopwords)  
  
  
def cut_words(sentence):  
    """ 
    �ִ� 
    :param sentence: 
    :return: 
    """  
    stopwords = load_stopwords()  
    return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))  
  
  
def get_abstract(content, size=3):  
    """ 
    ����textrank��ȡժҪ 
    :param content: 
    :param size: 
    :return: 
    """  
    docs = list(cut_sentence(content))  
    tfidf_model = TfidfVectorizer(tokenizer=jieba.cut, stop_words=load_stopwords())  
    tfidf_matrix = tfidf_model.fit_transform(docs)  
    normalized_matrix = TfidfTransformer().fit_transform(tfidf_matrix)  
    similarity = nx.from_scipy_sparse_matrix(normalized_matrix * normalized_matrix.T)  
    scores = nx.pagerank(similarity)  
    tops = sorted(scores.iteritems(), key=lambda x: x[1], reverse=True)  
    size = min(size, len(docs))  
    indices = map(lambda x: x[0], tops)[:size]  
    return map(lambda idx: docs[idx], indices)  
  
  
s = u'Ҫ˵���ڵ����90�����ǣ��ǾͲ��ò���¹�ϡ����ෲ�����������ˡ�����躡���ѧ�������Ȼ��2016�����Ǵ����������������Ӱ�Ӿ硣��Щ90�����ǲ�������ֵ���вŻ�������Ŭ����2017������������Щ��������Ʒ�أ�����˭���Ϊ2017����������������Ŀ�Դ��ɡ�¹��2016����ݡ���Ĺ�ʼǡ��������ǡ������ڶ��ˡ��ȶಿ��Ӱ��2017����������ת���˵��Ӿ硣���͹����������ݵĹ�װ��õ��Ӿ硶����ǡ����ں����������ڵ�����������¹�ϸ��˵��ײ����Ӿ磬Ҳ�����һ�γ��ݹ�װ��ġ��þ�ı���è���ͬ������С˵������������ħ����ļܿ�������³���(¹������)Ϊ���������������һֽ���������񶼣���ʶ��һȺ־ͬ���ϵ�С��飬�ڹ���ѧԺ��һƬ����ء����ෲ��2017���и������Ʒ�Ƴ������ǳۼ��ơ����ִ���Ĵ��ڵ������η�ħƪ�������ෲ���ݡ���ʷ������˧�ġ���ɮ��ʦͽ������ȡ����·�ϣ��ɻ���Կ���ͬ�ĺ�������Ϊ�޼᲻�ݵ���ħ�Ŷӡ����ෲ��������ΰ�����̺�������Ƭ��ŷ�޹��ԡ�����Ƭ��������������һ���ڶ���������(����ΰ��)����С��(������)������ѣ���������������(���ෲ��)�ֱ�׷�ٵ��ߡ��ϵ�֮�֡�����ɵ����շƣ�����ȴ������ŷ�޺ڰ����CIA��ŷ�˴�����������ع��ǵ�ȫ���Ѳ��Ĺ��¡����ෲ2017���ڵ�Ӱ�����и���ͻ�ƣ������˺������Ƭ�������ع�3���ռ��ع顷���뷶�������������ӵ������ȡ��Ų����һ�ڴ����Ǵ��Ϊ��Ӱ�׳���������JUICE�������⣬�����������ˡ�����ִ���Ŀƻõ�Ӱ���Ǽ��ع���ǧ��֮�ǡ�����Ƭ����һ��������δ��28�����Ǽʾ��촩Խʱ�յĹ��£�ӰƬ����2017����ӳ��'  
for i in get_abstract(s):  
    print i  


