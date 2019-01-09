# awesome-chinese-nlp
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of resources for NLP (Natural Language Processing) for Chinese

中文自然语言处理相关资料

图片来自复旦大学邱锡鹏教授

![](/images/1.jpg)
  
  
## Contents 列表

### 1. [Chinese NLP Toolkits 中文NLP工具](https://github.com/crownpku/awesome-chinese-nlp#chinese-nlp-toolkits-中文nlp工具)

* #### [Toolkits 综合NLP工具包](https://github.com/crownpku/awesome-chinese-nlp#toolkits-综合nlp工具包-1)
* #### [Popular NLP Toolkits for English/Multi-Language 常用的英文或支持多语言的NLP工具包](https://github.com/crownpku/awesome-chinese-nlp#popular-nlp-toolkits-for-englishmulti-language-常用的英文或支持多语言的nlp工具包-1)
* #### [Chinese Word Segment 中文分词](https://github.com/crownpku/awesome-chinese-nlp#chinese-word-segment-中文分词-1)
* #### [Information Extraction 信息提取](https://github.com/crownpku/awesome-chinese-nlp#information-extraction-信息提取-1)
* #### [QA & Chatbot 问答和聊天机器人](https://github.com/crownpku/awesome-chinese-nlp#qa--chatbot-问答和聊天机器人-1)
### 2. [Corpus 中文语料](https://github.com/crownpku/awesome-chinese-nlp#corpus-中文语料)
### 3. [Organizations 中文NLP学术组织及竞赛](https://github.com/crownpku/awesome-chinese-nlp#organizations-%E4%B8%AD%E6%96%87nlp%E5%AD%A6%E6%9C%AF%E7%BB%84%E7%BB%87%E5%8F%8A%E7%AB%9E%E8%B5%9B)
### 4. [Industry 中文NLP商业服务](https://github.com/crownpku/awesome-chinese-nlp#industry-%E4%B8%AD%E6%96%87nlp%E5%95%86%E4%B8%9A%E6%9C%8D%E5%8A%A1)
### 5. [Learning Materials 学习资料](https://github.com/crownpku/awesome-chinese-nlp#learning-materials-学习资料)
  
<br />
<br />

## Chinese NLP Toolkits 中文NLP工具

### Toolkits 综合NLP工具包

- [THULAC 中文词法分析工具包](http://thulac.thunlp.org/) by 清华 (C++/Java/Python)

- [NLPIR](https://github.com/NLPIR-team/NLPIR) by 中科院 (Java)

- [LTP 语言技术平台](https://github.com/HIT-SCIR/ltp) by 哈工大 (C++)  [pylyp](https://github.com/HIT-SCIR/pyltp) LTP的python封装

- [FudanNLP](https://github.com/FudanNLP/fnlp) by 复旦 (Java)

- [BaiduLac](https://github.com/baidu/lac) by 百度 Baidu's open-source lexical analysis tool for Chinese, including word segmentation, part-of-speech tagging & named entity recognition. 

- [HanLP](https://github.com/hankcs/HanLP) (Java)

- [SnowNLP](https://github.com/isnowfy/snownlp) (Python) Python library for processing Chinese text

- [YaYaNLP](https://github.com/Tony-Wang/YaYaNLP) (Python) 纯python编写的中文自然语言处理包，取名于“牙牙学语”

- [小明NLP](https://github.com/SeanLee97/xmnlp) (Python) 轻量级中文自然语言处理工具

- [DeepNLP](https://github.com/rockingdingo/deepnlp) (Python) Deep Learning NLP Pipeline implemented on Tensorflow with pretrained Chinese models.

- [chinese_nlp](https://github.com/taozhijiang/chinese_nlp) (C++ & Python) Chinese Natural Language Processing tools and examples

- [Chinese-Annotator](https://github.com/crownpku/Chinese-Annotator) (Python) Annotator for Chinese Text Corpus 中文文本标注工具

- [Poplar](https://github.com/synyi/poplar) (Typescript) A web-based annotation tool for natural language processing (NLP)

### Popular NLP Toolkits for English/Multi-Language 常用的英文或支持多语言的NLP工具包

- [CoreNLP](https://github.com/stanfordnlp/CoreNLP) by Stanford (Java) A Java suite of core NLP tools.

- [NLTK](http://www.nltk.org/) (Python) Natural Language Toolkit

- [spaCy](https://spacy.io/) (Python) Industrial-Strength Natural Language Processing

- [textacy](https://github.com/chartbeat-labs/textacy) (Python) NLP, before and after spaCy

- [OpenNLP](https://opennlp.apache.org/) (Java) A machine learning based toolkit for the processing of natural language text.

- [gensim](https://github.com/RaRe-Technologies/gensim) (Python) Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. 


  
### Chinese Word Segment 中文分词

- [Jieba 结巴中文分词](https://github.com/fxsjy/jieba) (Python及大量其它编程语言衍生) 做最好的 Python 中文分词组件

- [北大中文分词工具](https://github.com/lancopku/pkuseg-python) (Python) 高准确度中文分词工具，简单易用，跟现有开源工具相比大幅提高了分词的准确率。

- [kcws 深度学习中文分词](https://github.com/koth/kcws) (Python) BiLSTM+CRF与IDCNN+CRF

- [ID-CNN-CWS](https://github.com/hankcs/ID-CNN-CWS) (Python) Iterated Dilated Convolutions for Chinese Word Segmentation

- [Genius 中文分词](https://github.com/duanhongyi/genius) (Python) Genius是一个开源的python中文分词组件，采用 CRF(Conditional Random Field)条件随机场算法。

- [loso 中文分词](https://github.com/fangpenlin/loso) (Python)

- [yaha "哑哈"中文分词](https://github.com/jannson/yaha) (Python)

- [ChineseWordSegmentation](https://github.com/Moonshile/ChineseWordSegmentation) (Python) Chinese word segmentation algorithm without corpus（无需语料库的中文分词）

  
### Information Extraction 信息提取

- [MITIE](https://github.com/mit-nlp/MITIE) (C++) library and tools for information extraction

- [Duckling](https://github.com/facebookincubator/duckling) (Haskell) Language, engine, and tooling for expressing, testing, and evaluating composable language rules on input strings.

- [IEPY](https://github.com/machinalis/iepy) (Python)  IEPY is an open source tool for Information Extraction focused on Relation Extraction.

- [Snorkel](https://github.com/HazyResearch/snorkel) A training data creation and management system focused on information extraction 

- [Neural Relation Extraction implemented with LSTM in TensorFlow](https://github.com/thunlp/TensorFlow-NRE)

- [A neural network model for Chinese named entity recognition](https://github.com/zjy-ucas/ChineseNER)

- [Information-Extraction-Chinese](https://github.com/crownpku/Information-Extraction-Chinese) Chinese Named Entity Recognition with IDCNN/biLSTM+CRF, and Relation Extraction with biGRU+2ATT 中文实体识别与关系提取

- [Familia](https://github.com/baidu/Familia) 百度出品的 A Toolkit for Industrial Topic Modeling

- [Text Classification](https://github.com/brightmart/text_classification) All kinds of text classificaiton models and more with deep learning. 用知乎问答语聊作为测试数据。

  
### QA & Chatbot 问答和聊天机器人 

- [Rasa NLU](https://github.com/RasaHQ/rasa_nlu) (Python) turn natural language into structured data, a Chinese fork at [Rasa NLU Chi](https://github.com/crownpku/Rasa_NLU_Chi)

- [Rasa Core](https://github.com/RasaHQ/rasa_core) (Python) machine learning based dialogue engine for conversational software

- [Snips NLU](https://github.com/snipsco/snips-nlu) (Python) Snips NLU is a Python library that allows to parse sentences written in natural language and extracts structured information.

- [DeepPavlov](https://github.com/deepmipt/DeepPavlov) (Python) An open source library for building end-to-end dialog systems and training chatbots.

- [ChatScript](https://github.com/bwilcox-1234/ChatScript) Natural Language tool/dialog manager, a rule-based chatbot engine.

- [Chatterbot](https://github.com/gunthercox/ChatterBot) (Python) ChatterBot is a machine learning, conversational dialog engine for creating chat bots.

- [Chatbot](https://github.com/zake7749/Chatbot) (Python) 基於向量匹配的情境式聊天機器人

- [Tipask](https://github.com/sdfsky/tipask) (PHP) 一款开放源码的PHP问答系统，基于Laravel框架开发，容易扩展，具有强大的负载能力和稳定性。

- [QuestionAnsweringSystem](https://github.com/ysc/QuestionAnsweringSystem) (Java) 一个Java实现的人机问答系统，能够自动分析问题并给出候选答案。

- [QA-Snake](https://github.com/SnakeHacker/QA-Snake) (Python) 基于多搜索引擎和深度学习技术的自动问答

- [使用TensorFlow实现的Sequence to Sequence的聊天机器人模型](https://github.com/qhduan/Seq2Seq_Chatbot_QA) (Python)

- [使用深度学习算法实现的中文阅读理解问答系统](https://github.com/S-H-Y-GitHub/QA) (Python)

- [DuReader中文阅读理解Baseline代码](https://github.com/baidu/DuReader) (Python)

- [基于SmartQQ的自动机器人框架](https://github.com/Yinzo/SmartQQBot) (Python)

- [QASystemOnMedicalKG](https://github.com/liuhuanyong/QASystemOnMedicalKG) (Python) 以疾病为中心的一定规模医药领域知识图谱，并以该知识图谱完成自动问答与分析服务。

<br />
<br />

## Corpus 中文语料

- [开放知识图谱OpenKG.cn](http://openkg.cn)

- [开放中文知识图谱的schema](https://github.com/cnschema/cnschema)

- [大规模中文概念图谱CN-Probase](http://kw.fudan.edu.cn/cnprobase/search/) [公众号介绍](https://mp.weixin.qq.com/s?__biz=MzI0MTI1Nzk1MA==&mid=2651675884&idx=1&sn=1a43a93fd5bb53c8a9e48518bfa41db8&chksm=f2f7a05dc580294b227332b1051bfa2e5c756c72efb4d102c83613185b571ac31343720a6eae&mpshare=1&scene=1&srcid=1113llNDS1MvoadhCki83ERW#rd)

- [农业知识图谱](https://github.com/qq547276542/Agriculture_KnowledgeGraph) 农业领域的信息检索，命名实体识别，关系抽取，分类树构建，数据挖掘

- [CLDC中文语言资源联盟](http://www.chineseldc.org/)

- [中文 Wikipedia Dump](https://dumps.wikimedia.org/zhwiki/)

- [98年人民日报词性标注库@百度盘](https://pan.baidu.com/s/1gd6mslt)

- [搜狗20061127新闻语料(包含分类)@百度盘](https://pan.baidu.com/s/1bnhXX6Z)

- [UDChinese](https://github.com/UniversalDependencies/UD_Chinese) (for training spaCy POS)

- [中文word2vec模型](https://github.com/to-shimo/chinese-word2vec)

- [上百种预训练中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)

- [Tencent AI Lab Embedding Corpus for Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/embedding.html)

- [Synonyms:中文近义词工具包](https://github.com/huyingxi/Synonyms/) 基于维基百科中文和word2vec训练的近义词库，封装为python包文件。

- [Chinese_conversation_sentiment](https://github.com/z17176/Chinese_conversation_sentiment) A Chinese sentiment dataset may be useful for sentiment analysis.

- [中文突发事件语料库](https://github.com/shijiebei2009/CEC-Corpus) Chinese Emergency Corpus

- [dgk_lost_conv 中文对白语料](https://github.com/rustch3n/dgk_lost_conv) chinese conversation corpus

- [用于训练中英文对话系统的语料库](https://github.com/candlewill/Dialog_Corpus) Datasets for Training Chatbot System 

- [八卦版問答中文語料](https://github.com/zake7749/Gossiping-Chinese-Corpus)

- [中国股市公告信息爬取](https://github.com/startprogress/China_stock_announcement) 通过python脚本从巨潮网络的服务器获取中国股市（sz,sh）的公告(上市公司和监管机构)

- [tushare财经数据接口](http://tushare.org/) TuShare是一个免费、开源的python财经数据接口包。

- [保险行业语料库](https://github.com/Samurais/insuranceqa-corpus-zh)   [[52nlp介绍Blog](http://www.52nlp.cn/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%BF%9D%E9%99%A9%E8%A1%8C%E4%B8%9A%E9%97%AE%E7%AD%94%E5%BC%80%E6%94%BE%E6%95%B0%E6%8D%AE%E9%9B%86)] OpenData in insurance area for Machine Learning Tasks

- [最全中华古诗词数据库](https://github.com/chinese-poetry/chinese-poetry) 唐宋两朝近一万四千古诗人, 接近5.5万首唐诗加26万宋诗. 两宋时期1564位词人，21050首词。

- [DuReader中文阅读理解数据](http://ai.baidu.com/broad/subordinate?dataset=dureader) 

- [中文语料小数据](https://github.com/crownpku/Small-Chinese-Corpus) 包含了中文命名实体识别、中文关系识别、中文阅读理解等一些小量数据

- [中文人名语料库](https://github.com/wainshine/Chinese-Names-Corpus) 中文姓名,姓氏,名字,称呼,日本人名,翻译人名,英文人名。

- [中文敏感词词库](https://github.com/observerss/textfilter) 敏感词过滤的几种实现+某1w词敏感词库

- [中文简称词库](https://github.com/zhangyics/Chinese-abbreviation-dataset) A corpus of Chinese abbreviation, including negative full forms.   

- [中文数据预处理材料](https://github.com/dongxiexidian/Chinese) 中文分词词典和中文停用词

- [漢語拆字字典](https://github.com/kfcd/chaizi)

- [SentiBridge: 中文实体情感知识库](https://github.com/rainarch/SentiBridge) 刻画人们如何描述某个实体，包含新闻、旅游、餐饮，共计30万对。

- [OpenCorpus](https://github.com/hankcs/OpenCorpus) A collection of freely available (Chinese) corpora. 

- [ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus) 情感/观点/评论 倾向性分析，中文命名实体识别，推荐系统
<br />
<br />

## Organizations 中文NLP学术组织及竞赛

- [清华大学自然语言处理与人文计算实验室](http://nlp.csai.tsinghua.edu.cn/site2/index.php/zh)

- [北京大学计算语言学教育部重点实验室](http://klcl.pku.edu.cn/)

- [中科院计算所自然语言处理研究组](http://www.nlpir.org/)

- [哈工大智能技术与自然语言处理实验室](http://insun.hit.edu.cn/)

- [复旦大学自然语言处理组](http://nlp.fudan.edu.cn/)

- [苏州大学自然语言处理组](http://nlp.suda.edu.cn/index.html)

- [南京大学自然语言处理研究组](http://nlp.nju.edu.cn)

- [东北大学自然语言处理实验室](http://www.nlplab.com/)

- [厦门大学智能科学与技术系自然语言处理实验室](http://nlp.xmu.edu.cn/)

- [郑州大学自然语言处理实验室](http://nlp.zzu.edu.cn/)

- [微软亚洲研究院自然语言处理](https://www.msra.cn/zh-cn/research/nlp)

- [华为诺亚方舟实验室](http://www.noahlab.com.hk/)

- [CUHK Text Mining Group](http://www1.se.cuhk.edu.hk/~textmine/)

- [PolyU Social Media Mining Group](http://www4.comp.polyu.edu.hk/~cswjli/Group.html)

- [HKUST Human Language Technology Center](http://www.cse.ust.hk/~hltc/)

- [National Taiwan University NLP Lab](http://nlg.csie.ntu.edu.tw/)

- [中国中文信息学会](http://www.cipsc.org.cn/)

- [NLP Conference Calender](http://cs.rochester.edu/~omidb/nlpcalendar/) Main conferences, journals, workshops and shared tasks in NLP community.

- [2017 第一届“讯飞杯”中文机器阅读理解评测](http://www.cips-cl.org/static/CCL2017/iflytek.html)

- [2017 AI-Challenger 图像中文描述](https://www.challenger.ai/competition/caption) 用一句话描述给定图像中的主要信息，挑战中文语境下的图像理解问题。

- [2017 AI-Challenger 英中机器文本翻译](https://www.challenger.ai/competition/translation) 用大规模的数据，提升英中文本机器翻译模型的能力。

- [2017 知乎看山杯机器学习挑战赛](https://biendata.com/competition/zhihu/) 根据知乎给出的问题及话题标签的绑定关系的训练数据，训练出对未标注数据自动标注的模型。

- [2018 开放领域的中文问答任务](https://biendata.com/competition/CCKS2018_4/) 对于给定的一句中文问题，问答系统从给定知识库中选择若干实体或属性值作为该问题的答案。

- [2018 微众银行智能客服问句匹配大赛](https://biendata.com/competition/CCKS2018_3/) 针对中文的真实客服语料，进行问句意图匹配；给定两个语句，判定两者意图是否相近。
  
<br />
<br />

## Industry 中文NLP商业服务

- [百度云NLP](https://cloud.baidu.com/product/nlp.html) 提供业界领先的自然语言处理技术，提供优质文本处理及理解技术

- [阿里云NLP](https://data.aliyun.com/product/nlp) 为各类企业及开发者提供的用于文本分析及挖掘的核心工具

- [腾讯云NLP](https://cloud.tencent.com/product/nlp) 基于并行计算、分布式爬虫系统，结合独特的语义分析技术，一站满足NLP、转码、抽取、数据抓取等需求

- [讯飞开放平台](https://www.xfyun.cn/) 以语音交互为核心的人工智能开放平台

- [搜狗实验室](http://www.sogou.com/labs/webservice/) 分词和词性标注

- [玻森数据](http://bosonnlp.com/) 上海玻森数据科技有限公司，专注中文语义分析技术

- [云孚科技](https://www.yunfutech.com/) NLP工具包、知识图谱、文本挖掘、对话系统、舆情分析等

- [智言科技](http://www.webot.ai) 专注于深度学习和知识图谱技术突破的人工智能公司

- [追一科技](https://zhuiyi.ai/) 主攻深度学习和自然语言处理

<br />
<br />

## Learning Materials 学习资料

- [中文Deep Learning Book](https://github.com/exacity/deeplearningbook-chinese)

- [Stanford CS224n Natural Language Processing with Deep Learning 2017](http://web.stanford.edu/class/cs224n/syllabus.html)

- [Oxford CS DeepNLP 2017](https://github.com/oxford-cs-deepnlp-2017)

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Dan Jurafsky and James H. Martin

- [52nlp 我爱自然语言处理](http://www.52nlp.cn/)

- [hankcs 码农场](http://www.hankcs.com/)

- [文本处理实践课资料](https://github.com/Roshanson/TextInfoExp) 文本处理实践课资料，包含文本特征提取（TF-IDF），文本分类，文本聚类，word2vec训练词向量及同义词词林中文词语相似度计算、文档自动摘要，信息抽取，情感分析与观点挖掘等实验。

- [nlp_tasks](https://github.com/Kyubyong/nlp_tasks) Natural Language Processing Tasks and Selected References

