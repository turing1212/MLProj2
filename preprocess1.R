###################### step 1: Load everything we need #########################
#### load packages ####
library(tidyverse)
library(tidytext)
library(jiebaR)
library(wordcloud)
library(reshape2)
library(word2vec)
library(mlr)
#### load data ####
rawDataPath <- "D:/Rexercise/MLProj2/rawdata"
rawDataList <- list.files(rawDataPath)
reviewData <- read_csv(paste0(rawDataPath, "/", rawDataList[5]), 
                       locale=locale(encoding="UTF-8"))
reviewData

# 计算没有图片的评论的数目
sum(is.na(reviewData$rate_pic_url))/length(reviewData$rate_pic_url) > 0.9 

################## step 2: set labels by affection analysis ####################
# 只使用评论数据
text <- reviewData %>% 
    mutate(reviewid = seq(1,length(feedback))) %>%
    select(reviewid, feedback)
text$feedback <- gsub(" ", "", text$feedback)
text
#### 将整句评论按标点符号拆成片段 ####
pieces <- strsplit(text$feedback,",|\\.|!|\\?|;|~|，|。|！！|！！！|！|\\？|；|～|…|﹏﹏|  |。。。。。。|\\.\\.\\.\\.\\.\\.")
temp <- unlist(lapply(pieces, length)) # 计算每一条评论被拆成了几段
id <- rep(text$reviewid, temp) # 生成标签序列 x-y 中的x序列
term <- unlist(pieces) # 生成片段序列

groupid <- function(x) { # 生成标签序列 x-y 中的y序列
    subid <- seq(1:x)
    return(subid)
}
subid<-paste0(id, "-", unlist(lapply(temp, groupid)), seq="")
piecedata <- tibble(id = id, term = term, subid = subid)
piecedata

#### 分词 ####
wk <- worker(bylines = TRUE)
wordlist <- wk[piecedata$term] # 分词，生成一个包含分词的list
head(wordlist, 10)

temp_fc <- unlist(lapply(wordlist, length)) # 计算每一段被分成了几个词
id_fc <- rep(piecedata$id, temp_fc) # 生成标签序列
subid_fc <- rep(piecedata$subid, temp_fc) # 生成标签序列
term_fc <- unlist(wordlist)

fc <- tibble(id = id_fc, term = term_fc, subid = subid_fc)
fc

#### 情感分析 ####
#导入情感词典
dicPath <- "D:/Rexercise/MLProj2/dict" # 存放词典的路径
dicList <- list.files(dicPath) #列出该路径下所有文件
positive <- readLines(paste0(dicPath, "/", "posWord.txt"), 
                      encoding="GBK") #读入积极词
negative <- readLines(paste0(dicPath, "/", "negWord.txt"), 
                      encoding="GBK") #读入消极词
sameword1 <- intersect(positive,negative) #求两个词典的交集，做这一步是为了防止两个词典有相同的单词
pos1 <- setdiff(positive,sameword1) #去重
neg1 <- setdiff(negative,sameword1) #去重
pos <- tibble(term = pos1, weight = rep(1,length(pos1))) 
neg <- tibble(term = neg1, weight = rep(-1,length(neg1)))
posneg <- rbind(pos,neg) # 合并词典
posneg$term <- gsub(" ", "", posneg$term) #删去空格
posneg
which(posneg$term=="满意") #查一下满意这个词在词典哪里

# 关联情感词
fc_posneg <- left_join(fc, posneg, by = "term")
fc_posneg

sum(fc_posneg$weight==1, na.rm = TRUE) #属于积极词典的词的数目
sum(fc_posneg$weight==-1, na.rm = TRUE) #属于消极词典的词的数目
sum(is.na(fc_posneg$weight)) #不属于这两个词典的词的数目

# 计算情感得分
fc_posneg <- fc_posneg %>%
    mutate(score = weight, sentence = as.factor(id))
fc_posneg$score[is.na(fc_posneg$weight)] <- 0
fc_posneg

emoScore <- fc_posneg %>%
    group_by(sentence) %>%
    summarise(affectScore = sum(score))
emoScore
sum(emoScore$affectScore>0) # 好评数量
sum(emoScore$affectScore==0) # 中评数量
sum(emoScore$affectScore<0) # 差评数量

text$sentence <- as.factor(text$reviewid) 
text_labeled <- left_join(text, emoScore, by = "sentence") # 合并标签和评论
text_labeled
which(is.na(text_labeled$affectScore))

#### 画词云图 ####
# all worlds
fc %>% count(term) %>% with(wordcloud(term, n, max.words = 100))

# 分情感
fc_posneg$sentiment <- fc_posneg$weight
fc_posneg$sentiment[which(fc_posneg$weight==1)] <- "positive"
fc_posneg$sentiment[which(fc_posneg$weight==-1)] <- "negative"
fc_posneg %>% mutate(sentiment = as.factor(sentiment)) %>%
    filter(!is.na(weight)) %>%
    count(term, sentiment) %>%
    acast(term ~ sentiment, value.var = "n", fill = 0) %>%
    comparison.cloud(colors = c("gray80", "gray20"), max.words = 100)

###################### step 3: extract review features #########################
#### word2vec ####
segmented_text <- stringr::str_c(fc$term, collapse = " ") %>% c()
readr::write_file(segmented_text, file = './segmented.txt')
# generate 10-dim word vector
model10 <- word2vec(x = './segmented.txt', 
                  dim = 10,  
                  iter = 50, 
                  split = c(" ",  "。？！；、～"),
                  threads = parallel::detectCores()) #并行，使用cpu多核加速
emb10 <- as.matrix(model10)

# generate 15-dim word vector
model15 <- word2vec(x = './segmented.txt', 
                    dim = 15,  
                    iter = 50, 
                    split = c(" ",  "。？！；、～"),
                    threads = parallel::detectCores()) #并行，使用cpu多核加速
emb15 <- as.matrix(model15)

# generate 20-dim word vector
model20 <- word2vec(x = './segmented.txt', 
                    dim = 20,  
                    iter = 50, 
                    split = c(" ",  "。？！；、～"),
                    threads = parallel::detectCores()) #并行，使用cpu多核加速

emb20 <- as.matrix(model20)
#predict(model, '漂亮', type='nearest', top_n = 10)

generateModel <- function(filedir, ndim, niter) {
    model <- word2vec(x = filedir, 
                        dim = ndim,  
                        iter = niter, 
                        split = c(" ",  "。？！；、～"),
                        threads = parallel::detectCores())
    return(model)
}



#### generate ML data ####
generateMLdata <- function(model, mydata) {
    f2vec <- function(x) {
        y <- doc2vec(object = model, newdata = x, split=' ')
        return(y)
    }
    vecdata <- sapply(as.list(mydata$feedback), f2vec)
    vecdatat <- as_tibble(t(vecdata))
    nf <- dim(vecdatat)[2]
    nameseq <- lapply(seq(1, nf), function(x){paste0("f",x)})
    names(vecdatat) <- nameseq
    vecdatat$reviewid <- seq(1,length(vecdatat$f1))
    vecdatat$sentence <- as.factor(vecdatat$reviewid)
    
    return(vecdatat)
}

vecdatat10 <- generateMLdata(model10, reviewData) 
vecdatat15 <- generateMLdata(model15, reviewData) 
vecdatat20 <- generateMLdata(model20, reviewData) 


#### 合并数据 ####
mergedata <- function(vecx, emox, labelnum = 3) {
    MLdata <- left_join(vecx, emox, by = "sentence")
    nf <- dim(vecx)[2] - 2
    if (labelnum == 3) {
        MLdata$label[which(MLdata$affectScore>0)] <- "good"
        MLdata$label[which(MLdata$affectScore==0)] <- "neutral"
        MLdata$label[which(MLdata$affectScore<0)] <- "bad"
    } else {
        MLdata$label[which(MLdata$affectScore>0)] <- "good"
        MLdata$label[which(MLdata$affectScore<=0)] <- "not good"
    }

    finaldata <- MLdata %>%
        filter(!is.na(label)) %>%
        mutate(label = as.factor(label)) %>%
        select(names(vecx)[1:nf], label)
    return(finaldata)
}

finaldata10 <- mergedata(vecdatat10, emoScore, 3)
finaldata15 <- mergedata(vecdatat15, emoScore, 3)
finaldata20 <- mergedata(vecdatat20, emoScore, 3)

###################### step 4: naive bayes analysis #########################
#### 朴素贝叶斯分类器 ####
reviewTask <- makeClassifTask(data = finaldata, target = "label")
bayes <- makeLearner("classif.naiveBayes")
bayesModel <- train(bayes, reviewTask)

kFold <- makeResampleDesc(method = "RepCV", folds = 10, 
                          rep = 50, stratify = TRUE)
bayesCV <- resample(learner = bayes, task = reviewTask,
                    resampling = kFold,
                    measures = list(mmce, acc))
bayesCV$aggr

bayesClassifier <- function(mydata, n_fold, n_rep) {
    reviewTask <- makeClassifTask(data = mydata, target = "label")
    bayes <- makeLearner("classif.naiveBayes")
    kFold <- makeResampleDesc(method = "RepCV", folds = n_fold, 
                              rep = n_rep, stratify = TRUE)
    bayesCV <- resample(learner = bayes, task = reviewTask,
                        resampling = kFold,
                        measures = list(mmce, acc))
    return(bayesCV$aggr)
}

Result1 <- bayesClassifier(finaldata10, 10, 50)
Result2 <- bayesClassifier(finaldata15, 10, 50)
Result3 <- bayesClassifier(finaldata20, 10, 50)

#bayesModel <- train(bayes, reviewTask)

#################### 补充: 加入停用词 #########################
# 分词时加入stopword
stopword <- "D:/Rexercise/MLProj2/dict/stopwords.txt"
wk_stopword <- worker(stop_word = stopword,bylines = TRUE)
wordlist_s <- wk_stopword[piecedata$term] # 分词，生成一个包含分词的list
head(wordlist_s, 10)

temp_fc_s <- unlist(lapply(wordlist_s, length)) # 计算每一段被分成了几个词
id_fc_s <- rep(piecedata$id, temp_fc_s) # 生成标签序列
subid_fc_s <- rep(piecedata$subid, temp_fc_s) # 生成标签序列
term_fc_s <- unlist(wordlist_s)

fc_s <- tibble(id = id_fc_s, term = term_fc_s, subid = subid_fc_s)
fc_s
# 关联情感词
fc_posneg_s <- left_join(fc_s, posneg, by = "term")
fc_posneg_s

sum(fc_posneg_s$weight==1, na.rm = TRUE) #属于积极词典的词的数目
sum(fc_posneg_s$weight==-1, na.rm = TRUE) #属于消极词典的词的数目
sum(is.na(fc_posneg_s$weight)) #不属于这两个词典的词的数目

# 计算情感得分
fc_posneg_s <- fc_posneg_s %>%
    mutate(score = weight, sentence = as.factor(id))
fc_posneg_s$score[is.na(fc_posneg_s$weight)] <- 0
fc_posneg_s

emoScore_s <- fc_posneg_s %>%
    group_by(sentence) %>%
    summarise(affectScore = sum(score))
emoScore_s
sum(emoScore_s$affectScore>0) # 好评数量
sum(emoScore_s$affectScore==0) # 中评数量
sum(emoScore_s$affectScore<0) # 差评数量

text$sentence <- as.factor(text$reviewid) 
text_labeled_s <- left_join(text, emoScore_s, by = "sentence") # 合并标签和评论
text_labeled_s
which(is.na(text_labeled_s$affectScore))

#### 画词云图 ####
# all worlds
fc_s %>% count(term) %>% with(wordcloud(term, n, max.words = 100))

# 分情感
fc_posneg_s$sentiment <- fc_posneg_s$weight
fc_posneg_s$sentiment[which(fc_posneg_s$weight==1)] <- "positive"
fc_posneg_s$sentiment[which(fc_posneg_s$weight==-1)] <- "negative"
fc_posneg_s %>% mutate(sentiment = as.factor(sentiment)) %>%
    filter(!is.na(weight)) %>%
    count(term, sentiment) %>%
    acast(term ~ sentiment, value.var = "n", fill = 0) %>%
    comparison.cloud(colors = c("gray80", "gray20"), max.words = 100)

#### 提取特征 ####
segmented_text_s <- stringr::str_c(fc_s$term, collapse = " ") %>% c()
readr::write_file(segmented_text_s, file = './segmenteds.txt')
model_s10 <- generateModel('./segmenteds.txt', 10, 50)
model_s15 <- generateModel('./segmenteds.txt', 15, 50)
model_s20 <- generateModel('./segmenteds.txt', 20, 50)
emb_s10 <- as.matrix(model_s10)

#### 生成特征向量 ####
vecdatat10_s <- generateMLdata(model_s10, reviewData) 
vecdatat15_s <- generateMLdata(model_s15, reviewData) 
vecdatat20_s <- generateMLdata(model_s20, reviewData) 

#### 合并数据 ####
finaldata10_s <- mergedata(vecdatat10_s, emoScore_s, 3)
finaldata15_s <- mergedata(vecdatat15_s, emoScore_s, 3)
finaldata20_s <- mergedata(vecdatat20_s, emoScore_s, 3)

#### 模型训练 ####
Result1_s <- bayesClassifier(finaldata10_s, 10, 50)
Result2_s <- bayesClassifier(finaldata15_s, 10, 50)
Result3_s <- bayesClassifier(finaldata20_s, 10, 50)