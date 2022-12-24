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

#### generate ML data ####
generateMLdata <- function(model, mydata) {
    f2vec <- function(x) {
        y <- doc2vec(object = model, newdata = x, split=' ')
        return(y)
    }
    vecdata <- sapply(as.list(mydata$feedback), f2vec)
    vecdatat <- as_tibble(t(vecdata))
    names(vecdatat) <- c("f1", "f2", "f3", "f4", "f5",
                         "f6", "f7", "f8", "f9", "f10")
    vecdatat$reviewid <- seq(1,length(vecdatat$f1))
    vecdatat$sentence <- as.factor(vecdatat$reviewid)
    
    return(vecdatat)
}

vecdatat10 <- generateMLdata(model10, reviewData) 


f2vec <- function(x) {
    y <- doc2vec(object = model, newdata = x, split=' ')
    return(y)
}

vecdata <- sapply(as.list(reviewData$feedback), f2vec)
vecdatat <- as_tibble(t(vecdata))
names(vecdatat) <- c("f1", "f2", "f3", "f4", "f5",
                     "f6", "f7", "f8", "f9", "f10")
vecdatat$reviewid <- seq(1,length(vecdatat$f1))
vecdatat$sentence <- as.factor(vecdatat$reviewid)

vecdatat
#### 合并数据 ####
MLdata <- left_join(vecdatat, emoScore, by = "sentence")
MLdata$label[which(MLdata$affectScore>0)] <- "good"
MLdata$label[which(MLdata$affectScore==0)] <- "neutral"
MLdata$label[which(MLdata$affectScore<0)] <- "bad"
finaldata <- MLdata %>%
    filter(!is.na(label)) %>%
    mutate(label = as.factor(label)) %>%
    select(f1:f10, label)
finaldata

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

#### 合并数据2 ####
MLdata2 <- left_join(vecdatat, emoScore, by = "sentence")
MLdata2$label[which(MLdata$affectScore>0)] <- "good"
MLdata2$label[which(MLdata$affectScore<=0)] <- "not good"
finaldata2 <- MLdata2 %>%
    filter(!is.na(label)) %>%
    mutate(label = as.factor(label)) %>%
    select(f1:f10, label)
finaldata2
#### 朴素贝叶斯分类器 ####
reviewTask2 <- makeClassifTask(data = finaldata2, target = "label")
bayesModel2 <- train(bayes, reviewTask2)


bayesCV2 <- resample(learner = bayes, task = reviewTask2,
                    resampling = kFold,
                    measures = list(mmce, acc, fpr, fnr))
bayesCV2$aggr



###################### step 5: 加入停用词 #########################
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
text_labeled <- left_join(text, emoScore, by = "sentence") # 合并标签和评论
text_labeled
which(is.na(text_labeled$affectScore))

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
