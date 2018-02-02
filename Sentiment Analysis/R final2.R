install.packages("ggplot2")
install.packages("wordcloud")
install.packages("dplyr")
install.packages("tidytext")
install.packages("tidyr")
install.packages("stringr")

library(readxl)
Consumer_Complaints2 <- read_excel("C:/Users/iamka/Desktop/Quarter 3/Hadoop/Final Project/Abishek's project/wetransfer-662912/Consumer_Complaints2.xlsx", 
                                   col_names = FALSE)
View(Consumer_Complaints2)

library(dplyr)
data<- data_frame(line = 1:66806, text = Consumer_Complaints2$X__1)
missing_line <- data_frame(line = 1:66806, sentiment1 = rep(0,66806))

#tokenized each word in each line
library(tidytext)
# Tokenized:text is the colname of text in data
# Remove stop_words
data("stop_words")
tidydata <- data %>% unnest_tokens(word, text) %>% 
  anti_join(stop_words) %>%
  count(word, sort = TRUE)

#visualizeted words ranking
#You can leave this step

library(ggplot2)
tidydata %>% filter(n > 200) %>%
  mutate(word = reorder(word,n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip()

#Another visualization
library(wordcloud)
tidydata %>% with(wordcloud(word, n, max.words = 500))

#Sentiment Analysis
library(tidyr)

#sentiments include three lexicons: AFINN, bing, nrc
nrcjoy <- get_sentiments( "afinn") #%>%
#filter(sentiments == "joy")

tddata <- data %>% unnest_tokens(word, text) %>% 
  anti_join(stop_words)     %>%
  inner_join(get_sentiments("bing")) %>% 
  group_by(line) %>%
  count(sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)

#Create a data : line word n(word) without removing stop words
line_word <- data %>%
  unnest_tokens(word, text) %>%
  count(line, word, sort = TRUE) %>%
  ungroup()

#View(line_word)
total_word <- line_word %>%
  group_by(word) %>%
  summarize(total = sum(n))

line_word <- left_join(line_word, total_word)

#Mutate tf-idf variables
total = total_word$total
freq_by_rank <- line_word %>%
  group_by(line)# %>%
#  mutate(rank = row_number(), 
#        'term frequency' = n/total_word)
View(freq_by_rank)

line_words <- line_word %>% bind_tf_idf(word, line, n)

line_words %>% select(-total) %>%
  arrange(desc(tf_idf))
View(line_words)

## N-gram
ngramdata  <- data %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2 )
View(ngramdata)

## Counting and filtering n-grams
ngramdata %>% 
  count(bigram, sort = TRUE)

bigrams_separated <- ngramdata %>%
  separate(bigram, c("word1", "word2"), sep = " ")

bigrams_filtered <- bigrams_separated %>%
  filter(!word1 %in% stop_words$word) %>%
  filter(!word2 %in% stop_words$word)

# new bigram counts:
bigram_counts <- bigrams_filtered %>%
  count(word1, word2, sort = TRUE)

#bigram_counts
bigrams_united <- bigrams_filtered %>%
  unite(bigram, word1, word2, sep = " ")

#View(bigrams_united)
#trigram
bigram_total <- data %>%
  unnest_tokens(trigram, text, token = "ngrams", n = 3)      %>%
  separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word3 %in% stop_words$word)                        %>%
  count(word1, word2, word3, sort = TRUE)


##If you want to filter
bigrams_filtered %>%
  filter(word2 == "street") %>%
  count(line, word1, sort = TRUE)