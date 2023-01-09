import boto3
import os
from nltk.corpus import stopwords
import nltk
import spacy
import logging
from gensim.models import CoherenceModel
import sys
import re
import pandas as pd
from textblob import TextBlob
import matplotlib as mpl

import numpy as np
import matplotlib.pyplot as plt
import gensim
from wordcloud import WordCloud,ImageColorGenerator
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import pyLDAvis
import pyLDAvis.gensim


boto3.setup_default_session(profile_name='', region_name='')
S3_BUCKET = ''
S3_PREFIX = ''
mallet_path = ""
base_folder = ""

class TweetsBatchScanner:
    def __init__(self, s3_bucket, bucket_path, on_the_fly_input=False):
        self.s3_bucket = s3_bucket
        self.bucket_path = bucket_path
        self.logger = self.make_logger()
        self.best_number_of_topics = 6
        self.nlp = self.setup_nlp()
        try:
            self.stop_words = stopwords.words('english')
        except:
            try:
                nltk.download('stopwords')
                self.stop_words = stopwords.words('english')
            except Exception as ex:
                self.logger.error(f"Unable to download stopwords due to {ex}")

        self.stop_words.extend(["get", "say", "go", "come", "see", "today", "make", "going", "way", "rt", "man", "good",
                                "take", "well", "girl", "woman", "use", "look", "need", "feel", "day", "time", "put", "tell"])
        self.tweets, self.corpus, self.id2word, self.data_words_nostops, self.text, self.tweets_cleaned = self.preprocess_data()
        self.logger.info(f"tweets cleaned {self.tweets_cleaned[:2]}")
        self.logger.info(f"text {self.text[:2]}")
        self.logger.info(f"data words nostops {self.data_words_nostops[:2]}")
        self.logger.info(f"tweets {self.tweets[:2]}")
        self.words_freq_dict = self.get_frequent_words(self.data_words_nostops)
        self.words_pol = self.avg_pol_tweets_containing()

    def make_logger(self):
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        logger = logging.getLogger('lgr')
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            logger.removeHandler(handler)
        consoleHandler = logging.StreamHandler(stream=sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
        return logger

    def setup_nlp(self):
        nlp = spacy.load('en_core_web_sm')
        return nlp

    def s3_client(self):
        try:
            boto3.setup_default_session(profile_name='trendspotting', region_name="us-east-1")
            return boto3.client('s3')
        except Exception as ex:
            self.logger.error("S3 client creation failed due to error : %s ", str(ex))
            sys.exit(1)

    def est_polarity(self, text):
        try:
            return TextBlob(text).sentiment_assessments.polarity

        except:
            return 0

    def random_color_func(self, word=None, font_size=None, position=None,  orientation=None, font_path=None, e=None):
        h = int(360.0 * 21.0 / 255.0)
        s = int(100.0 * 255.0 / 255.0)
        return "hsl({}, {}%, {}%)".format(h, s, e)

    def read_tweets_json(self):
        list_keys = self.get_list_keys_for_prefix()
        tweets = []
        for key in list_keys:
            try:
                rs = boto3.resource('s3')
                response = rs.Object(self.s3_bucket, key).get()
                df = response['Body'].read().decode("utf-8")
                regr = re.compile("text\"[\s]*:[\s]*\"([\d\w\s,.?&@!:;%$\*><\\\/£|~+=\-_]+)\"")
                df = regr.findall(df)
                tweets = tweets+df
            except Exception as e:
                self.logger.info(f"Failed to read tweets due to {e}")
        return tweets

    def make_bigrams(self, sentences):
        try:
            gs = gensim.models.Phrases(sentences, min_count=15)
            return gs[sentences]
        except Exception as e:
            self.logger.error(e)

    def remove_stopwords(self, texts):
        without_stop_words = [[word for word in simple_preprocess(str(doc)) if word not in self.stop_words] for doc in texts]
        return without_stop_words

    def base_word_cloud(self, text, stopwords, max_words):
        return WordCloud(max_words=max_words, stopwords=stopwords).generate(text)

    def wordcloud_polarity(self):
        self.logger.info(f"words freq dicr {self.words_freq_dict}")
        self.words_pol = self.avg_pol_tweets_containing()
        exte = [k for k, v in self.words_pol.items() if v == 'na']
        self.words_pol = {k: v for k, v in self.words_pol.items() if v != 'na'}
        self.logger.info(f"exte {exte}")
        self.words_freq_dict = {k: v for k, v in self.words_freq_dict.items() if k not in exte}
        wc = WordCloud(max_words=200, collocations=False, width=2000, height=1000).generate_from_frequencies(self.words_freq_dict)
        fig, ax = plt.subplots(1,figsize=(40, 20))
        fig.subplots_adjust(bottom=0.5)
        plt.axis('off')
        cmap = mpl.cm.autumn
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        #
        # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        #              cax=ax, orientation='horizontal', label='Polarity')
        plt.imshow(wc.recolor(color_func=self.grey_color_func, random_state=3), interpolation="bilinear")
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
        plt.savefig(os.path.join(base_folder, 'wordcloud_polarity.png'))

    def preprocess_data(self):
        tweets = self.read_tweets_json()
        self.logger.info("read tweets from s3")
        data_words = [self.clean_text(a) for a in tweets]
        data_words_nostops = self.remove_stopwords(data_words)
        self.logger.info("removed stopwords and cleaned text")
        id2word = corpora.dictionary.Dictionary(data_words_nostops)
        corpus = [id2word.doc2bow(text) for text in data_words_nostops]
        fulltext = ""
        for tweet in data_words_nostops:
            fulltext = fulltext + " ".join([word for word in tweet])
        tweets_cleaned = [" ".join([word for word in tweet]) for tweet in data_words_nostops]
        return tweets, corpus, id2word, data_words_nostops, fulltext, tweets_cleaned

    def fit_lda_model(self, corpus, id2word, random_state=7):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=self.best_number_of_topics,
                                                    random_state=random_state,
                                                    passes=4)
        return lda_model

    def create_topics_visualisation(self, lda_model):
        vis = pyLDAvis.gensim.prepare(lda_model, self.corpus, self.id2word, R=10)
        pyLDAvis.save_html(vis, os.path.join(base_folder, 'LDA_Visualization.html'))
        self.logger.info("created topics visualisation")


    def df_polarity(self):
        return pd.DataFrame.from_dict({'tweet': self.tweets_cleaned, 'polarity': [self.est_polarity(i) for i in
                                                                                  self.tweets_cleaned]})


    def coherence_plot(self, start, limit, step):
        model_list, coherence_values = self.get_coherence_scores(dictionary=self.id2word, corpus=self.corpus,
                                                                 texts=self.data_words_nostops, start=start,
                                                                 limit=limit, step=step)

        x = range(start, limit+1, step)
        self.best_number_of_topics = x[np.argmax(coherence_values)]
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence by num of topics "), loc='best')
        plt.savefig(os.path.join(base_folder, 'coherence.png'))

    def get_coherence_scores(self, dictionary, corpus, texts, limit, start, step):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit+1, step):
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values

    def clean_text(self, t):
        t = re.sub(
            r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
            "", t)
        t = re.sub(r"(@[^ ]*)", "", t)

        t = " ".join([x.lemma_ for x in self.nlp(t) if
                      not x.is_stop and x.pos_ in ["NOUN", "AUX", "VERB", "PROPN", "ADJ", "ADV"] and x.is_alpha])
        return t


    def get_list_keys_for_prefix(self):
        try:
            keys = []
            paginator = boto3.client('s3').get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=self.bucket_path)
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if re.match(".*.jsonl", obj["Key"]):
                            keys.append(obj["Key"])
        except Exception as ex:
            self.logger.error(f"Failed to extract the list of keys because of error: %s", str(ex))
            sys.exit(-1)
        return keys


    def groupby_polarity(self, df):
        e = df.groupby('topic_num').mean().reset_index().rename({'polarity': 'polarity_mean'}, axis=1)
        f = df.groupby('topic_num').var().reset_index().rename({'polarity': 'polarity_var'}, axis=1)
        g = e.merge(f, on='topic_num')
        return g[["topic_num", "polarity_mean", "polarity_var"]]


    def get_frequent_words(self, data_words_nostops):
        words = {}
        for tokenised_tweet in data_words_nostops:
            for word in tokenised_tweet:
                if word in words.keys():
                    words[word] = words[word]+1
                else:
                    words[word] = 1
        s = sorted(words.items(), key=lambda item: item[1], reverse=False)
        return (dict(s))


    def avg_pol_tweets_containing(self):

        df = self.df_polarity()
        words_pol = {}
        for word, freq in self.words_freq_dict.items():
            avg_pol = np.mean(df[df["tweet"].apply(lambda a: word in a)]['polarity']) or 'na'
            words_pol[word] = avg_pol
        return words_pol


    def grey_color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        x = self.change_scale(self.words_pol[word])
        return f"hsl({x}, 80%, 60%)"


    def tweets_analysis(self):

        # self.coherence_plot(4, 10, 2)
        lda_model = self.fit_lda_model(self.corpus, self.id2word)
        self.create_topics_visualisation(lda_model)
        c.wordcloud_polarity()


    def change_scale(self, i):
        return (i+1)/2*60


if __name__ == "__main__":
    c = TweetsBatchScanner(s3_bucket=S3_BUCKET,
                           bucket_path=S3_PREFIX,
                           on_the_fly_input=False)
    c.tweets_analysis()
