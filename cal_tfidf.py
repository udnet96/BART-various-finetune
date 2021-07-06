from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
from scipy.sparse.csr import csr_matrix  # need this if you want to save tfidf_matrix
import random
random.seed(42)

with open('xsum/train_small.source', 'r', encoding='utf8') as f:
    train_src = f.readlines()
with open('xsum/train_tfidf_small.target', 'r', encoding='utf8') as f:
    train_tgt = f.readlines()
tf = TfidfVectorizer(input='content', analyzer='word', ngram_range=(1, 1),
                     min_df=0, stop_words='english', sublinear_tf=True, )
start = 0
end = 5000

c1, c2, c3 = 0, 0, 0
stop_words = nltk.corpus.stopwords.words('english')
ad1 = '~ ! @ # $ % ^ & * ( ) - _ + = / ? > <'.split()
ad = ['.', ',', '``', "'s", '-lrb-', '-rrb-']
stop_words.extend(ad)
stop_words.extend(ad1)
with open('xsum/train_tfidf_small_<s>.source', 'w', encoding='utf8') as f:
    tfidf_matrix = tf.fit_transform(train_src[start:end])
    feature_names = tf.get_feature_names()

    tfidf_matrix_tgt = tf.fit_transform(train_tgt[start:end])
    feature_names_tgt = tf.get_feature_names()
    for idx, tgt in enumerate(train_src):
        if idx % 5000 == 0 and idx != 0:
            start += 5000
            if idx > len(train_src) - 5000:
                end = len(train_src)
            else:
                end += 5000
            tfidf_matrix = tf.fit_transform(train_src[start:end])
            feature_names = tf.get_feature_names()

            tfidf_matrix_tgt = tf.fit_transform(train_tgt[start:end])
            feature_names_tgt = tf.get_feature_names()

        ref_words = nltk.word_tokenize(train_tgt[idx])
        src_words = nltk.word_tokenize(train_src[idx])

        idf_idx = idx % 5000
        feature_index = tfidf_matrix[idf_idx, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[idf_idx, x] for x in feature_index])

        tfidf_rank = []
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            tfidf_rank.append(w)

        cnt = 0
        key_words = []
        for word in tfidf_rank:
            if word in ref_words:
                key_words.append(word)
                cnt += 1
            if cnt > 3:
                break

        if len(key_words) <= 3:
            feature_index_tgt = tfidf_matrix_tgt[idf_idx, :].nonzero()[1]
            tfidf_scores_tgt = zip(feature_index_tgt, [tfidf_matrix_tgt[idf_idx, x]
                                                       for x in feature_index_tgt])
            tfidf_rank_tgt = []
            for w, s in [(feature_names_tgt[i], s) for (i, s) in tfidf_scores_tgt]:
                tfidf_rank_tgt.append(w)

            for word in tfidf_rank_tgt:
                if (word in src_words) and (word not in stop_words) and (word not in key_words):
                    key_words.append(word)
                    cnt += 1
                if cnt > 3:
                    break
            c2 += 1

        random.shuffle(src_words)
        if len(key_words) <= 3:
            for w in src_words:
                if w in ref_words and (w not in stop_words) and (w not in key_words):
                    key_words.append(w)
                    cnt += 1
                if cnt > 3:
                    break
            c3 += 1
        print(key_words, len(key_words), c2, c3)
        key_tgt = tgt
        for key in key_words:
            key_tgt = key_tgt.replace(key, '<s> ' + key + ' </s>')
        f.write(key_tgt)
print('c2, c3 : ', c2, c3)

