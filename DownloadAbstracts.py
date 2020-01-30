import pandas as pd
import re
import os
import spacy
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk.data
nltk.download('stopwords')
from nltk.corpus import stopwords
from pprint import pprint
from gensim.models import CoherenceModel


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 'background', 'method', 'result','conclusion','aim','objective','study','outcome','use','may','treatment','anxiety','disorder'])

def load_data(text_path):
  abstract = list()
  start_article=0
  with open (text_path,"r" ,encoding="utf-8") as f:
      data=f.read()
  end_article_list=[m.start() for m in re.finditer('LINE]', data)]
  for  index in  end_article_list:
    article=data[start_article:index+5]
    start_article=index+6
    abstract.append(article.strip("\n").replace("\n\n\n\n","\n\n").split("\n\n"))
  names = ['title', 'abstract', 'id' ]
  return pd.DataFrame(abstract,columns=names)
def proccess(data):
    # Remove punctuation
    data['abstract_processed'] = data['abstract'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert to lowercase
    data['abstract_processed'] = data['abstract_processed'].map(lambda x: x.lower())
    abstract = data.abstract_processed.values.tolist()
    #word_cloud(abstract)
    return list(tokenize(abstract))

def word_cloud(list):
    from wordcloud import WordCloud
    long_string = ','.join(list)
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

    img= wordcloud.generate(long_string)
    img.to_file('worcloud.jpeg')



def tokenize(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# supporting function
def compute_coherence_values(corpus, dictionary,texts, k, a, b):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)


    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')

    return coherence_model_lda.get_coherence()

def parameter_tunning(corpus,id2word,data_lemmatized):
    import numpy as np
    import tqdm

    grid = {}
    grid['Validation_Set'] = {}

    # Topics range
    min_topics = 30
    max_topics = 40
    step_size = 10
    topics_range = range(min_topics, max_topics, step_size)

    # Alpha parameter
    alpha = list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')

    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')

    # Validation sets
    num_of_docs = len(corpus)
    corpus_sets = [  # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25),
                     # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5),
                     # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.75),
                     corpus]

    corpus_title = ['100% Corpus']

    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     }

    # Can take a long time to run
    if 1 == 1:
        pbar = tqdm.tqdm(total=(len(beta) * len(alpha) * len(topics_range) * len(corpus_title)))

        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for b in beta:
                        # get the coherence score for the given parameters
                        cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word,texts=data_lemmatized,k=k, a=a, b=b)
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)

                        pbar.update(1)
        pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
        pbar.close()
def build_lda_model(corpus,id2word):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=30,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha='asymmetric'
                                           , eta=0.9)


    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    return lda_model

def compute_coherence_score(lda_model,data_lemmatized,id2word):
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    return coherence_lda

def visualize(lda_model,corpus,id2word):
    import pyLDAvis.gensim
    import pyLDAvis
    # Visualize the topics
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis_prepared, 'index30a_finallly_lda.html')

def main():
    data=load_data("data/input.txt")
    data_words =proccess(data)
    #print(data_words[:1][0][:30])

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

   # print(data_lemmatized[:1][0][:30])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
    # View
    #print(corpus[:1])

    #parameter_tunning(corpus,id2word,data_lemmatized)

    # Build LDA model
    lda_model=build_lda_model(corpus, id2word)

    compute_coherence_score(lda_model, data_lemmatized, id2word)

    visualize(lda_model, corpus, id2word)

if __name__ == '__main__':
    main()
