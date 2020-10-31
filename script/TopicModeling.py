import pandas as pd 
import pickle
from contextualized_topic_models.models.ctm import CTM
from contextualized_topic_models.utils.data_preparation import TextHandler
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.evaluation.measures import TopicDiversity, CoherenceNPMI,\
    CoherenceWordEmbeddings, InvertedRBO
import numpy as np

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# code for training

def model_topics(file, n_components=20):
    handler = TextHandler(file)
    handler.prepare() # create vocabulary and training data

    # generate BERT data
    training_bert = bert_embeddings_from_file(file, "distiluse-base-multilingual-cased")

    training_dataset = CTMDataset(handler.bow, training_bert, handler.idx2token)

    ctm = CTM(input_size=len(handler.vocab), bert_input_size=512, inference_type="combined", n_components=n_components, num_epochs=10)

    ctm.fit(training_dataset) # run the model

    return ctm

def save_model(model_filename, model):
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

# code for evaluation

def load_predefined_words(file='../data/skills.csv'):
    keyword_dict = pd.read_csv(file)
    predefined = list(keyword_dict.values.flatten())
    predefined = filter(lambda x: x==x, predefined) # filter out the nan values
    predefined = ' '.join(predefined).split() # split the dictionary to single words
    predefined = list(set(predefined))
    return predefined

def calc_topic_diversity(model, topk = 10):
    td = TopicDiversity(model.get_topic_lists(topk))
    print("topic diversity", td.score(topk=topk))
    rbo = InvertedRBO(model.get_topic_lists(topk))
    print("inverted RBO", rbo.score(topk=topk))

def get_topic_score(model, predefined, n_components=20, topk=400):
    '''find the topic with the highest overlap with predefined dictionary'''
    # select the document type
    topic_score_array = np.zeros(n_components)

    for i in range(n_components):
        counter = 0
        topics = model.get_topic_lists(topk)[i]
        for x in topics:
            if x in predefined:
                counter += 1
        topic_score_array[i] = counter / topk
    return topic_score_array

def plot_overlap_bar(topic_score_array):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')
    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    plt.bar(range(1,1+topic_score_array.shape[0]), topic_score_array * 100)
    plt.xticks(range(1,1+topic_score_array.shape[0]))
    plt.xlabel("Topic Index")
    plt.ylabel("Percentage (%)")
    plt.show()

def plot_prcurve(model, predefined, top_topic):
    defined_ls = []
    catched_ls = []
    topk_ls = [50, 100, 150, 200, 250, 300, 350, 400]
    for i in topk_ls:
        topics = model.get_topic_lists(i)[top_topic]
        n_overlap = len(set(topics).intersection(set(predefined)))
        
        defined_perc = n_overlap/len(topics)
        catched_perc = n_overlap/len(predefined)
        
        defined_ls.append(defined_perc)
        catched_ls.append(catched_perc)
    
    plt.plot(catched_ls, defined_ls)
    plt.xlabel("Recall") # % of topic words existed in predefined dictionary
    plt.ylabel("Precision") # % of predefined words captured by topic modelling
    for i, txt in enumerate(topk_ls):
        plt.annotate(txt, (catched_ls[i], defined_ls[i]))
    plt.show()

def plot_word_cloud_topics(topics, predefined):

    # capture the words to plot
    undefined_topics = []
    uncatched_topics = []
    overlap = []

    for x in topics:
        if x not in predefined:
            undefined_topics.append(x)
    for x in predefined:
        if x not in topics:
            uncatched_topics.append(x)
            
    for x in predefined:
        if x in topics:
            overlap.append(x)

    # define color scheme
    def orange_color_func(word, font_size, position,orientation,random_state=0, **kwargs):
        return("hsl(12,100%%, %d%%)" % np.random.randint(30, 70))
    
    def purple_color_func(word, font_size, position,orientation,random_state=0, **kwargs):
        return("hsl(260,100%%, %d%%)" % np.random.randint(30, 70))

    def cyan_color_func(word, font_size, position,orientation,random_state=0, **kwargs):
        return("hsl(210,100%%, %d%%)" % np.random.randint(30, 70))

    uncatched_wordcloud = WordCloud(width=250, height=400, max_words=50, prefer_horizontal=1,
                                    background_color="white")\
        .generate(' '.join(uncatched_topics))
    uncatched_wordcloud.recolor(color_func = orange_color_func)

    undefined_wordcloud = WordCloud(width=250, height=400, max_words=50, prefer_horizontal=1,
                                    background_color="white")\
        .generate(' '.join(undefined_topics))
    undefined_wordcloud.recolor(color_func = cyan_color_func)

    overlap_wordcloud = WordCloud(width=250, height=400, max_words=50, prefer_horizontal=1,
                                background_color="white")\
        .generate(' '.join(overlap))
    overlap_wordcloud.recolor(color_func = purple_color_func)

    # combine the plots
    f, axes = plt.subplots(1,3,figsize=(100,50))

    axes[0].imshow(uncatched_wordcloud)
    axes[0].set_title("Uncaptured words", fontsize=100)
    axes[0].axis("off")

    axes[1].imshow(overlap_wordcloud)
    axes[1].set_title("Overlapped words", fontsize=100)
    axes[1].axis("off")

    axes[2].imshow(undefined_wordcloud)
    axes[2].set_title("Undefined words", fontsize=100)
    axes[2].axis("off")



if __name__ == "__main__":

    # train the model
    file = "../data/clean_text.txt"
    n_components = 20
    ctm = model_topics(file, n_components = n_components)

    # save the model
    model_filename = "../model/bert_model2.pkl"
    save_model(model_filename, ctm)

    # get top k words for each topic
    k=50
    topics = model.get_topic_lists(k)

    # load predefined words
    predefined = load_predefined_words()

    # evaluate model
    calc_topic_diversity(model)

    # identify the topic with top score
    topk=400
    scores = get_topic_score(model, predefined, n_components=n_components, topk=topk)
    top_topic = np.where(scores, np.max(scores))

    # plot precision recall curve
    plot_prcurve(model, predefined, top_topic)

    # plot word cloud
    top_topic_words = model.get_topic_lists(topk)[top_topic]
    plot_word_cloud_topics(top_topic_words, predefined)




