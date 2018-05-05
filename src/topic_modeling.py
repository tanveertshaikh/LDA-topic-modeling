import csv
import json
import logging
import os
import pprint
import re
import sys
import time

import nltk
import numpy as np
import pandas as pd
from nltk import Tree, pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn import linear_model, metrics
from tqdm import tqdm

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('treebank')


dictionary = dict.fromkeys(words.words(), None)
lemmatizer = WordNetLemmatizer()



#########################################################################################################################################################################################*****HELPER_FUNCTIONS*****##########################################################################################################################################################################################################################################



def is_english_word(word):
    # Parameter: word
    # Return: True if the word is a valid English word, otherwise False
    # nltk corpus word validator
    try:
        dictionary[word.lower()]
        return True
    except KeyError:
        return False


def manual_check(word):
    # Parameter: word
    # Return: False if the word is illegal, True otherwise
    # Manual spelling mistakes checker
    manual_set = set(['to', 'them', 'his', 'cannot', 'they', 'during', 'him', 'should', 'this', "'ve", 'where',
                      'because', 'their', 'what', "'", 'since', 'your', 'everything', 'we', 'how', 'although',
                      'others', 'would', 'anything', 'could', 'against', 'you', 'among', 'into', 'everyone', 'with',
                      'everybody', 'from', '.', ",", 'anyone', 'until', ':', "'s", 'than', 'those', 'these', "n't", 'of', 'my',
                      'and', 'itself', 'something', 'our', 'themselves', 'if', '!', 'that', '-', 'ourselves',
                      'when', 'without', 'which', 'towards', 'shall', 'whether', 'unless', 'the', 'for',
                      'whenever', 'anytime', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                      "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                      'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they',
                      'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                      'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
                      'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                      'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                      'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                      'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
                      't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                      'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                      "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                      'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
                      "wouldn't", ",", ".", "?", "@", "$", "&", ":", ";", "!", "'", '"', "<", ">", "(", ")", "{", "}", "[", "]", "|", "~", "`", 'ago', "#", "%", "*"])
    return word in manual_set


def get_score_a(s, partial_scores):
    mean = np.mean(partial_scores)
    sd = np.std(partial_scores)
    if s > mean + (2 * sd):
        return 1
    elif s > mean + sd:
        return 2
    elif s > mean:
        return 3
    elif s > mean - sd:
        return 4
    else:
        return 5


def get_score_b(s, partial_scores):
    mean = np.mean(partial_scores)
    sd = np.std(partial_scores)
    if s > mean + (2 * sd):
        return 4
    elif s > mean + sd:
        return 3
    elif s > mean:
        return 2
    elif s > mean - sd:
        return 1
    else:
        return 0


def get_score_ci(s, partial_scores):
    mean = np.mean(partial_scores)
    sd = np.std(partial_scores)
    if s > mean + (2 * sd):
        return 1
    elif s > mean + sd:
        return 2
    elif s > mean:
        return 3
    elif s > mean - sd:
        return 4
    else:
        return 5


def get_score_cii(s, partial_scores):
    mean = np.mean(partial_scores)
    sd = np.std(partial_scores)
    if s > mean + (2 * sd):
        return 5
    elif s > mean + sd:
        return 4
    elif s > mean:
        return 3
    elif s > mean - sd:
        return 2
    else:
        return 1


def get_score_di(s, partial_scores):
    mean = np.mean(partial_scores)
    sd = np.std(partial_scores)
    if s > mean + (2 * sd):
        return 5
    elif s > mean + sd:
        return 4
    elif s > mean:
        return 3
    elif s > mean - sd:
        return 2
    else:
        return 1


def get_score_dii(s, partial_scores):
    mean = np.mean(partial_scores)
    sd = np.std(partial_scores)
    if s > mean + (2 * sd):
        return 1
    elif s > mean + sd:
        return 2
    elif s > mean:
        return 3
    elif s > mean - sd:
        return 4
    else:
        return 5


def zero_one(label):
    if label == 'high':
        return 1
    else:
        return 0


def high_low(label):
    if label == 1:
        return 'high'
    else:
        return 'low'


def grade(score):
    """Final score scale for the class label

    Arguments:
        score {float} -- Final score

    Returns:
        final label {str} -- The final class label of the essay 'high' or 'low'
    """
    if score >= 23:
        # return 'high'
        return 'unknown'
    else:
        # return 'low'
        return 'unknown'


def get_lemmas(prompt_nouns):
    s1 = []
    for noun in prompt_nouns:
        for syn in wn.synsets(noun, pos='n'):
            for lemma in syn.lemmas():
                s1.append(lemma.name())

    for noun in prompt_nouns:
        s1.append(noun)

    s2 = []
    for s in s1:
        s = s.replace('_', ' ').lower()
        s2.append(s)

    s2 = list(set(s2))
    return s2


def get_entailments(prompt_verbs):
    s1 = []
    for verb in prompt_verbs:
        for syn in wn.synsets(verb, pos='v'):
            for entailment in syn.entailments():
                s1.append(entailment.name())

    for verb in prompt_verbs:
        s1.append(verb)

    sa = []
    for s in s1:
        s = s.split('.')
        sa.append(s)

    s3 = []
    s2 = np.array(sa)
    for s in s2:
        s3.append(s[0])

    s0 = []
    for s in s3:
        s = s.replace('_', ' ').lower()
        s0.append(s)

    s0 = list(set(s0))
    return s0


def process_prompts_tr(prompts_list):
    p1 = []
    # Processing prompts to extract the questions
    for prompt in prompts_list:
        p0 = prompt.split('\t')
        p1.append(p0)

    p2 = np.array(p1)
    p3 = []
    for p in p2:
        p3.append(p[2])

    prompt_dict = pd.factorize(p3)

    p4 = list(zip(range(8), prompt_dict[1]))
    for n, p in p4:
        prompts_map_tr[p] = n

    #prompt_no = prompt_dict[0][99]
    #print(prompt_no)
    return prompt_dict


def process_prompts_te(prompts_list):
    p1 = []
    # Processing prompts to extract the questions
    for prompt in prompts_list:
        p0 = prompt.split('\t')
        p1.append(p0)

    p2 = np.array(p1)
    p3 = []
    for p in p2:
        p3.append(p[2])

    prompt_dict = pd.factorize(p3)
    #prompt_no = prompt_dict[0][99]
    #print(prompt_no)
    return prompt_dict, p3

def get_search_words(prompts_list):
    # Initialize prompt_nouns, prompt_verbs and search_words
    prompt_nouns = []
    search_words = []
    prompt_verbs = []

    # Processing prompts to extract the questions
    de = ["agree", "reasons"]
    prompts = ' '.join(prompts_list)
    prompts = prompts.split('\t')
    prompts = list(filter(None, prompts))
    prompts = list(set(prompts))
    prompts = [el for el in prompts if not any(ignore in el for ignore in de)]
    prompts = ' '.join(prompts)
    prompt_tag_list = pos_tag(word_tokenize(prompts))

    # Extract nouns in question
    for token in prompt_tag_list:
        if token[1] in singular_noun or token[1] in plural_noun:
            prompt_nouns.append(token[0])
    prompt_nouns = list(set(prompt_nouns))

    # Extract verbs in question
    for token in prompt_tag_list:
        if token[1] in singular_verb or token[1] in plural_verb:
            prompt_verbs.append(token[0])
    prompt_verbs = list(set(prompt_verbs))

    synonyms = get_lemmas(prompt_nouns)
    entails = get_entailments(prompt_verbs)

    print("Synonyms of all nouns in the topic")
    print(synonyms)
    print("Verb entailments of all the verb in the topic")
    print(entails)

    return search_words


def tokenize_sentence(text):
    # sentences = []
    # Todo Check if theres an SBAR in the syntactic tree and if it does not preceed a verb (or has IN)
    # nltk_sentences = sent_tokenize(text)
    # for s in nltk_sentences:
    #     pos = nltk.pos_tag(word_tokenize(s))
    #     ws = s.split(" ")
    #     cws = []

    #     for i in range(0,len(ws)):
    #         if ws[i].strip()[0].isupper() and i != 0:
    #             if "NPP" not in pos[i]:
    #                 cws.append("#")

    #         cws.append(ws[i])

    #     sn = ' '.join(cws)
    #     new_sentences = sn.split(" # ")
    #     for ns in new_sentences:
    #         sentences.append(ns)

    return sent_tokenize(text)


def get_clauses(tree, clauses):
    if tree.label() == "S" or tree.label() == "ROOT":
        clauses.append(tree)
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            get_clauses(subtree, clauses)


def get_subject(tree):
    for s in tree:
        if s.label() in ["S", "NP", "FRAG"]:
            return get_subject(s)

    return {"pos": tree[len(tree)-1].label(), "word": tree[len(tree)-1].leaves()[0]}


def get_np_vp(tree, np, vp):
    t_np = None
    t_vp = None
    for s in tree:
        if type(s) == nltk.tree.Tree:
            if s.label() == "VP":
                t_vp = s
            if s.label() == "NP":
                t_np = s

    if t_np != None and t_vp != None:
        np.append(t_np)
        vp.append(t_vp)
    for s in tree:
        if type(s) == nltk.tree.Tree:
            get_np_vp(s, np, vp)


def get_predicate_verbs(tree, verbs, verbs_pos):
    if len(tree) == 1:
        if tree.label() in verb_pos:
            verbs_pos.append(tree.label())
            verbs.append(tree.leaves()[0])
    else:
        for s in tree:
            if type(s) == nltk.tree.Tree:
                get_predicate_verbs(s, verbs, verbs_pos)


def subject_verb_agreement(s, s_pos, v_pos, v):
    n_group = "c"
    if s.lower() == "i":
        n_group = "a"
    elif s_pos == "NNS" or (s_pos == "PRP" and s.lower() in ["we", "you", "they"]):
        n_group = "b"

    verb_agreement_errors = {
        # 1st person singular (I)
        "a": ["are", "is", "were", "VBZ", "VBG", "VBN"],
        # 2nd person (you), 3rd person plural (They, we, girls)
        "b": ["am", "is", "was", "VBZ", "VBG", "VBN"],
        # 3rd person singular (he, she, it, girl), ProperNoun(Chicago)
        "c": ["am", "are", "were", "VBP", "VBG", "VBN"]
    }

    v_group = v_pos
    if v.lower() in ["am", "are", "is", "was", "were"]:
        v_group = v.lower()

    # Note: For a sentence like "The girls running play soccer.", This would return an error but we shouldn't as if the NP has another NP that has a VP, we should accept this.
    return v_group in verb_agreement_errors[n_group]


def tense_agreement(v_words, v_pos):
    print("tense: ", ' '.join(v_words))
    cv_words = []
    cv_pos = []

    perfect_prog = False
    if "has been" in " ".join(v_words).lower() or "have been" in " ".join(v_words).lower() or "had been" in " ".join(v_words).lower():
        perfect_prog = True

    for i in range(0, len(v_words)):
        if perfect_prog:
            if v_words[i].lower() == "been":
                cv_words.append("has been")
                cv_pos.append("VERB.PER.PROG")
            elif v_words[i].lower() not in ["have", "had", "has"]:
                cv_words.append(v_words[i])
                cv_pos.append(v_pos[i])
        else:
            if v_words[i].lower() in ["was", "were", "am", "are", "is", "be"]:
                cv_words.append(v_words[i])
                cv_pos.append("VERB.PROG")
            elif v_words[i].lower() in ["have", "has", "had"]:
                cv_words.append(v_words[i])
                cv_pos.append("VERB.PER")
            else:
                cv_words.append(v_words[i])
                cv_pos.append(v_pos[i])

    for i in range(0, len(cv_words)):
        if cv_pos[i] == "VERB.PER.PROG":
            if i + 1 < len(cv_words) and (cv_pos[i+1] not in ["VBG", "VBN"]):
                print("!!error " + cv_pos[i] + "-tense")
                return True
        elif cv_pos[i] == "VERB.PER" or cv_pos[i] == "VERB.PROG":
            if i + 1 < len(cv_words) and (cv_pos[i+1] not in ["VBN"]):
                print("!!error " + cv_pos[i] + "-tense")
                return True
        elif cv_pos[i] == "VBN":
            if i - 1 <= 0 and (cv_pos[i-1] not in ["VERB.PER.PROG", "VERB.PROG",  "VERB.PER"]):
                print("!!error " + cv_pos[i] + "-tense")
                return True
        elif cv_pos[i] == "VBG":
            if i - 1 >= 0 and (cv_pos[i-1] not in ["VERB.PER.PROG", "VERB.PROG"]):
                print("!!error " + cv_pos[i] + "-tense")
                return True
        # elif cv_pos[i] == "VB":
            # Todo Check if theres a modal before this in the syntactic tree

    return False



#########################################################################################################################################################################################*****MAIN_FUNCTIONS*****########################################################################################################################################################################################################################################



def evaluate_spelling(text):
    """Spelling Mistakes - b

    Arguments:
        tokens {str} -- Word tokens

    Returns:
        number {float} -- Incorrect spelling value
    """
    total_words = 0
    mistakes = 0
    tokens = word_tokenize(text)
    for token in tokens:
        token = token.lower()
        # Verify unknown wordnet tokens against a manual list of legal words
        # and also doubled checking it's validity with nltk words corpus
        if wn.synsets(token) == [] and manual_check(token) == False and is_english_word(token) == True:
            # Increment mistakes count if the token is invalid
            mistakes += 1
        total_words += 1
    if total_words == 0:
        return 0
    mistake_val = mistakes/float(total_words)
    return mistake_val


def evaluate_length(text):
    """Number of sentences and length - a

    Arguments:
        text {str} -- Test data from the text file (raw text data) which is not tokenized

    Returns:
        Lexical richness of the document (lengths of sentences)
        {float} -- Ratio of number of words to number of sentences
    """
    # The words must contain letters or digits (removing punctuations)
    nonPunct = re.compile('.*[A-Za-z0-9].*')
    # Tokenize text for sentences
    sents = tokenize_sentence(text)
    tokens_raw = word_tokenize(text)
    # Count the number of words
    filtered_words = [w for w in tokens_raw if nonPunct.match(w)]
    num_sents = len(sents)
    num_words = len(filtered_words)
    if num_sents == 0:
        return 0
    s = num_words/float(num_sents)
    return s


def evaluate_syntax(sentences):
    agreement_mistakes = 0
    tense_mistakes = 0
    sub_verb_pairs = 0
    for s in sentences:
        syntactic_tree = Tree.fromstring(nlp.parse(s))
        #syntactic_tree = Tree.fromstring(parsed_str["parse"])
        vp = []
        np = []

        get_np_vp(syntactic_tree, np, vp)

        print("\n\n\nEvaluating sentence: ", s)
        for i in range(0, len(np)):
            subject = get_subject(np[i])
            if subject["pos"] in noun_pos:
                verbs = []
                verbs_pos = []
                get_predicate_verbs(vp[i], verbs, verbs_pos)
                if len(verbs) > 0:
                    sub_verb_pairs = sub_verb_pairs + 1
                    print("verb agreement between:", subject["word"], verbs[0])
                    mistake = subject_verb_agreement(
                        subject["word"], subject["pos"], verbs_pos[0], verbs[0])
                    if mistake:
                        print("!!ERROR in subject verb agreement")
                        agreement_mistakes = agreement_mistakes + 1

                    mistake = tense_agreement(verbs, verbs_pos)
                    if mistake:
                        tense_mistakes = tense_mistakes + 1
    p1 = 0
    p2 = 0
    if sub_verb_pairs > 0:
        p1 = agreement_mistakes / sub_verb_pairs
        p2 = tense_mistakes / sub_verb_pairs
    return {"agreement": p1, "tense": p2}


def evaluate_text_coherence(text):
    return 0


def evaluate_topic_coherence(text, search_words):
    lemma_tokens = []
    total_nouns_verbs = 0
    matched_nouns_verbs = 0
    t1 = word_tokenize(text.lower())
    for t in t1:
        lemma_tokens.append(lemmatizer.lemmatize(t))
    tokens = pos_tag(lemma_tokens)
    for token in tokens:
        if token[1] in singular_noun or token[1] in plural_noun or token[1] in singular_verb or token[1] in plural_verb:
            total_nouns_verbs = total_nouns_verbs + 1
            if token[0] in search_words:
                matched_nouns_verbs = matched_nouns_verbs + 1

    if total_nouns_verbs == 0 or matched_nouns_verbs == 0:
        return 0
    topic_coherence = matched_nouns_verbs/total_nouns_verbs
    
    return topic_coherence



#########################################################################################################################################################################################*****MAIN*****##################################################################################################################################################################################################################################################



start_time = time.time()

# Initialization
scores = []
prompts_train = []
prompts_dict_tr = []
prompts_map_tr = {}
search_words = []
X_train = []
Y_true = []
Y_pred = []

spelling_scores = []
length_scores = []
agreement_scores = []
tense_scores = []
syntax_scores = []
text_coherence_scores = []
topic_coherence_scores = []

a = []
b = []
ci = []
cii = []
ciii = 0
di = 0
dii = []

pred = []
final_score = []

for i in range(0, 2):
    print("")

pp = pprint.PrettyPrinter(indent=4)
train_essays_dir = "../input/training/essays/"
test_essays_dir = "../input/testing/essays/"
resources_dir = "../executable/resources/"
train_essays = sorted(os.listdir(train_essays_dir))
test_essays = sorted(os.listdir(test_essays_dir))
results_dir = "../output/"
index_dir_train = "../input/training/"
indexfile_train = open(index_dir_train + "index.csv", 'r')
index_reader_train = csv.DictReader(indexfile_train, delimiter=";")

for r in index_reader_train:
    scores.append(r)
    prompts_train.append(r['prompt'])
    Y_true.append(zero_one(r['grade']))
    
indexfile_train.close()

get_search_words(prompts_train)
prompts_dict_tr = process_prompts_tr(prompts_train)

i = 0
for e in tqdm(train_essays):
    if e.endswith(".txt"):
        essay = ""
        with open(train_essays_dir + e) as f:
            essay = f.read().replace('\n', '').replace('\t', ' ')

        sentences = tokenize_sentence(essay)

        length_scores.append(evaluate_length(essay))
        spelling_scores.append(evaluate_spelling(essay))
        syntax_score = evaluate_syntax(sentences)
        agreement_scores.append(syntax_score["agreement"])
        tense_scores.append(syntax_score["tense"])
        #syntax_scores.append(syntax_score["syntax"])
        text_coherence_scores.append(evaluate_text_coherence(essay))
        
        j = prompts_dict_tr[0][i]
        with open(resources_dir + str(j) + ".txt") as f:
            search_words = f.read().replace('\n', '').replace('\t', ' ')

        topic_coherence_scores.append(evaluate_topic_coherence(essay, search_words))

        i = i + 1

with open(results_dir + "train_results.txt", 'w') as f:
    f.write("")

i = 0
for e in tqdm(train_essays):
    
    print("\nGrading " + e + '.')

    a.append(get_score_a(length_scores[i], length_scores))
    b.append(get_score_b(spelling_scores[i], spelling_scores))
    ci.append(get_score_ci(agreement_scores[i], agreement_scores))
    cii.append(get_score_cii(tense_scores[i], tense_scores))
    #ciii.append(get_score_ciii(syntax_scores[i], syntax_scores))
    #di.append(get_score_di(text_coherence_scores[i], text_coherence_scores))
    dii.append(get_score_dii(topic_coherence_scores[i], topic_coherence_scores))

    final_score.append((2 * a[i]) - b[i] + ci[i] + cii[i] + (2 * ciii) + (2 * di) + (3 * dii[i]))

    with open(results_dir + "train_results.txt", 'a') as f:
        f.write(e + ';' + str(a[i]) + ';' + str(b[i]) + ';' + str(ci[i]) + ';' + str(cii[i]) +
                ';' + str(ciii) + ';' + str(di) + ';' + str(dii[i]) + ';' + str(final_score[i]) +
                ';' + grade(final_score[i]) + '\n')

    i = i + 1



print("\n\nTraining on 100 essays dataset\n")

regr = linear_model.LogisticRegressionCV()

trn_data = open(results_dir + "train_results.txt", 'r')
res_reader = csv.reader(trn_data, delimiter=";")
# next(res_reader)  # skip header

data = [row for row in res_reader]
for row in data:
    X_train.append([int(i) for i in row[1:8]])

regr.fit(X_train, Y_true)
coef = regr.coef_
pp.pprint(coef)

final_score = []
with open(results_dir + "train_results.txt", 'w') as f:
    f.write("")

print("\n\nValidating the model on 100 essays training dataset")

i = 0
for e in tqdm(train_essays):
    
    final_score.append((coef[0][0] * a[i]) + (coef[0][1] * b[i]) + (coef[0][2] * ci[i]) + (
        coef[0][3] * cii[i]) + (coef[0][4] * ciii) + (coef[0][5] * di) + (coef[0][6] * dii[i]))

    pred.append(regr.predict([X_train[i]]))

    with open(results_dir + "train_results.txt", 'a') as f:
        f.write(e + ';' + str(a[i]) + ';' + str(b[i]) + ';' + str(ci[i]) + ';' + str(cii[i]) +
                ';' + str(ciii) + ';' + str(di) + ';' + str(dii[i]) + ';' + str(7 * int(final_score[i])) +
                ';' + str(high_low(pred[i].item())) + '\n')

    i = i + 1

print("\nPredictions on test essays dataset:\n")

# Initialization
scores = []
prompts_test = []
prompts_dict_te = []
search_words = []
X_test = []

spelling_scores = []
length_scores = []
agreement_scores = []
tense_scores = []
syntax_scores = []
text_coherence_scores = []
topic_coherence_scores = []

a = []
b = []
ci = []
cii = []
ciii = 0
di = 0
dii = []

pred = []
final_score = []

for i in range(0, 2):
    print("")

index_dir_test = "../input/testing/"
indexfile_test = open(index_dir_test + "index.csv", 'r')
index_reader_test = csv.DictReader(indexfile_test, delimiter=";")

for r in index_reader_test:
    scores.append(r)
    prompts_test.append(r['prompt'])

indexfile_train.close()

get_search_words(prompts_test)
prompts_dict_te, processed_prompts_test = process_prompts_te(prompts_test)

i = 0
for e in tqdm(test_essays):
    if e.endswith(".txt"):
        essay = ""
        with open(test_essays_dir + e) as f:
            essay = f.read().replace('\n', '').replace('\t', ' ')

        sentences = tokenize_sentence(essay)

        length_scores.append(evaluate_length(essay))
        spelling_scores.append(evaluate_spelling(essay))
        syntax_score = evaluate_syntax(sentences)
        agreement_scores.append(syntax_score["agreement"])
        tense_scores.append(syntax_score["tense"])
        #syntax_scores.append(syntax_score["syntax"])
        text_coherence_scores.append(evaluate_text_coherence(essay))

        j = prompts_map_tr[processed_prompts_test[i]]
        with open(resources_dir + str(j) + ".txt") as f:
            search_words = f.read().replace('\n', '').replace('\t', ' ')

        topic_coherence_scores.append(evaluate_topic_coherence(essay, search_words))

        i = i + 1

with open(results_dir + "test_results.txt", 'w') as f:
    f.write("")

i = 0
for e in tqdm(test_essays):

    print("\nGrading " + e + '.')

    a.append(get_score_a(length_scores[i], length_scores))
    b.append(get_score_b(spelling_scores[i], spelling_scores))
    ci.append(get_score_ci(agreement_scores[i], agreement_scores))
    cii.append(get_score_cii(tense_scores[i], tense_scores))
    #ciii.append(get_score_ciii(syntax_scores[i], syntax_scores))
    #di.append(get_score_di(text_coherence_scores[i], text_coherence_scores))
    dii.append(get_score_dii(topic_coherence_scores[i], topic_coherence_scores))

    final_score.append((coef[0][0] * a[i]) + (coef[0][1] * b[i]) + (coef[0][2] * ci[i]) + (
        coef[0][3] * cii[i]) + (coef[0][4] * ciii) + (coef[0][5] * di) + (coef[0][6] * dii[i]))

    pred.append(regr.predict([[a[i], b[i], ci[i], cii[i], ciii, di, dii[i]]]))

    with open(results_dir + "test_results.txt", 'a') as f:
        f.write(e + ';' + str(a[i]) + ';' + str(b[i]) + ';' + str(ci[i]) + ';' + str(cii[i]) +
                ';' + str(ciii) + ';' + str(di) + ';' + str(dii[i]) + ';' + str(7 * int(final_score[i])) +
                ';' + str(high_low(pred[i].item())) + '\n')

    i = i + 1
    print("Done testing " + e + '.')

print("\n\nCalculating training accuracy...\n")

trn_data = open(results_dir + "train_results.txt", 'r')
res_reader = csv.reader(trn_data, delimiter=";")
# next(res_reader)  # skip header

data = [row for row in res_reader]
for row in data:
    Y_pred.append(zero_one(row[9]))

accuracy = metrics.accuracy_score(Y_true, Y_pred)
print("The training accuracy is " + str(accuracy * 100) + "%")

print("Running Time --- %s seconds " % (time.time() - start_time))

fscore = metrics.classification_report(Y_true, Y_pred)
pp.pprint(fscore)
