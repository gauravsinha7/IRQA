
from nltk import pos_tag,word_tokenize,ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet,stopwords

class ProcessedQuestion:
    def __init__(self, question, useStemmer = False, useSynonyms = False, removeStopwords = False):
        self.question = question
        self.useStemmer = useStemmer
        self.useSynonyms = useSynonyms
        self.removeStopwords = removeStopwords
        self.stopWords = stopwords.words("english")
        self.stem = lambda k : k.lower()
        if self.useStemmer:
            ps = PorterStemmer()
            self.stem = ps.stem
        self.qType = self.determineQuestionType(question)
        self.searchQuery = self.buildSearchQuery(question)
        self.qVector = self.getQueryVector(self.searchQuery)
        self.aType = self.determineAnswerType(question)

    def determineQuestionType(self, question):
        questionTaggers = ['WP','WDT','WP$','WRB']
        qPOS = pos_tag(word_tokenize(question))
        qTags = []
        for token in qPOS:
            if token[1] in questionTaggers:
                qTags.append(token[1])
        qType = ''
        if(len(qTags)>1):
            qType = 'complex'
        elif(len(qTags) == 1):
            qType = qTags[0]
        else:
            qType = "None"
        return qType

    def determineAnswerType(self, question):
        questionTaggers = ['WP','WDT','WP$','WRB']
        qPOS = pos_tag(word_tokenize(question))
        qTag = None

        for token in qPOS:
            if token[1] in questionTaggers:
                qTag = token[0].lower()
                break

        if(qTag == None):
            if len(qPOS) > 1:
                if qPOS[1][1].lower() in ['is','are','can','should']:
                    qTag = "YESNO"
        #who/where/what/why/when/is/are/can/should
        if qTag == "who":
            return "PERSON"
        elif qTag == "where":
            return "LOCATION"
        elif qTag == "when":
            return "DATE"
        elif qTag == "what":
            # Defination type question
            # If question of type whd modal noun? its a defination question
            qTok = self.getContinuousChunk(question)
            #print(qTok)
            if(len(qTok) == 4):
                if qTok[1][1] in ['is','are','was','were'] and qTok[2][0] in ["NN","NNS","NNP","NNPS"]:
                    self.question = " ".join([qTok[0][1],qTok[2][1],qTok[1][1]])
                    #print("Type of question","Definition",self.question)
                    return "DEFINITION"

            # ELSE USE FIRST HEAD WORD
            for token in qPOS:
                if token[0].lower() in ["city","place","country"]:
                    return "LOCATION"
                elif token[0].lower() in ["company","industry","organization"]:
                    return "ORGANIZATION"
                elif token[1] in ["NN","NNS"]:
                    return "FULL"
                elif token[1] in ["NNP","NNPS"]:
                    return "FULL"
            return "FULL"
        elif qTag == "how":
            if len(qPOS)>1:
                t2 = qPOS[2]
                if t2[0].lower() in ["few","great","little","many","much"]:
                    return "QUANTITY"
                elif t2[0].lower() in ["tall","wide","big","far"]:
                    return "LINEAR_MEASURE"
            return "FULL"
        else:
            return "FULL"

    def buildSearchQuery(self, question):
        qPOS = pos_tag(word_tokenize(question))
        searchQuery = []
        questionTaggers = ['WP','WDT','WP$','WRB']
        for tag in qPOS:
            if tag[1] in questionTaggers:
                continue
            else:
                searchQuery.append(tag[0])
                if(self.useSynonyms):
                    syn = self.getSynonyms(tag[0])
                    if(len(syn) > 0):
                        searchQuery.extend(syn)
        return searchQuery

    def getQueryVector(self, searchQuery):
        vector = {}
        for token in searchQuery:
            if self.removeStopwords:
                if token in self.stopWords:
                    continue
            token = self.stem(token)
            if token in vector.keys():
                vector[token] += 1
            else:
                vector[token] = 1
        return vector

    def getContinuousChunk(self,question):
        chunks = []
        answerToken = word_tokenize(question)
        nc = pos_tag(answerToken)

        prevPos = nc[0][1]
        entity = {"pos":prevPos,"chunk":[]}
        for c_node in nc:
            (token,pos) = c_node
            if pos == prevPos:
                prevPos = pos
                entity["chunk"].append(token)
            elif prevPos in ["DT","JJ"]:
                prevPos = pos
                entity["pos"] = pos
                entity["chunk"].append(token)
            else:
                if not len(entity["chunk"]) == 0:
                    chunks.append((entity["pos"]," ".join(entity["chunk"])))
                    entity = {"pos":pos,"chunk":[token]}
                    prevPos = pos
        if not len(entity["chunk"]) == 0:
            chunks.append((entity["pos"]," ".join(entity["chunk"])))
        return chunks

    def getSynonyms(word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                w = l.name().lower()
                synonyms.extend(w.split("_"))
        return list(set(synonyms))

    def __repr__(self):
        msg = "Q: " + self.question + "\n"
        msg += "QType: " + self.qType + "\n"
        msg += "QVector: " + str(self.qVector) + "\n"
        return msg
