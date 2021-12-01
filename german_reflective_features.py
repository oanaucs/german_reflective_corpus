#!/usr/bin/env python
# coding: utf-8
import sys
from flair.models import MultiTagger
from flair.data import Sentence
from flair.models import SequenceTagger
from pathlib import Path
import subprocess
import spacy
import pandas as pd
del_var=['a_func', 'a_num', 'adja', 'adjd', 'adv','PTKANT','ET','entry','itj',
 'advers', 'all_toks', 'although', 'an', 'art', 'artikles', 'aux', 'auximp', 'auxinf', 'auxpp',
 'average_chars', 'average_tokens', 'c', 'card', 'chars', 'chars_sum', 'claims', 'clamy_words',
 'coco', 'comma', 'comp', 'comparisons', 'compl', 'comple', 'cond', 'conditions', 'cons', 'contrast',
 'coor', 'count', 'cpt', 'cpt1', 'cpt2', 'd', 'd_c', 'damit', 'dem', 'dempr', 'dependancy', 
 'discourse_markers', 'distancing', 'dodo', 'e', 'eins', 'filtered', 'final', 
 'find', 'finite_verbs_plus', 'fla', 'fla2', 'flair_tags', 'fm', 'for_tagging',
 'grund_da_deshalb', 'header', 'i', 'indefdet', 'indpro', 'interadvrel', 'interdet',
 'interpro', 'interprorel', 'kokom', 'kon', 'konj', 'kous', 'lemma', 'lemmas', 'lex', 
 'lexical_diversity', 'lexical_words_number', 'm', 'markers', 'matrix', 'modal', 
 'modal_verbs', 'modalinf', 'modality_words', 'modalpp', 'morph', 'morphology', 'mpos', 
 'na', 'ne', 'neg', 'nes', 'new_tokens', 'nextpunct', 'nn', 'number_gender', 'o', 
'orth', 'out', 'output', 'p', 'par_analysis', 'path_in_str', 
'pathlist', 'patterns', 'patterns2', 'pd', 'percent_lexical',
         'percent_stw', 'personalizing', 'pos', 'pos2', 'pper', 'pposs',
         'prefix', 'prelat', 'prep', 'previouspunct', 'prf',
         'pronouns', 'ptka', 'punct', 'purpose', 'quantity', 'question_words',
         'quit', 'r1', 're', 'reason', 'ref', 'reflxive_adj', 'rel', 'relat', 'relpro', 'root',
         'sim', 'simple', 'sodass', 'spacy', 'statistics', 'stopWords', 'stopwords', 'stw',
         'stw_number', 'sub', 'subprocess', 'suffix', 'supports', 'sys', 't', 'tag', 'tag2', 'tagged','tags', 
         'tc', 'temp', 'temp1', 'temp2', 'temp_nes', 'tempd', 'tempp', 'text',
         'texts', 'time', 'token', 'token_quant', 'tokens', 'tree', 'trunc',
         'two_part', 'um', 'um_contructions', 'umlaut', 'verb', 'verbimp',
         'verbinf', 'verbpp', 'verbzu', 'weildenn', 'xml', 'zuso', 'zwei']
long=['für wahr halten', 'verlasse dich auf', 'gewiss sein','einen Standpunkt einnehmen', 'Widerspruch einlegen',
      'Glauben schenken', 'Glauben haben', 'keinen Zweifel haben', 'den Glauben bewahren', 'Vertrauen schenken',
'ins Auge fassen','rechnen Sie mit',' Gewicht geben', 'sicher sein','überzeugt sein von', 'leichtgläubig sein', 'der Meinung sein',
      'zählen auf' 'einer Meinung sein'  'für eine Tatsache', 'frei von Zweifeln']
add_modal=['tatsächlich', 'kategorisch', 'definitiv', 'zweifellos', 'positiv', 'genau', 'wirklich', 'sicher', 'wirklich', 'bedingungslos', 'unzweifelhaft', 'schlüssig', 'entschieden', 'entschlossen', 'einfach', 'unzweideutig', 'konsequent', 'ständig', 'häufig', 'gelegentlich', 'in Abständen', 'gelegentlich', 'von Zeit zu Zeit', 'hier und da', 'sporadisch', 'hin und wieder', 'jetzt und dann', 'aus und ein', 'bei Gelegenheit', 'ab und zu', 'periodisch', 'wiederkehrend', 'kaum', 'kaum', 'unregelmäßig', 'selten', 'fast nie', 'besonders', 'außerordentlich', 'extrem', 'fein', 'fast nie', 'wenig', 'bemerkenswert', 'bemerkenswert', 'so gut wie nie', 'einmalig', 'ungewöhnlich', 'ungewöhnlich', 'unregelmäßig', 'nicht oft', 'ungewöhnlich', 'fett', 'kategorisch', 'klar und deutlich', 'definitiv', 'eindeutig', 'eindeutig', 'offensichtlich', 'greifbar', 'positiv', 'präzise', 'ausgeprägt', 'spezifisch', 'geradlinig', 'unzweideutig', 'unzweideutig', 'unmissverständlich', 'klar definiert', 'hörbar', 'klar und deutlich', 'vollständig', 'knackig', 'bestimmt', 'unterscheidbar', 'geradezu', 'ausdrücklich', 'festgelegt', 'unverblümt', 'vollständig', 'grafisch', 'einschneidend', 'markiert', 'genau', 'nicht vage', 'besonders', 'deutlich', 'klingelnd', 'stark', 'scharf', 'silhouettiert', 'greifbar', 'undeutlich', 'sichtbar', 'anschaulich', 'gut geerdet', 'gut beschriftet', 'überzeugt', 'positiv', 'sicher', 'durchsetzungsfähig', 'sicher', 'gläubig', 'ruhig', 'sicher', 'fraglos', 'sanguinisch', 'zufrieden', 'sicher', 'selbstbewusst', 'unbesorgt', 'ungestört', 'unbesorgt', 'unbesorgt', 'unbeirrt', 'unerschüttert', 'scheinbar', 'sicher', 'definitiv', 'deutlich', 'offensichtlich', 'offensichtlich', 'offenkundig', 'deutlich', 'positiv', 'genau', 'scheinbar', 'sicher', 'zweifelsohne', 'akut', 'hörbar', 'zweifellos', 'unübersehbar', 'entschieden', 'erkennbar', 'unbestreitbar', 'unbestreitbar', 'unzweifelhaft', 'einleuchtend', 'offenkundig', 'auffallend', 'merklich', 'offenkundig', 'offenkundig', 'durchdringend', 'spürbar', 'auffallend', 'rein', 'erkennbar', 'scharf', 'klangvoll', 'grundlegend', 'entscheidend', 'entscheidend', 'grundlegend', 'unabdingbar', 'unverzichtbar', 'zwingend', 'erforderlich', 'überragend', 'erforderlich', 'wichtig', 'unvermeidlich', 'dringend', 'lebenswichtig', 'verbindlich', 'kardinal', 'Chef', 'zweckmäßig', 'Voraussetzung', 'drängend', 'primär', 'Haupt', 'erforderlich', 'ganz wichtig', 'unterm Strich', 'zwingend', 'obligatorisch', 'de rigueur', 'elementar', 'zwingend', 'obliegend', 'bedeutsam', 'Name des Spiels', 'notwenig', 'obligatorisch', 'Quintessenz', 'angegeben', 'gewünscht', 'vermeidbar', 'vergeblich', 'unentgeltlich', 'irrelevant', 'unnötig', 'überflüssig', 'überflüssig', 'unnötig', 'nutzlos', 'wertlos', 'zufällig', 'zusätzlich', 'nebensächlich', 'beiläufig', 'grundlos', 'zufällig', 'entbehrlich', 'überflüssig', 'exorbitant', 'entbehrlich', 'entbehrlich', 'extrinsisch', 'unwesentlich', 'verschwenderisch', 'nicht obligatorisch', 'unwesentlich', 'optional', 'verschwenderisch', 'üppig', 'beliebig', 'überflüssig', 'überflüssig', 'unaufgefordert', 'unkritisch', 'unerwünscht', 'unwesentlich', 'unerwünscht']
 #------- models loading
nlp = spacy.load("de_dep_news_trf")
tagger = MultiTagger.load(['de-ner','de-pos'])
import nltk
nltk.download('stopwords')
 #------- initial taggers
def print_ann(text):
    doc = nlp(text)
    tokenized=[token.text for token in doc]
    tokenized2=[tok for tok in tokenized if tok.count(' ')<1 and '\t' not in tok]
    lemmas=[token.lemma_ for token in doc]
    lemmas=[tok for tok in lemmas if tok.count(' ')<1 and '\t' not in tok]
    dep=[token.dep_ for token in doc]
    dep=[dep[tok] for tok in range(len(dep)) if tokenized[tok].count(' ')<1 and '\t' not in tokenized[tok]]
    sentence = Sentence(tokenized2)
    tagger.predict(sentence)
    s=sentence.to_tagged_string().split()[1::2]
    info=[tokenized2,lemmas, dep,s]
    return info

 #download file 
 #change file.tsv into your file
with open('file.tsv','r') as path:
    pathlist=list(path)
    print(pathlist[0])
    pathlist=[e.split('\t') for e in pathlist]
    sentences=[e[-3] for e in pathlist[1:]]
    labels=[e[-2] for e in pathlist[1:]]
    meta=['\t'.join(e[0:-2]) for e in pathlist[1:]]
    sentiment=[e[-1].replace('\n','') for e in pathlist[1:]]
    print(sentences[1])
    print(labels[1])
    suffix=['ref' if e.strip()=='0' else 'nonref' for e in labels]
    final=[]
    for sent in sentences:
        an=print_ann(sent)
        final.append(an)

cptthem=0
ff=[]
for file in range(len(final)):
    tokens=[final[file][0]]
    new_tokens=[[i for i in e if i.count(' ')<1 and '\t' not in i] for e in tokens]
    def number_gender():
        import subprocess
        subprocess.call(['./RFTagger/src/rft-annotate','./RFTagger/lib/german.par','./RFTagger/test/'+suffix[file]+'.txt','pronouns_'+suffix[file]+'.txt'])
    with open('./RFTagger/test/'+suffix[file]+'.txt','w',encoding='utf-8') as for_tagging:
        for e in new_tokens:
            for token in e:
                for_tagging.write(str(token)+'\n')
            for_tagging.write('\n')
    number_gender()
    with open('pronouns_'+suffix[file]+'.txt') as tagged:
        tagged=list(tagged)
        tag=[e.replace('\n','').split('\t') for e in tagged]
    tag2=[]
    temp=[]
    for e in tagged:
        if e!='\n':
            temp.append(e.replace('\n','').split('\t')[1])
        else:
            tag2.append(temp)
            temp=[]
    tokens=[final[file][0]]
    lemmas=[final[file][1]]
    dependancy=[final[file][2]]
    morphology=tag2
    flair_tags=[final[file][3]]
    find=[1 for e in range (len(tokens[0]))  if flair_tags[0][e]==tokens[0][e]]
    if sum(find)==len(flair_tags[0]):
        sentence = Sentence(tokens[0])
        tagger.predict(sentence)
        flair_tags=[sentence.to_tagged_string().split()[1::2]]
        print(flair_tags,'ft?')
        print(tokens,'toks')
        print(final[file])
    nes=[]
    pos=[]
    temp=[]
    temp_nes=[]
    for  o in flair_tags:
        for e in range(len(o)):
            if '/' in o[e]:
                NE=o[e].replace('<','').replace('>','').split('/')[0]
                POS=o[e].replace('<','').replace('>','').split('/')[1]
                temp.append(POS)
                temp_nes.append(NE)
            else:
                temp.append(o[e].replace('<','').replace('>',''))
                temp_nes.append('-')

        pos.append(temp)
        temp=[]    
        nes.append(temp_nes)
        temp_nes=[]
        
    punct=['.','!','?','...']
    dependancy=[[i for i in e if i!=''] for e in dependancy]
    morph=[[i.split('.') for i in e] for e in morphology]
    pos2=[[i[0] for i in e] for e in morph]
    lemma=[]
    temp=[]
    for e in lemmas:
        for i in e:
            if i.count(' ')==1 or i.count(' ')==2:
                t=i.replace(' ','')
                if t!='':
                    temp.append(i)
            else:
                if i.count(' ')<1 and '\t' not in i:
                    temp.append(i)
        lemma.append(temp)
        temp=[]
    texts=[' '.join(e) for e in new_tokens]
    for e,i,o,m in zip(new_tokens,pos,lemma,morph):
        if '$' not in i[-1] or e[-1] not in punct:
            i.append('$.')
            e.append('.')
            o.append('.')
            m.append(['SYM', 'Pun', 'Sent'])
    texts=[' '.join(e) for e in new_tokens]
    matrix=[]
    all_toks=[]
    for e in new_tokens:
        for i in e:
            all_toks.append(i)
    #######stats#######
    token_quant=[len(e) for e in new_tokens]
    matrix.append(sum(token_quant))
    chars=[[len(i) for i in e] for e in new_tokens]
    chars_sum=[sum(e) for e in chars]
    matrix.append(sum(chars_sum))
    average_tokens=sum(token_quant)/len(token_quant)
    #average amount of tokens per paragraph
    import statistics
    #median amount of tokens
    average_chars=sum(chars_sum)/len(chars_sum)
    #average amount of characters per paragraph
    #median amount of characters
    def lexical_diversity(text):
        return len(set(text)) / len(text)
    lex=lexical_diversity(all_toks)
    #lexical diversity index
    from nltk.corpus import stopwords
    stopWords = set(stopwords.words('german'))
    stw=[[e for e in toks if e in stopWords] for toks in new_tokens]
    filtered=[[e for e in toks if e not in stopWords] for toks in new_tokens]
    stw_number=sum([len(e) for e in stw])
    lexical_words_number=sum([len(e) for e in filtered])
    matrix.append(stw_number)
    matrix.append(lexical_words_number)
    #'amount of stop words'
    #amount of lexical words
    percent_stw=(stw_number*100)/len(all_toks)
    percent_lexical=(lexical_words_number*100)/len(all_toks)
    #percent of stop words
    #percent of lexical words
    #'Adjectives'
    adja=[e.count('ADJA') for e in pos]
    adjd=[e.count('ADJD') for e in pos]
    matrix.append(sum(adja)+sum(adjd))
    na=[e.count('NA') for e in pos]
    #'Adverbs'
    adv=[e.count('ADV') for e in pos]
    matrix.append(sum(adv))
    #'Prepositions'
    prep=[e.count('APPR')+e.count('APPO')+e.count('APZR')+e.count('KOUI') for e in pos]
    matrix.append(sum(prep))
    art=[e.count('ART') for e in pos]
    dem=[e.count('PDAT') for e in pos]
    dempr=[e.count('PDS') for e in pos]
    indefdet=[e.count('PIAT') for e in pos]
    matrix.append(sum(dem)+sum(dempr)+sum(indefdet))
    #'Numbers'
    card=[e.count('CARD') for e in pos]
    matrix.append(sum(card))
    #Foreign words'
    fm=[e.count('FM') for e in pos]
    matrix.append(sum(fm))
    #'Interjections'
    itj=[e.count('ITJ') for e in pos]
    #'Conjunctions'
    kon=[e.count('KON') for e in pos]
    #'co-ordinating conjunction	oder ich bezahle nicht')
    kokom=[e.count('KOKOM') for e in pos]
    ##'comparative conjunction or particle	er arbeitet als Straßenfeger, so gut wie du')
    #'amount of co-ordinating conjunctions'
    matrix.append(sum(kon)+ sum(kokom))
    kous=[e.count('KOUS') for e in pos]
    #'amount of subordinating conjunctions',sum(kous),'subordinating conjunction	weil er sie gesehen hat'
    matrix.append(sum(kous))
    #'Nouns'
    ne=[e.count('NE') for e in pos]
    matrix.append(sum(ne))
    #'proper nouns'
    nn=[e.count('NN') for e in pos]
    #commun nouns'
    matrix.append(sum(ne)+sum(nn))
    #'Pronouns'
    indpro=[e.count('PIS') for e in pos]
    #indefinite pronoun'
    pper=[e.count('PPER') for e in pos]
    #'personal pronoun'
    prf=[e.count('PRF') for e in pos]
    #'reflexive pronoun'
    pposs=[e.count('PPOSS')+e.count('PPOSAT') for e in pos]
    #'possessive pronoun and determiners'
    prelat=[e.count('PRELAT')+e.count('PRELS') for e in pos]
    #'relative pronouns'
    matrix.append(sum(indpro)+sum(pper)+sum(prf)+sum(pposs)+sum(prelat))
    #'particles'
    ptka=[e.count('PTKA') for e in pos]
    #'particle with adjective or adverb	am besten, zu schnell, aufs herzlichste')
    PTKANT=[e.count('PTKANT') for e in pos]
    #'answer particle	ja, nein'
    neg=[e.count('PTKNEG') for e in pos]
    matrix.append(sum(neg))
    #negative particle	nicht'
    zuso=[e.count('PTKREL')+e.count('PTKZU') for e in pos]
    #infinitive particle	zu,indeclinable relative particle	so'
    prefix=[e.count('PTKVZ') for e in pos]
    #separable prefix	sie kommt an'
    #'interrogations'
    interpro=[e.count('PWS') for e in pos]
    #interrogative pronoun	wer kommt?'
    interdet=[e.count('PWAT') for e in pos]
    #'interrogative determiner	 welche Farbe?'
    interdet=[e.count('PWAV') for e in pos]
    #'interrogative determiner	 interrogative adverb	wann kommst du?'
    interadvrel=[e.count('PWAVREL') for e in pos]
    #'interrogative adverb used as relative	der Zaun, worüber sie springt'
    interprorel=[e.count('PWREL') for e in pos]
    #'interrogative pronoun used as relative	etwas, was er sieht'
    trunc=[e.count('TRUNC') for e in pos]
    #TRUNC	truncated form of compound	Vor- und Nachteile'

    #'Verbs'

    aux=[e.count('VAFIN') for e in pos]
    #'finite auxiliary verb	sie ist gekommen'
    auximp=[e.count('VAIMP') for e in pos]
    #'imperative of auxiliary	sei still!'
    auxinf=[e.count('VAINF') for e in pos]
    #'infinitive of auxiliary	er wird es gesehen haben'
    auxpp=[e.count('VAPP') for e in pos]
    #'past participle of auxiliary	sie ist es gewesen'
    modal=[e.count('VMFIN') for e in pos]
    #'finite modal verb	sie will kommen'
    modalinf=[e.count('VMINF') for e in pos]
    #'infinitive of modal	er hat es sehen müssen'
    modalpp=[e.count('VMPP') for e in pos]
    #past participle of auxiliary	sie hat es gekonnt'
    verb=[e.count('VVFIN') for e in pos]
    #'finite full verb'
    verbimp=[e.count('VVIMP') for e in pos]
    #'imperative of full verb	bleibt da!'
    verbinf=[e.count('VVINF') for e in pos]
    #'infinitive of full verb	er wird es sehen'
    verbzu=[e.count('VVIZU') for e in pos]
    #'infinitive with incorporated zu	sie versprach aufzuhören'
    verbpp=[e.count('VVPP') for e in pos]
    #'past participle of full verb	sie ist gekommen'
    matrix.append(sum(aux)+sum(auximp)+sum(auxinf)+sum(modal)
                  +sum(modalinf)+sum(modalpp)+sum(verb)+sum(verbimp)+sum(verbinf)+sum(verbzu)+sum(verbpp))
    header=['n_tokens','n_chars','n_stopwords','n_lexical_w','n_adj','n_adv','n_prep','n_demostrativ','n_numerals','foreign_w','coo_conj',
           'sub_conj','n_prop_nouns','n_nouns','n_pronouns','negations','n_verbs','sub_purpose','len_purpose','sub_reason','len_reason','sub_cond','sub_konsek','sub_temp',
            'sub_modal','sub_rel','sub_konsess','sub_advers','modals','ich_fin_v_constr','adj_sein','complex_sent','simpl_sent','claims','supports','disourse_markers','konjuktive',
    'modality_words','personalizing','distansing','present','future','past','label_ref','label_sent']
    
    par_analysis=[0 for e in new_tokens]
    
    #subordinates of purpopse um zu inf
    um_contructions=[]
    temp=[]
    cpt=[]

    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i].lower()=='um':
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                zu_exists=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]=='zu']
                search_vvizu=[f for f in range (len(p[i:nextpunct])) if p[i:nextpunct][f]=='VVIZU']
                if search_vvizu!=[]:
                    vvizu=search_vvizu[0]+i
                    if e[vvizu+1]=='und':
                        zu_exists_2=[f for f in range(len(e[vvizu+2:nextpunct])) if e[vvizu+2:nextpunct][f]=='zu']
                        if zu_exists_2!=[]:
                            fullsearch=zu_exists_2[0]
                            if 'INF'in p[vvizu+2+fullsearch+1]:
                                temp.append(e[i:vvizu+2+fullsearch+2])
                    else:
                        temp.append(e[i:vvizu+1])

                elif zu_exists!=[]:
                    search=zu_exists[0]+i
                    if 'INF' in p[search+1]:
                        if e[search+2]=='und':
                            zu_exists_2=[f for f in range(len(e[search+2:nextpunct])) if e[search+2:nextpunct][f]=='zu']
                            if zu_exists_2!=[]:
                                fullsearch=zu_exists_2[0]
                                if 'INF'in p[search+2+fullsearch+1]:
                                    temp.append(e[i:search+2+fullsearch+2])
                        else:
                            temp.append(e[i:search+2])

        um_contructions.append(temp)
        cpt.append(len(temp))
        temp=[]
    cpt_l=0
    for e in um_contructions[0]:
        cpt_l+=len(e)
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    #subordinates of purpopse damit
    purpose=int(sum(cpt))
    damit=[]
    temp=[]
    cpt=[]
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i]=='damit' and p[i]=='KOUS':
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                comma=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]==',']
                if comma!=[]:
                    temp.append(e[i:comma[0]])           
                else:
                    temp.append(e[i:nextpunct])
        damit.append(temp)
        cpt.append(len(temp))
        temp=[]
    for e in damit[0]:
        cpt_l+=len(e)
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    purpose=purpose+sum(cpt)
    matrix.append(purpose)
    matrix.append(cpt_l)
    #subordinates of reason
    reason=0
    import re
    cpt=[]
    grund_da_deshalb=[]
    temp=[]
    for e in texts:
        r1 = re.findall(r"der grund dafür.+?\.|der grund für.+?\.|ohne besonderen grund.+?\.|aus diesem grund.+?\.|daher.+?\.",e.lower())
        if r1!=[]:    
            for o in r1:
                temp.append(o.split(' '))
                grund_da_deshalb.append(temp)
                cpt.append(len(r1))
                temp=[]
        else:
            grund_da_deshalb.append([])
    # Der Grund dafür
    # Der Grund für ohne besonderen Grund Aus diesem Grund
    cpt_l=0
    for e in grund_da_deshalb[0]:
        cpt_l+=len(e)
    if sum(cpt)!=0:
        reason=int(sum(cpt))
    else:
        cpt=[0 for e in new_tokens]
        reason=int(sum(cpt))
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    
    #subordinates of reason continuation
    weildenn=[]

    temp=[]
    cpt=[]
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i].lower()=='weil' or e[i].lower()=='denn' or e[i].lower()=='da' or e[i].lower()=='deshalb' or  e[i].lower()=='wegen' or e[i].lower()=='durch':
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct]
                if nextpunct!=[]:
                    nextpunct=nextpunct[0]+i
                    comma=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]==',']
                    if comma!=[]:
                        temp.append(e[i:comma[0]])
                    else:
                        temp.append(e[i:nextpunct])
                else:
                    temp.append(e[i:])
        cpt.append(len(temp))
        weildenn.append(temp)
        temp=[]
    for e in weildenn[0]:
        cpt_l+=len(e)
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    reason+=sum(cpt)
    matrix.append(reason)
    matrix.append(cpt_l)
    #'subordinates of condition
    temp=[]
    cond=[]
    pronouns=['ich','wir']
    cpt=[]
    c=0
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i].lower()=='wenn' and e[i-1] in punct:
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                comma=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]==',']
                if comma!=[]:
                    temp.append(e[i:comma[0]])
                else:
                    temp.append(e[i:nextpunct])
            elif e[i].lower()=='wenn' or e[i].lower()=='falls':
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                temp.append(e[i:nextpunct])
        cond.append(temp)
        cpt.append(len(temp))
        temp=[]
        c+=1
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    matrix.append(sum(cpt))
    
    #konsekutiv subordinates

    sodass=[]
    temp=[]
    cpt=[]
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i]=='so' and e[i+1]=='dass':
                punct=['.','!','?']
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                temp.append(e[i:nextpunct])
        sodass.append(temp)
        cpt.append(len(temp))
        temp=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    matrix.append(sum(cpt))    
    time=[]
    comparisons=[]
    tc=[]
    punct=['.','!','?','...']
    temp=[]
    cpt=[]
    coco=0
    dodo=0
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i].lower()=='seit' or e[i].lower()=='seitdem':
                coco+=1
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                comma=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]==',']
                if comma!=[]:
                    comma=comma[0]+i
                    if e[i+1]=='dem' or e[i+1]=='der' or e[i+1]=='den':
                        verb=[f for f in range(len(e[i:nextpunct])) if "FIN" in p[i:nextpunct][f]][0]+i
                        if verb>comma:
                            temp.append(e[i:comma])
                        else:
                            temp.append(e[i:verb])
                    else:
                        temp.append(e[i:comma])


                else:
                    temp.append(e[i:nextpunct])
            elif e[i].lower()=='bis' or e[i].lower()=='während' or e[i].lower()=='solange' or e[i].lower()=='eher' or e[i].lower()=='nachdem' or e[i].lower()=='bevor':
                dodo+=1
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                comma=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]==',']
                if comma!=[]:
                    comma=comma[0]+i
                    verb=[f for f in range(len(e[i:nextpunct])) if "FIN" in p[i:nextpunct][f]]
                    if verb!=[]:
                        verb=verb[0]+i
                        if verb>comma:
                            temp.append(e[i:comma])
                        else:
                            temp.append(e[i:verb+1])
                    else:
                        temp.append(e[i:comma])
                else:
                    temp.append(e[i:nextpunct])


            elif e[i].lower()=='damals' or e[i].lower()=='sobald': 
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                comma=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]==',']
                if comma!=[]:
                    comma=comma[0]+i
                    temp.append(e[i:comma])
                else:
                    temp.append(e[i:nextpunct])
            elif e[i].lower()=='zur' and e[i+1].lower()=='der' and e[i+2].lower()=='Zeit' : 
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
        cpt.append(len(temp))
        time.append(temp)
        temp=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    matrix.append(sum(cpt))

    modal=[]
    temp=[]
    cpt=[]
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i].lower()=='indem':
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                if e[i][0]=='I':
                    comma=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]==',']
                    if comma!=[]:
                        comma=comma[0]+i
                        temp.append(e[i:comma])
                    else:
                        verb=[f for f in range(len(e[i:nextpunct])) if "FIN" in p[i:nextpunct][f]][0]+i
                        temp.append(e[i:verb])
                else:
                    verb=[f for f in range(len(e[i:nextpunct])) if "FIN" in p[i:nextpunct][f]][0]+i
                    temp.append(e[i:verb+1])
            elif e[i].lower()=='als' and e[i+1].lower()=='ob':
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                verb=[f for f in range(len(e[i:nextpunct])) if "FIN" in p[i:nextpunct][f]][0]+i
                temp.append(e[i:verb+1])
        cpt.append(len(temp))
        modal.append(temp)
        temp=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    matrix.append(sum(cpt))

    relpro=['PAVREL','PRELAT','PRELS','PWAVREL','PWREL']
    rel=[]
    temp=[]
    cpt=[]
    cpt1=0
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if p[i] in relpro:
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                comma=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]==',']
                if e[i-1]!=',' and e[i-1]!='(' and e[i-1]!='-':
                    if comma!=[]:
                        comma=comma[0]+i
                        
                        verb=[f for f in range(len(e[i:nextpunct])) if "FIN" in p[i:nextpunct][f]]
                        if verb!=[]:
                            verb=verb[0]+i
                            if verb>comma:
                                temp.append(e[i-1:comma])
                            else:
                                temp.append(e[i-1:verb+1])
                        else:
                            temp.append(e[i-1:comma])
                    else:
                        temp.append(e[i-1:nextpunct])
                else:
                    if comma!=[]:
                        comma=comma[0]+i
                        verb=[f for f in range(len(e[i:nextpunct])) if "FIN" in p[i:nextpunct][f]]
                        if verb!=[]:
                            verb=verb[0]+i
                            if verb>comma:
                                temp.append(e[i:comma])
                            else:
                                temp.append(e[i:verb+1])
                        else:
                            temp.append(e[i:comma])
                    else:
                        temp.append(e[i:nextpunct])


        cpt1+=1
        cpt.append(len(temp))
        rel.append(temp)
        temp=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    matrix.append(sum(cpt))
    #'“Obwohl” and “Obgleich” (Concessive Clauses)

    cons=[]
    temp=[]
    cpt=[]
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i].lower()=='obwohl' or e[i].lower()=='allerdings' or e[i].lower()=='dennoch' or e[i].lower()=='obgleich' or e[i].lower()=='enngleich' or e[i].lower()=='obschon' or e[i].lower()=='obzwar' or e[i].lower()=='wiewohl':
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                comma=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]==',']
                if comma!=[]:
                    comma=comma[0]+i
                    verb=[f for f in range(len(e[i:nextpunct])) if "FIN" in p[i:nextpunct][f]]
                    if verb!=[]:
                        verb=verb[0]+i
                        if verb>comma:
                            temp.append(e[i:comma])
                        else:
                            temp.append(e[i:verb+1])
                    else:
                        temp.append(e[i:comma])
                else:
                    temp.append(e[i:nextpunct])
            elif e[i].lower()=='trotzdem':
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                temp.append(e[i:nextpunct])
            elif e[i].lower()=='wenn' and e[i+1].lower()=='auch':
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                temp.append(e[i:nextpunct])



        cpt.append(len(temp))
        cons.append(temp)
        temp=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    matrix.append(sum(cpt))
    #'consessive'
    ##'im Gegensatz dazu','im Kontrast dazu',konträr dazu'im Unterschied dazu 'im Gegenteil'
    #adversive clause
    adverses=['wohingegen','wenngleich','wobei','wogegen','wohingegen','alldieweil',
              'andererseits', 'dagegen','demgegenüber','dieweil','hingegen','hinwieder',
              'hinwiederum', 'dahingegen']
    advers=[]
    temp=[]
    cpt=[]
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i].lower() in adverses:
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                comma=[f for f in range(len(e[i:nextpunct])) if e[i:nextpunct][f]==',']
                if comma!=[]:
                    comma=comma[0]+i
                    verb=[f for f in range(len(e[i:nextpunct])) if "FIN" in p[i:nextpunct][f]]
                    if verb!=[]:
                        verb=verb[0]+i
                        if verb>comma:
                            temp.append(e[i:comma])
                        else:
                            temp.append(e[i:verb+1])
                    else:
                        temp.append(e[i:comma])
                else:
                    temp.append(e[i:nextpunct])
        cpt.append(len(temp))
        advers.append(temp)
        temp=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    additional=['im gegensatz dazu','im kontrast dazu','konträr dazu','im unterschied dazu',
                'im gegenteil']
    adds=0
    for e in texts:
        find=[1 for w in additional if w in e.lower()]
        adds+=sum(find)
    adds+=sum(cpt)
    #adversive clause
    matrix.append(adds)
    mpos=[[i[0] for i in e] for e in morph]
    modal_verbs=[]
    temp=[]
    cpt=[]

    artikles=['der','die','das','den','dem','des','daraus','nie','ein','sein','täglich','nicht','einfach','weiter','es','meine','eine','einen','uns','mehr','darin','alle','zur','sie','einmal','als','auch','nur','im','laut','nur','drinnen','bei','damit','in',',','zu','beide','über','mit','besonders','diese','darauf','einige','auf']
    for tags,text in zip(pos,new_tokens):
        for e in range(len(tags)):
            if tags[e]=='VMFIN' and text[e].lower() not in artikles:
                if text[e-1].lower()=='ich' or text[e-1].lower()=='wir':
                    nextpunct=[f for f in range(len(text[e:])) if text[e:][f] in punct][0]+e
                    verb=[f for f in range(len(text[e+1:nextpunct])) if "INF" in tags[e+1:nextpunct][f]]
                    if verb!=[] and nextpunct>verb[0]+e:
                        if text[e+1]in punct:
                            temp.append(text[e-2:verb[0]+e])
                        else:

                            temp.append(text[e-1:verb[0]+e+2])
                    else:
                        if text[e+1]==text[nextpunct]:
                            temp.append(text[e-2:nextpunct])
                        else:
                            temp.append(text[e-1:nextpunct])

            elif tags[e]=='VMINF':
                temp.append(text[e-1:e+1])
        modal_verbs.append(temp)
        cpt.append(len(temp))
        temp=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    matrix.append(sum(cpt))
    mpos=[[i[0] for i in e] for e in morph]
    finite_verbs_plus=[]
    temp=[]
    cpt=[]
    artikles=['der','die','das','den','dem','des','ein','eine','einen','uns','mehr','darin','alle','zur','sie','einmal','als','auch','nur','im','laut','nur','drinnen','bei','damit','in',',','zu','beide','über','mit','besonders','diese','darauf','einige','auf']
    for tags,text in zip(pos,new_tokens):
        for e in range(len(tags)):
            if tags[e]=='VVFIN' and text[e].lower() not in artikles:
                if text[e-1].lower()=='ich' or text[e-1].lower()=='wir':
                    if text[e-1].lower()+'_'+text[e]+'_'+tags[e+1] not in temp:
                        temp.append(text[e-1].lower()+'_'+text[e]+'_'+tags[e+1])
                    elif text[e-1].lower()+'_'+text[e] not in temp:
                        temp.append(text[e-1].lower()+'_'+text[e])
                    elif text[e-1].lower()+'_'+text[e]+'_'+text[e+1] not in temp:
                        temp.append(text[e-1].lower()+'_'+text[e]+'_'+text[e+1])
        finite_verbs_plus.append(temp)
        cpt.append(len(temp))
        temp=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    matrix.append(sum(cpt))
    reflxive_adj=[]
    patterns=[['VAFIN', 'PPER', 'ADJD'],['PTKNEG', 'ADV', 'ADJD'],['VAFIN', 'PIS', 'ADJD'],['ART', 'NN', 'ADJD'],['PPER', 'VAFIN', 'ADJD'],['PPER', 'ADV', 'ADJD'],['VAFIN', 'ADV', 'ADJD']]            
    patterns2=['VAFIN', 'ADJD']
    cpt=[]
    for e,p in zip(lemma,pos):
        for i in range(len(e)):
            if 'ADJ' in p[i]:
                if p[i-1:i+1] == patterns2 and e[i-1]=='sein':
                    temp.append(e[i-1:i+1])
                elif p[i-2:i+1] in patterns:
                    temp.append(e[i-2:i+1])
        reflxive_adj.append(temp)
        cpt.append(len(temp))
        temp=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    matrix.append(sum(cpt))
    # complex and simple sentences count
    coor=['und','aber','denn','oder','sondern','beziehungsweise','doch','jedoch','entweder','weder','einerseits','andererseits']
    two_part=['mal','teils']
    simple=[]
    comple=[]
    sim=0
    comp=0
    temp1=[]
    temp2=[]
    sub=['bevor', 'nachdem', 'ehe', 'seit, seitdem', 'während', 'als', 'wenn', 'wann', 'bis', 'obwohl', 'sooft', 'sobald', 'solange', 'da', 'indem', 'weil', 'ob', 'falls', 'wenn', 'dass', 'sodass', 'damit','damals','deshalb','wegen','wiewohl','obzwar','obschon','enngleich','obgleich','allerdings','dennoch','trotzdem']
    question_words=['wo','was','wer','wo','wen','wie','warum','wohin','woher','weshalb']
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i].lower() in sub:
                temp2.append(e[i])
                comp+=1
            elif e[i].lower() in coor:
                temp1.append(e[i])
                sim+=1
            elif e[i].lower() in two_part:
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                if e[i] in e[i+1:nextpunct]:
                    temp1.append(e[i])
                    sim+=1
            elif e[i] in question_words:
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct][0]+i
                if e[nextpunct]!='?':
                    temp2.append(e[i])
                    comp+=1
        simple.append(temp1)
        comple.append(temp2)
        temp1=[]
        temp2=[]
    um=sum([len(e) for e in um_contructions])
    relat=sum([len(e) for e in rel])
    compl=comp+relat+um
    #'complex sentences'
    #'simple sentences'
    cpt1=[len(e) for e in simple]
    cpt2=[len(e) for e in comple]
    par_analysis = [a + b for a, b in zip(cpt1, par_analysis)]
    par_analysis = [a + b for a, b in zip(cpt2, par_analysis)]
    matrix.append(compl)
    matrix.append(sim)
    
    # argumentative structures
    clamy_words=['wahr', 'gewiss','Standpunkt', 'Widerspruch','Glauben', 'Zweifel', 'bewahren', 'Vertrauen',
    'rechnen','Gewicht', 'sicher','überzeugt', 'leichtgläubig', 'Meinung', 'Tatsache', 'Zweifeln','annehmen', 'betrachten', 'erwarten', 'fühlen', 'vermuten', 'beurteilen', 'erkennen', 'sehen', 'schließen', 'erachten', 'vorstellen', 'wertschätzen', 'schätzen', 'vorstellen', 'vermuten', 'betrachten', 'spüren', 'widersprechen', 'vermuten', 'vermuten', 'überzeugt', 'glauben', 'akzeptieren', 'zugeben', 'betrachten', 'vermuten', 'vertrauen', 'bejahen', 'postulieren', 'voraussetzen',  'glauben', 'schwören', 'erwarten', 'vermuten', 'spekulieren', 'vermuten', 'anerkennen', 'zugeben', 'zulassen', 'zugestehen', 'anerkennen', 'zugeben', 'anerkennen', 'sich einigen', 'argumentieren', 'debattieren', 'trotzen', 'bestreiten', 'widersprechen', 'streiten', 'konfrontieren', 'kämpfen', 'protestieren', 'widerstehen', 'widersprechen', 'kontern', 'missbilligen', 'widersprechen', 'widersprechen', 'widersprechen',  'widersprechen', 'divergieren', 'widersprechen', 'kontern', 'abweichend', 'Meinung', 'richtig', 'wahrheitsgemäß', 'sachlich', 'präzise', 'rechtmäßig', 'vertrauenswürdig', 'unbestreitbar', 'unzweifelhaft', 'unbestreitbar', 'wahrheitsgetreu', 'wahrheitsgemäß', 'wahrheitsgemäß', 'Einschätzung', 'annahme', 'Einstellung', 'Schlussfolgerung', 'Gefühl', 'Idee', 'Eindruck', 'Urteil', 'Geist', 'Vorstellung', 'Sichtweise','Reaktion', 'Gefühl', 'Spekulation', 'Theorie', 'Gedanke', 'Ansicht', 'Sichtweise', 'Vorstellung', 'schätzung', 'Schätzung', 'Hypothese', 'Neigung', 'Inferenz', 'Überredung', 'Postulat', 'Vermutung', 'Vorannahme', 'Vermutung', 'Vermutung', 'Verdacht', 'Behauptung', 'meinen', 'zugesichert', 'klar', 'sicher', 'überzeugend', 'zweifelsfrei', 'echt', 'unbestreitbar', 'unbestreitbar', 'überzeugt', 'unerschütterlich', 'standhaft', 'unveränderlich', 'unveränderlich', 'unbestreitbar','unzweideutig', 'unfehlbar', 'unbeugsam', 'unqualifiziert', 'unanfechtbar', 'unhinterfragbar', 'unerschütterlich', 'gültig']
    reason=sum([len(e) for e in weildenn])+ sum([len(e) for e in grund_da_deshalb])
    purpose=sum([len(e) for e in damit])+um
    conditions=sum([len(e) for e in cond])
    although=sum([len(e) for e in cons])
    contrast=sum([len(e) for e in advers])
    supports=reason+purpose+conditions+although+contrast
    claims=sum([len([e.count(i) for i in e if i.lower() in clamy_words]) for e in lemma])
    matrix.append(claims)
    matrix.append(supports)
    #logical connectors
    import xml
    from xml.etree import cElementTree as ET

    tree = ET.parse('ConAnoConnectorLexicon.xml')
    root = tree.getroot()
    discourse_markers=[]
    for entry in root.findall('entry'):
        for orth in entry.findall('orth'):
            if orth.find('part').text.lower() not in [e.lower() for e in discourse_markers]:
                discourse_markers.append(orth.find('part').text)
    d=['tatsächlich' , 'möglicherweise' , 'außerdem' , 'jedoch' , 'zweifellow' , 'gleichzeitig' , ' in diesem Fall' , ' äußerst' , ' zu erst' , ' stattdessen' , '  im Grunde' , ' im Prinzip' , ' im Wirklichkeit' , '  z.B.' , ' Beispiel' , ' im Vergleich mit/zu' , ' auf der einen/anderen Seite' , ' im Gegesatz zu' , ' erstens' , ' zweitens' , ' drittens' , ' In diesem Zusammenhang' , ' zum Schluss' , ' schlißlich' , ' Kurtzgesagt' , ' kurz und bündig' , ' auf den Punkt gebracht' , ' zu beginnen mit' , ' allerdings' , ' dennoch' , ' trotzdem' , ' dann' , ' anschließend' , ' danach' , ' damals ' , ' folglich' , ' endlich' , ' endgültig ' , ' zum Schluss' , ' am Ende']
    for e in d:
        if e not in discourse_markers:
            discourse_markers.append(e.strip())
    quantity=[]
    markers=[]
    for t in texts:
        cpt=[e for e in discourse_markers if e in t]
        quantity.append(len(cpt))
        markers.append(cpt)
    par_analysis = [a + b for a, b in zip(quantity, par_analysis)]
    matrix.append(sum(quantity))
    
    #konjuktiv
    eins=[]
    zwei=[]
    konj=[]
    temp=[]
    cpt=[]
    umlaut=['ü','ï','ä','ö','ë']
    for e,p in zip(new_tokens,morph):
        for i in range(len(e)):
            if 'V' in p[i][0] and p[i][0]!='ADV':
                if p[i][-1]=='Subj':
                    temp.append(e[i])
                    find=[l for l in e[i] if l in umlaut]
                    if len(find)>0:
                        zwei.append(e[i])
                    else:
                        eins.append(e[i])

        konj.append(temp)
        cpt.append(len(temp))
        temp=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    matrix.append(sum(cpt))
    modality_words=[]
    temp=[]
    cpt=[]
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i].lower() in clamy_words or e[i] in add_modal:
                temp.append(e[i])
        modality_words.append(temp)
        cpt.append(len(temp))
        temp=[]
    #modality_words
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]            
    matrix.append(sum(cpt))
    personalizing=[]
    distancing=[]
    tempd=[]
    tempp=[]
    cpt=[]
    count=0
    d_c=0
    cpt2=[]
    for e,p in zip(new_tokens,pos):
        for i in range(len(e)):
            if e[i].lower()=='ich' or e[i].lower()=='wir' or e[i].lower()=='mir' or e[i].lower()=='uns' or e[i].lower()=='mich' or e[i][0:4]=='mein':
                count+=1
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct]
                if nextpunct!=[]:
                    nextpunct=nextpunct[0]+i
                    previouspunct=[f for f in range(len(e[:i])) if e[:i][f] in punct]
                    if previouspunct!=[]:
                        if e[previouspunct[-1]+1:nextpunct] not in tempp:
                            tempp.append(e[previouspunct[-1]+1:nextpunct])
                    else:
                        if e[:nextpunct] not in tempp:
                            tempp.append(e[:nextpunct])
            elif e[i]=='man':
                d_c+=1
                temp.append(e[i])
                nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct]
                previouspunct=[f for f in range(len(e[:i])) if e[:i][f] in punct]
                if previouspunct!=[]:
                    if nextpunct!=[]:
                        nextpunct=nextpunct[0]+i
                        if e[previouspunct[-1]+1:nextpunct] not in tempd:
                            tempd.append(e[previouspunct[-1]+1:nextpunct])
                        
                else:
                    if nextpunct!=[]:
                        nextpunct=nextpunct[0]+i
                        
                        if e[:nextpunct] not in tempd:
                            tempd.append(e[:nextpunct])
                    else:
                        if e[i:] not in tempd:
                            tempd.append(e[i:])
                        

        personalizing.append(tempp)
        distancing.append(tempd)
        cpt.append(len(tempp))
        cpt2.append(len(tempd))
        tempp=[]
        tempd=[]
    par_analysis = [a + b for a, b in zip(cpt, par_analysis)]
    par_analysis = [b - a for a, b in zip(cpt2, par_analysis)]
    matrix.append(sum(cpt))
    #'perosonalizing'
    matrix.append(sum(cpt2))
    #distansing'
    werde=['werde','wirst','wird','werden','werdet','werden']
    #tenses
    past=0
    present=0
    future=0
    for e,p in zip(new_tokens,morph):
            for i in range(len(e)):
                if e[i].lower() in werde and 'VINF' not in p[i]:
                    nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct]
                    if nextpunct!=[]:
                        nextpunct=nextpunct[0]+i
                        if 'VINF' in p[nextpunct-1]:
                            future+=1
                elif 'Past' in p[i]:
                    past+=1
                elif 'Haben' in p[i] or 'Sein' in p[i]:
                    if e[i].lower()!='sein' and 'VINF' not in p[i]:
                        nextpunct=[f for f in range(len(e[i:])) if e[i:][f] in punct]
                        if nextpunct!=[]:
                            nextpunct=nextpunct[0]+i
                            if 'VPP' in p[nextpunct-1] and 'Psp' in p[nextpunct-1]:
                                past+=1
                elif 'Pres' in p[i] and 'Mod' not in p[i]:
                    present+=1
    matrix.append(present)
    matrix.append(future)
    matrix.append(past)
    matrix.append(labels[file])
    matrix.append(sentiment[file])
    ff.append(matrix)
    if cptthem % 100 == 0:
        print(cptthem)
    cptthem+=1
print(len(ff))
meta=['\t'.join(e[0:-2]) for e in pathlist[1:]]
print(meta[0].split('\t'))
with open(meta[0].split('\t')[0]+'/matrix_output'+meta[0].split('\t')[0]+'.tsv','w') as out:
        out.write('coll_lecture	coll_session	subj	text_id	sentence\t')
        for e in header:
            out.write(e+'\t')
        out.write('\n')
        for e in range (len(ff)):
            out.write(meta[e]+'\t'+'\t'.join([str(i) for i in ff[e]])+'\n')
        out.write('\n')
