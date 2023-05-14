import MeCab
import numpy as np
import copy

def split_text(texts):
    splits = []
    for t_ in texts:
        if t_ == '。':
            splits.append(t_)

    # texts = texts.replace('、', '。')
    texts = texts.split('。')[:-1]
    return texts, splits



def keitaiso(text, mt, cut_position):
    #入力：len_limitより長いテキスト、MeCab.Tagger("-Ochasen")、切りたいindex
    len_ = 0
    inserts = []
    lines = mt.parse(text).splitlines()
    res = [line.split('\t') for line in lines][:-1]
    words = [line[0] for line in res]
    keitais = [line[3] for line in res]
    lengths = []
    for word in words:
        len_ = len_ + len(word)
        lengths.append(len_)
    lengths_ = copy.deepcopy(lengths)
    #words, keitais, lengthsを取得

    for pos in cut_position:
        first_index = np.argmin(np.abs(np.array(lengths)-pos))
        index = copy.deepcopy(first_index)
        flag = False
        while flag == False:
            flag = keitais[index][:2] == '名詞' and keitais[index-1][:2] != '名詞'
            if flag:
                break
            index = index - 1
            if index < 5:
                index = first_index
                break
        print("index:",index)
        inserts.append(index)
    for i,index_ in enumerate(inserts):
        words.insert(index_+i, '、')
    print("words:",words)

    return ''.join(words)

def split_text_check_length(texts, len_limit=-1):
    if len_limit == -1:
        len_limit = 1<<30
    import ipadic
    CHASEN_ARGS = r' -F "%m\t%f[7]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n"'
    CHASEN_ARGS += r' -U "%m\t%m\t%m\t%F-[0,1,2,3]\t\t\n"'
    mt = MeCab.Tagger(ipadic.MECAB_ARGS + CHASEN_ARGS)
    #mt = MeCab.Tagger("")
    texts, splits = split_text(texts)
    #上の時点で空になっているリストの要素を削除
    texts = [text for text in texts if text != ""]
    texts_length = len(texts)
    #長さ合わせる
    splits = splits[:texts_length]
    res_texts = []
    res_splits = []
    res_sentences = []
    for i, text in enumerate(texts):
        len_ = 0
        if len(text) < len_limit:#長さを超えてないなら
            res_texts.append(texts[i])
            res_splits.append(splits[i])
            res_sentences.append(texts[i])
        else:#超えたらlen_limitの倍数に分ける
            text = text.replace('、', ',')
            split_to = int(len(text)/len_limit+(len(text)%len_limit!=0))
            cut_position = [len(text)//split_to*i for i in range(1, split_to)]
            text__ = keitaiso(text, mt, cut_position)
            res_sentences.append(text__)
            text = text__.split('、')
            for text_ in text:
                text_ = text_.replace(',', '、')
                res_texts.append(text_)
                res_splits.append('、')

    return res_texts, res_splits, res_sentences


def process_conmma(text_list):
    new_list = []
    for text, pitch, taco_text, japa, hinshi in text_list:
        process_text = ""
        process_pitch = ""
        process_taco = ""
        flg = False
        for text_str, pitch_str, taco_str in zip(text, pitch, taco_text):
            if text_str != "、" and text_str != "。":
                process_text += text_str
                process_pitch += pitch_str
                process_taco += taco_str
            else:
                flg = True
                process_text += "。"
                process_pitch += pitch_str
                process_taco += "。"
                if process_text[0] == "*":
                    new_list.append((process_text[1:], process_pitch[1:], process_taco[1:], japa, hinshi))
                else:
                    new_list.append((process_text, process_pitch, process_taco, japa, hinshi))
                process_text = ""
                process_pitch = ""
                process_taco = ""
        if flg == False:
            new_list.append((text, pitch, taco_text, japa, hinshi))
    return new_list


if __name__=='__main__':
    #input_text = '陪審制は刑事訴訟や民事訴訟の審理に際して民間から無作為で選ばれた陪審員によって構成される合議体が評議によって事実認定を行う司法制度である陪審員の人数は6～12名である場合が多くその合議体を「陪審」という陪審は刑事事件では原則として被告人の有罪・無罪について民事事件では被告の責任の有無や損害賠償額等について判断する。'
    input_text = "東パキスタンのように、西パキスタンの言語であるウルドゥー語の公用語化に反発してベンガル語を同格の国語とすることを求めたことから独立運動が起き、最終的にバングラデシュとして独立したような例もある。おはようございます。"
    # input_list = [('こんにちわ', '45665', 'こんにちわ', 'コンニチハ'), ('ひがしぱきすたん*の*よう*に*、*にし*ぱきすたん*の*げんご*で*ある*うるどぅーご*の*こうようご*か*に*はんぱつ*し*て*べんがるご*を*、*どうかく*の*こくご*と*する*こと*を*もとめ*た*こと*から*どくりつうんどう*が*おき*、*さいしゅうてき*に*ばんぐらでしゅ*として*どくりつ*し*た*よう*な*れい*も*ある*。', '47787764*4*43*2*3*47*88764*4*454*4*32*357665*5*46877*6*5*4555*4*4*34433*2*3*4688*7*667*7*66*66*4*356*4*44*32*35776643*2*22*3*4566655*5*4577764*332*3478*7*7*77*4*33*2*22*2', 'ひがしぱきすたん*の*よう*に*、*にし*ぱきすたん*の*げんご*で*ある*うるどぅーご*の*こうようご*か*に*はんぱつ*し*て*べんがるご*を*、*どうかく*の*こくご*と*する*こと*を*もとめ*た*こと*から*どくりつうんどう*が*おき*、*さいしゅうてき*に*ばんぐらでしゅ*として*どくりつ*し*た*よう*な*れい*も*ある*。', '東パキスタン*の*よう*に*、*西*パキスタン*の*言語*で*ある*ウルドゥー語*の*公用語*化*に*反発*し*て*ベンガル語*を*、*同格*の*国語*と*する*こと*を*求め*た*こと*から*独立運動*が*起き*、*最終的*に*バングラデシュ*として*独立*し*た*よう*な*例*も*ある*。')]
    # print(input_list)
    print(input_text)
    texts, splits, sentences = split_text_check_length(input_text,60)
    print(texts)
    print(splits)
    print(sentences)
    # new_list = process(input_list)
    # print(new_list)
    #print(texts)
