import math
import pandas as pd #✨중요 pandas 다운로드 받아야함 ✨중요

def main():

    data = pd.read_csv("tmdb_5000_movies.csv", encoding = "CP949")
    #data 갯수 4799개

    genres = data.pop("genres") #장르(여러개로 되어있음)
    keywords = data.pop("keywords") #키워드(여러개로 되어있음)
    ori_lan = data.pop("original_language") #언어
    popularity = data.pop("popularity") #인기도
    title = data.pop("title") #제목
    vote_aver = data.pop("vote_average") #평균 평점
    vote_count = data.pop("vote_count") #평점 개수

    print(title)

    #데이터 형태 수정
    for i in range(len(genres)):
        genres[i] = modify_data(genres[i])
        keywords[i] = modify_data(keywords[i])

    #좋아하는 영화 입력받음
    favorite_movies = []
    for i in range(10):
        while(True):
            mov = input("재미있게 보았던 영화를 입력하세요{}:".format(i))
            if mov in title.tolist():
                break
            print("다시 입력하세요:")
        favorite_movies.append(mov)
    fgen_dict, fkw_dict, flan_dict = create_BOW(favorite_movies, title.tolist(), [genres.tolist(),keywords.tolist(),ori_lan.tolist()])

    #싫어하는 영화 입력받음
    boring_movies = []
    for i in range(10):
        while(True):
            mov = input("지루하게 보았던 영화를 입력하세요{}:".format(i))
            if mov in title.tolist():
                break
            print("다시 입력하세요:")
        boring_movies.append(mov)
    bgen_dict, bkw_dict, blan_dict = create_BOW(boring_movies, title.tolist(), [genres.tolist(),keywords.tolist(),ori_lan.tolist()])

    movies_index = []
    movies_prob = []

    for i in range(len(genres)):

        if (title[i] in favorite_movies) or (title[i] in boring_movies):
            continue

        # 점수 비교
        alpha = 0.1
        ori_prob = 0.5
        fav_prob = naive_bayes([genres[i], keywords[i], ori_lan[i]], fgen_dict, fkw_dict, flan_dict, alpha, ori_prob)
        bor_prob = naive_bayes([genres[i], keywords[i], ori_lan[i]], bgen_dict, bkw_dict, blan_dict, alpha, ori_prob)
        
        prob = normalize_log_prob(fav_prob,bor_prob)

        movies_index.append(i)
        movies_prob.append(prob)

    print(movies_prob[0:10])
    print(movies_index[0:10])
    
    for i in range(5):
        index = movies_prob.index(max(movies_prob))
        print(title[movies_index[index]], movies_prob[index] * 100)
        movies_prob.remove(max(movies_prob))
        movies_index.pop(index)


def naive_bayes(features, gen_dict, kw_dict, lan_dict, alpha, prob):
    
    logprob = 0

    num_tokens_training = 0
    for gen in gen_dict:
        num_tokens_training += gen_dict[gen]
    for kw in kw_dict:
        num_tokens_training += kw_dict[kw]
    for lan in lan_dict:
        num_tokens_training += lan_dict[lan]
        
    num_ft_training = len(gen_dict) + len(kw_dict) + len(lan_dict)

    genre = features[0]
    keyword = features[1]
    lan = features[2]

    for gen in genre:
        if gen in gen_dict:
            logprob += math.log(gen_dict[gen] + alpha)
        else:
            logprob += math.log(alpha)

    for kw in keyword:
        if kw in kw_dict:
            logprob += math.log(kw_dict[kw] + alpha)
        else:
            logprob += math.log(alpha)

    if lan in lan_dict:
        logprob += math.log(lan_dict[lan] + alpha)
    else:
        logprob += math.log(alpha)
   
    logprob -= math.log(num_tokens_training + num_ft_training * alpha)

    return logprob

def normalize_log_prob(prob1,prob2):
    maxprob = max(prob1,prob2)

    prob1 -= maxprob
    prob2 -= maxprob
    
    prob1 = math.exp(prob1)
    prob2 = math.exp(prob2)

    normalize_constant = 1.0 / float(prob1 + prob2)
    prob1 *= normalize_constant

    return (prob1)

def modify_data(list_df):
    list_df = list_df.split(",")
    new_df = []
    for i in range(len(list_df)):
        if i%2 == 1:
            list_df[i] = list_df[i].lstrip(' "name": "').rstrip('"}]')
            new_df.append(list_df[i])
    return new_df

def create_BOW(movies, title, features):
    gen_dict, kw_dict, lan_dict = {}, {}, {}
    
    for m in movies:
        mindex = title.index(m)
        genre = features[0][mindex]
        keyword = features[1][mindex]
        lan = features[2][mindex]

        for gen in genre:
            if gen not in gen_dict:
                gen_dict[gen] = 1
            else :
                gen_dict[gen] += 1
                
        for kw in keyword:
            if kw not in kw_dict:
                kw_dict[kw] = 1
            else :
                kw_dict[kw] += 1 
                
        for lang in lan:
            if lang not in lan_dict:
                lan_dict[lang] = 1
            else :
                lan_dict[lang] += 1
            
        
    return gen_dict, kw_dict, lan_dict

if __name__ == "__main__":
    main()
